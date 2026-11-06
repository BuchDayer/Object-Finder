"""Пошук об'єкта на зображенні з командного рядка.

Приклади:
    python find_object.py data/templates/book_astronaut.png data/scenes/01_frontal.jpg
    python find_object.py зразок.jpg сцена.jpg --detector SIFT --ratio 0.8 --show
    python find_object.py data/templates/pack_tea.png data/scenes/09_two_copies.jpg --instances 2

Результати зберігаються в теку results/cli/ (рамка на сцені + лінії відповідностей).
"""

import argparse
import json
import os

import cv2

from finder import MatchParams, ObjectFinder, load_image
from finder.visualize import draw_detections, draw_matches


def build_parser():
    p = argparse.ArgumentParser(description="Пошук об'єкта за фотографією-зразком (ORB + RANSAC + гомографія)")
    p.add_argument("template", help="фото-зразок (обкладинка, плакат, упаковка)")
    p.add_argument("scene", help="зображення, на якому шукаємо")
    d = MatchParams()
    p.add_argument("--detector", default=d.detector, choices=["ORB", "AKAZE", "SIFT"])
    p.add_argument("--features", type=int, default=d.n_features, help="макс. кількість ключових точок")
    p.add_argument("--ratio", type=float, default=d.ratio, help="поріг тесту Лоу (0.6–0.9)")
    p.add_argument("--cross-check", action="store_true", help="взаємна перевірка пар")
    p.add_argument("--ransac", default=d.ransac_method, choices=["RANSAC", "MAGSAC"])
    p.add_argument("--ransac-thresh", type=float, default=d.ransac_thresh, help="поріг перепроєкції, px")
    p.add_argument("--min-inliers", type=int, default=d.min_inliers)
    p.add_argument("--min-coverage", type=float, default=d.min_coverage, help="мін. покриття зразка inliers (0..1)")
    p.add_argument("--clahe", action="store_true", help="вирівнювання контрасту (темні фото)")
    p.add_argument("--instances", type=int, default=1, help="скільки копій об'єкта шукати")
    p.add_argument("--out", default="results/cli", help="тека для результатів")
    p.add_argument("--show", action="store_true", help="показати вікно з результатом")
    p.add_argument("--json", action="store_true", help="вивести результат у форматі JSON")
    return p


def main():
    a = build_parser().parse_args()
    params = MatchParams(detector=a.detector, n_features=a.features, ratio=a.ratio, cross_check=a.cross_check,
                         ransac_method=a.ransac, ransac_thresh=a.ransac_thresh, min_inliers=a.min_inliers, min_coverage=a.min_coverage,
                         use_clahe=a.clahe, max_instances=a.instances)
    template, scene = load_image(a.template), load_image(a.scene)

    finder = ObjectFinder(params)
    finder.set_template(template)
    res = finder.find(scene)

    if a.json:
        print(json.dumps(res.summary(), ensure_ascii=False, indent=2))
    else:
        print(f"Детектор: {params.detector} | ratio={params.ratio} | RANSAC поріг={params.ransac_thresh}px")
        print(f"Ключові точки: зразок={res.n_kp_template}, сцена={res.n_kp_scene}")
        print(f"Пари: k-NN={res.n_raw_matches} -> після тесту Лоу={res.n_good_matches}")
        print(("ЗНАЙДЕНО: " if res.found else "НЕ ЗНАЙДЕНО: ") + res.reason)
        for i, d in enumerate(res.detections, 1):
            print(f"  #{i}: inliers {d.inliers}/{d.candidates}, покриття {d.coverage:.2f}, кути: "
                  + ", ".join(f"({x:.0f},{y:.0f})" for x, y in d.corners))
        print("Час, мс: " + ", ".join(f"{k}={v:.1f}" for k, v in res.timings_ms.items()))

    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.scene))[0]
    vis = draw_detections(scene, res)
    mat = draw_matches(template, scene, res, inliers_only=res.found)
    p1, p2 = os.path.join(a.out, f"{name}_result.jpg"), os.path.join(a.out, f"{name}_matches.jpg")
    cv2.imwrite(p1, vis)
    cv2.imwrite(p2, mat)
    if not a.json:
        print(f"Збережено: {p1}, {p2}")

    if a.show:
        cv2.imshow("result (any key to close)", vis if max(vis.shape) < 1400 else cv2.resize(vis, None, fx=0.6, fy=0.6))
        cv2.imshow("matches", mat if mat.shape[1] < 1600 else cv2.resize(mat, None, fx=0.6, fy=0.6))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
