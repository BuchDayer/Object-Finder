"""Пошук об'єкта на кожному кадрі відео.

Приклади:
    python find_in_video.py data/templates/book_astronaut.png data/video/demo_input.mp4
    python find_in_video.py зразок.jpg відео.mp4 --show
    python find_in_video.py зразок.jpg 0 --show          # 0 = вебкамера (необов'язково)

Дескриптори зразка обчислюються ОДИН раз, далі на кожному кадрі — лише детекція,
зіставлення і RANSAC. Опційне згладжування кутів (--smooth) зменшує «тремтіння» рамки.
"""

import argparse
import os
import time

import cv2
import numpy as np

from finder import MatchParams, ObjectFinder, load_image
from finder.visualize import draw_detections, _label


def main():
    ap = argparse.ArgumentParser(description="Пошук об'єкта-зразка на відео")
    ap.add_argument("template")
    ap.add_argument("source", help="шлях до відео або номер камери (0)")
    ap.add_argument("--detector", default="ORB", choices=["ORB", "AKAZE", "SIFT"])
    ap.add_argument("--features", type=int, default=2000)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--min-inliers", type=int, default=15)
    ap.add_argument("--smooth", type=float, default=0.5, help="0 = без згладжування, 0.9 = сильне")
    ap.add_argument("--out", default="results/video_result.mp4")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    finder = ObjectFinder(MatchParams(detector=a.detector, n_features=a.features, ratio=a.ratio,
                                      min_inliers=a.min_inliers, max_scene_side=960))
    finder.set_template(load_image(a.template))

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Не вдалося відкрити джерело: {a.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = None
    prev = None
    n = found = 0
    t_sum = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        res = finder.find(frame)
        dt = time.perf_counter() - t0
        t_sum += dt
        n += 1

        if res.found:
            found += 1
            d = res.detections[0]
            # експоненційне згладжування координат кутів між кадрами
            if prev is not None and a.smooth > 0 and np.abs(d.corners - prev).max() < 80:
                d.corners = a.smooth * prev + (1 - a.smooth) * d.corners
            prev = d.corners.copy()
        else:
            prev = None

        vis = draw_detections(frame, res)
        _label(vis, f"frame {n}  {1 / dt:.0f} FPS  {res.n_good_matches} good matches", (10, vis.shape[0] - 12),
               (255, 255, 255), 0.6, 1)

        if writer is None and a.out:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vis.shape[1], vis.shape[0]))
        if writer is not None:
            writer.write(vis)
        if a.show:
            cv2.imshow("video (q - exit)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if n:
        print(f"Кадрів: {n}, об'єкт знайдено на {found} ({found / n:.0%}), "
              f"середній час на кадр {1000 * t_sum / n:.1f} мс (~{n / t_sum:.0f} FPS)")
        if a.out:
            print(f"Результат: {a.out}")


if __name__ == "__main__":
    main()
