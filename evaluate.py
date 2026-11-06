"""Автоматичне тестування системи на наборі з відомою правильною відповіддю.

    python evaluate.py              # повне тестування (~1–2 хв)
    python evaluate.py --quick      # без перебору параметрів

Що рахується:
  1. 15 тестових випадків × 3 детектори (ORB, AKAZE, SIFT):
     чи знайдено, кількість inliers, середня похибка кутів (px) відносно еталону, час.
     Випадок «пройдено», якщо результат збігається з очікуванням і похибка <= 5 px.
  2. Перехресні негативні пари: кожен зразок шукається на сценах, де його НЕМАЄ.
     Будь-яке «знайдено» тут — хибне спрацювання (false positive).
  3. Перебір параметрів ORB (поріг Лоу × мін. inliers): компроміс між
     кількістю знайдених об'єктів і кількістю хибних спрацювань.

Результати: results/evaluation.md, results/evaluation.csv, results/cases/<детектор>/*.jpg
"""

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

from finder import MatchParams, ObjectFinder, load_image
from finder.visualize import draw_detections

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results")
DETECTORS = ["ORB", "AKAZE", "SIFT"]
ERR_OK = 5.0   # px — допустима середня похибка кутів


def corner_error(res, objects):
    """Для кожного знайденого екземпляра — похибка до найближчого справжнього об'єкта."""
    errs = []
    for d in res.detections:
        errs.append(min(float(np.linalg.norm(d.corners - np.float32(o["corners"]), axis=1).mean())
                        for o in objects) if objects else float("inf"))
    return errs


def verdict(case, res, errs):
    exp = case["expected"]
    if exp == "not_found":
        return not res.found
    need = 2 if exp == "found x2" else 1
    return len(res.detections) >= need and all(e <= ERR_OK for e in errs[:need])


def run_cases(gt, save_images=True):
    rows = []
    cache = {}
    for det in DETECTORS:
        for cid, c in gt.items():
            params = MatchParams(detector=det, max_instances=2 if c["expected"] == "found x2" else 1)
            key = (det, c["template"], params.max_instances)
            if key not in cache:
                f = ObjectFinder(params)
                f.set_template(load_image(os.path.join(DATA, c["template"])))
                cache[key] = f
            f = cache[key]
            scene = load_image(os.path.join(DATA, c["scene"]))
            res = f.find(scene)
            errs = corner_error(res, c["objects"])
            ok = verdict(c, res, errs)
            rows.append({
                "case": cid, "detector": det, "category": c["category"], "description": c["description"],
                "expected": c["expected"], "found": res.found, "instances": len(res.detections),
                "kp_template": res.n_kp_template, "kp_scene": res.n_kp_scene, "good_matches": res.n_good_matches,
                "inliers": res.detections[0].inliers if res.found else 0,
                "corner_err_px": round(max(errs), 2) if errs else "",
                "time_ms": round(res.timings_ms.get("total", 0), 1), "pass": ok, "reason": res.reason,
            })
            if save_images:
                d = os.path.join(OUT, "cases", det)
                os.makedirs(d, exist_ok=True)
                gtc = c["objects"][0]["corners"] if c["objects"] else None
                cv2.imwrite(os.path.join(d, f"{cid}.jpg"), draw_detections(scene, res, gtc),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  {det}: готово")
    return rows


def negative_pairs(gt):
    """Усі пари (зразок, сцена), де зразка на сцені немає."""
    templates = sorted(f"templates/{t}" for t in os.listdir(os.path.join(DATA, "templates")))
    pairs = []
    for t in templates:
        for cid, c in gt.items():
            if t not in c.get("present_templates", []):
                pairs.append((t, c["scene"]))
    return templates, pairs


def run_negatives(gt, det, params_kw=None, scenes_cache=None):
    templates, pairs = negative_pairs(gt)
    p = MatchParams(detector=det, **(params_kw or {}))
    finders = {}
    fp = []
    for t, s in pairs:
        if t not in finders:
            finders[t] = ObjectFinder(p)
            finders[t].set_template(load_image(os.path.join(DATA, t)))
        scene = scenes_cache[s]
        r = finders[t].find(scene)
        if r.found:
            fp.append((t, s, r.detections[0].inliers))
    return len(pairs), fp


SWEEP_CONFIGS = (
    [dict(ratio=r, min_inliers=mi) for r in (0.65, 0.75, 0.85) for mi in (8, 15, 30)]
    + [dict(min_coverage=c) for c in (0.0, 0.15, 0.3, 0.45, 0.6)]
)


def sweep(gt, scenes_cache):
    """Перебір параметрів ORB: (поріг Лоу × мін. inliers) і поріг покриття."""
    out = []
    positives = {cid: c for cid, c in gt.items() if c["expected"] != "not_found"}
    for kw in SWEEP_CONFIGS:
        if True:
            passed = 0
            for cid, c in gt.items():
                p = MatchParams(detector="ORB", max_instances=2 if c["expected"] == "found x2" else 1, **kw)
                f = ObjectFinder(p)
                f.set_template(load_image(os.path.join(DATA, c["template"])))
                r = f.find(scenes_cache[c["scene"]])
                if c["expected"] != "not_found" and verdict(c, r, corner_error(r, c["objects"])):
                    passed += 1
            n_neg, fp = run_negatives(gt, "ORB", kw, scenes_cache)
            full = {**{k: getattr(MatchParams(), k) for k in ("ratio", "min_inliers", "min_coverage")}, **kw}
            out.append({**full, "varied": "coverage" if "min_coverage" in kw else "ratio",
                        "positives_passed": passed, "positives_total": len(positives),
                        "false_positives": len(fp), "negatives_total": n_neg})
            print(f"  {kw}: знайдено {passed}/{len(positives)}, FP {len(fp)}/{n_neg}")
    return out


def write_report(rows, negs, sweep_rows, elapsed):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "evaluation.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by = {(r["case"], r["detector"]): r for r in rows}
    cases = list(dict.fromkeys(r["case"] for r in rows))
    L = ["# Результати тестування", "",
         f"Згенеровано `evaluate.py` за {elapsed:.0f} с. Критерій успіху: результат збігається з очікуванням, "
         f"а середня похибка кутів ≤ {ERR_OK:.0f} px. Параметри — за замовчуванням "
         "(ratio 0.75, RANSAC 5 px, мін. 15 inliers, покриття ≥ 0.3).", "",
         "## 1. Тестові випадки", "",
         "| Випадок | Опис | Очікування | " + " | ".join(DETECTORS) + " |",
         "|---|---|---|" + "---|" * len(DETECTORS)]
    for cid in cases:
        r0 = by[(cid, DETECTORS[0])]
        cells = []
        for d in DETECTORS:
            r = by[(cid, d)]
            mark = "✅" if r["pass"] else "❌"
            if r["found"]:
                cells.append(f"{mark} {r['inliers']} inl., {r['corner_err_px']} px")
            else:
                cells.append(f"{mark} не знайдено")
        exp = {"found": "знайти", "found x2": "знайти 2", "not_found": "не знайти"}[r0["expected"]]
        L.append(f"| `{cid}` | {r0['description']} | {exp} | " + " | ".join(cells) + " |")

    L += ["", "**Підсумок за детекторами:**", "",
          "| Детектор | Пройдено | Сер. похибка (знайдені), px | Сер. час на сцену, мс |", "|---|---|---|---|"]
    for d in DETECTORS:
        rs = [r for r in rows if r["detector"] == d]
        errs = [r["corner_err_px"] for r in rs if r["corner_err_px"] != "" and r["pass"]]
        L.append(f"| {d} | {sum(r['pass'] for r in rs)} / {len(rs)} | "
                 f"{np.mean(errs):.2f} | {np.mean([r['time_ms'] for r in rs]):.0f} |")

    L += ["", "## 2. Хибні спрацювання (перехресні негативні пари)", "",
          "Кожен із 7 зразків шукається на кожній сцені, де його немає.", "",
          "| Детектор | Пар перевірено | Хибних «знайдено» |", "|---|---|---|"]
    for d, (n, fp) in negs.items():
        L.append(f"| {d} | {n} | {len(fp)} |")
        for t, s, inl in fp:
            L.append(f"|  | ↳ `{t}` на `{s}` | {inl} inliers |")

    if sweep_rows:
        L += ["", "## 3. Вплив параметрів (ORB)", "",
              "Компроміс: скільки об'єктів знайдено (з 14 позитивних випадків, включно з проблемними) "
              "і скільки хибних спрацювань на 90 негативних парах.", "",
              "**3а. Поріг тесту Лоу × мін. inliers** (покриття = 0.3)", "",
              "| ratio | мін. inliers | Знайдено | Хибні спрацювання |", "|---|---|---|---|"]
        for s in [r for r in sweep_rows if r["varied"] == "ratio"]:
            L.append(f"| {s['ratio']} | {s['min_inliers']} | {s['positives_passed']} / {s['positives_total']} | "
                     f"{s['false_positives']} / {s['negatives_total']} |")
        L += ["", "**3б. Поріг покриття зразка inliers** (ratio 0.75, мін. inliers 15)", "",
              "| мін. покриття | Знайдено | Хибні спрацювання |", "|---|---|---|"]
        for s in [r for r in sweep_rows if r["varied"] == "coverage"]:
            L.append(f"| {s['min_coverage']} | {s['positives_passed']} / {s['positives_total']} | "
                     f"{s['false_positives']} / {s['negatives_total']} |")

    L += ["", "Зображення з результатами: `results/cases/<детектор>/<випадок>.jpg` "
          "(блакитна тонка лінія — еталон, кольорова — знайдена рамка)."]
    with open(os.path.join(OUT, "evaluation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="пропустити перебір параметрів")
    a = ap.parse_args()
    t0 = time.time()
    with open(os.path.join(DATA, "ground_truth.json"), encoding="utf-8") as f:
        gt = json.load(f)
    scenes_cache = {c["scene"]: load_image(os.path.join(DATA, c["scene"])) for c in gt.values()}

    print("1) Тестові випадки")
    rows = run_cases(gt)
    print("2) Перехресні негативні пари")
    negs = {}
    for d in DETECTORS:
        negs[d] = run_negatives(gt, d, None, scenes_cache)
        print(f"  {d}: хибних спрацювань {len(negs[d][1])} з {negs[d][0]}")
    sw = []
    if not a.quick:
        print("3) Перебір параметрів ORB")
        sw = sweep(gt, scenes_cache)
    write_report(rows, negs, sw, time.time() - t0)

    for d in DETECTORS:
        rs = [r for r in rows if r["detector"] == d]
        print(f"{d}: пройдено {sum(r['pass'] for r in rs)}/{len(rs)}")
    print("Звіт: results/evaluation.md")


if __name__ == "__main__":
    main()
