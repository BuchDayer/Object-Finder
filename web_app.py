"""Локальний веб-інтерфейс (без додаткових бібліотек — лише стандартний http.server).

Запуск:
    python web_app.py            # відкриє http://127.0.0.1:8018 у браузері
    python web_app.py --port 9000 --no-browser

У браузері можна:
  * вибрати готовий тестовий випадок або завантажити власні фото (зразок + сцена);
  * змінювати параметри алгоритму повзунками — результат перераховується одразу;
  * дивитися проміжні етапи: ключові точки, пари після тесту Лоу, inliers після RANSAC;
  * порівняти ORB / AKAZE / SIFT на одній парі зображень.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import cv2
import numpy as np

from finder import MatchParams, ObjectFinder
from finder.matcher import params_to_dict
from finder.visualize import draw_detections, draw_keypoints, draw_matches

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
WEB = os.path.join(ROOT, "web")


def _decode_image(spec: dict) -> np.ndarray:
    """spec = {"sample": "templates/x.png"} або {"data": "data:image/png;base64,...."}"""
    if spec.get("sample"):
        path = os.path.normpath(os.path.join(DATA, spec["sample"]))
        if not path.startswith(DATA):
            raise ValueError("недозволений шлях")
        buf = np.fromfile(path, np.uint8)
    elif spec.get("data"):
        b64 = spec["data"].split(",", 1)[-1]
        buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    else:
        raise ValueError("зображення не передано")
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("не вдалося декодувати зображення")
    return img


def _to_b64(img: np.ndarray, max_side=1400, quality=85) -> str:
    s = max(img.shape[:2])
    if s > max_side:
        img = cv2.resize(img, None, fx=max_side / s, fy=max_side / s, interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()


def _gt_error(res, case_id):
    """Похибка кутів відносно еталону (лише для синтетичних тестів)."""
    if not case_id:
        return None
    try:
        with open(os.path.join(DATA, "ground_truth.json"), encoding="utf-8") as f:
            gt = json.load(f)[case_id]
    except Exception:
        return None
    if not gt["objects"] or not res.detections:
        return None
    errs = []
    for d in res.detections:
        errs.append(min(float(np.linalg.norm(d.corners - np.float32(o["corners"]), axis=1).mean())
                        for o in gt["objects"]))
    return [round(e, 2) for e in errs]


def run_find(body: dict) -> dict:
    template = _decode_image(body["template"])
    scene = _decode_image(body["scene"])
    params = MatchParams.from_dict(body.get("params", {}))
    finder = ObjectFinder(params)
    finder.set_template(template)
    res = finder.find(scene)
    out = res.summary()
    out["params"] = params_to_dict(params)
    out["gt_error_px"] = _gt_error(res, body.get("case_id"))
    out["images"] = {
        "result": _to_b64(draw_detections(scene, res)),
        "inliers": _to_b64(draw_matches(template, scene, res, inliers_only=True)) if res.found else None,
        "good": _to_b64(draw_matches(template, scene, res, inliers_only=False)),
        "kp_template": _to_b64(draw_keypoints(finder.tmpl_work, res.kp_template), 700),
        "kp_scene": _to_b64(draw_keypoints(scene, res.kp_scene, res.scene_scale)),
    }
    return out


def run_compare(body: dict) -> dict:
    template = _decode_image(body["template"])
    scene = _decode_image(body["scene"])
    rows = []
    for det in ["ORB", "AKAZE", "SIFT"]:
        p = MatchParams.from_dict({**body.get("params", {}), "detector": det})
        try:
            f = ObjectFinder(p)
            f.set_template(template)
            r = f.find(scene)
        except Exception as e:  # напр. AKAZE недоступний
            rows.append({"detector": det, "error": str(e)})
            continue
        rows.append({"detector": det, "found": r.found, "instances": len(r.detections),
                     "kp_template": r.n_kp_template, "kp_scene": r.n_kp_scene, "good": r.n_good_matches,
                     "inliers": r.detections[0].inliers if r.found else 0,
                     "time_ms": round(r.timings_ms.get("total", 0), 1), "reason": r.reason,
                     "gt_error_px": _gt_error(r, body.get("case_id"))})
    return {"rows": rows}


def list_samples() -> dict:
    out = {"templates": sorted(os.listdir(os.path.join(DATA, "templates"))),
           "scenes": sorted(os.listdir(os.path.join(DATA, "scenes"))), "cases": []}
    try:
        with open(os.path.join(DATA, "ground_truth.json"), encoding="utf-8") as f:
            gt = json.load(f)
        for cid, c in gt.items():
            out["cases"].append({"id": cid, "scene": c["scene"], "template": c["template"],
                                 "description": c["description"], "expected": c["expected"],
                                 "category": c["category"]})
    except FileNotFoundError:
        pass
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # тихіший лог
        pass

    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        path = unquote(self.path.split("?")[0])
        if path in ("/", "/index.html"):
            with open(os.path.join(WEB, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/samples":
            return self._json(list_samples())
        if path.startswith("/data/"):
            fp = os.path.normpath(os.path.join(DATA, path[len("/data/"):]))
            if fp.startswith(DATA) and os.path.isfile(fp):
                ext = os.path.splitext(fp)[1].lower()
                ctype = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "application/octet-stream")
                with open(fp, "rb") as f:
                    return self._send(200, f.read(), ctype)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            if self.path == "/api/find":
                return self._json(run_find(body))
            if self.path == "/api/compare":
                return self._json(run_compare(body))
            self._json({"error": "unknown endpoint"}, 404)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8018)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}/"
    print(f"Веб-інтерфейс запущено: {url}   (зупинити — Ctrl+C)")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nЗупинено.")


if __name__ == "__main__":
    main()
