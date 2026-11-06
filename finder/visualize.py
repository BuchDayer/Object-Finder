"""Візуалізація етапів алгоритму.

Підписи на зображеннях — латиницею, бо cv2.putText не підтримує кирилицю.
"""

from __future__ import annotations

import cv2
import numpy as np

from .matcher import MatchResult, _resize_max_side

COLORS = [(60, 220, 60), (0, 170, 255), (255, 120, 0), (200, 60, 255), (0, 230, 230)]


def _label(img, text, org, color=(255, 255, 255), scale=0.7, thick=2):
    """Текст з темною підкладкою, щоб читався на будь-якому фоні."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    x = max(0, min(x, img.shape[1] - tw - 6))
    y = max(th + 6, min(y, img.shape[0] - 4))
    cv2.rectangle(img, (x - 3, y - th - 6), (x + tw + 3, y + base - 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_detections(scene_bgr: np.ndarray, res: MatchResult, gt_corners=None) -> np.ndarray:
    """Малює рамки знайдених об'єктів на сцені (у вихідному розмірі)."""
    out = scene_bgr.copy()
    lw = max(2, int(round(max(out.shape[:2]) / 400)))
    if gt_corners is not None:
        cv2.polylines(out, [np.int32(gt_corners).reshape(-1, 1, 2)], True, (255, 255, 0), 1, cv2.LINE_AA)
    for i, d in enumerate(res.detections):
        c = COLORS[i % len(COLORS)]
        pts = np.int32(np.round(d.corners)).reshape(-1, 1, 2)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], c)
        out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
        cv2.polylines(out, [pts], True, c, lw + 1, cv2.LINE_AA)
        # лівий верхній кут зразка — щоб було видно орієнтацію
        cv2.circle(out, tuple(pts[0, 0]), lw * 3, (0, 0, 255), -1, cv2.LINE_AA)
        top = pts[:, 0, :][np.argmin(pts[:, 0, 1])]
        _label(out, f"#{i + 1} inliers {d.inliers} ({d.inlier_ratio:.0%})", (top[0], top[1] - 8), c)
    status = "FOUND" if res.found else "NOT FOUND"
    _label(out, status, (10, 34), (60, 220, 60) if res.found else (60, 60, 255), 0.9, 2)
    return out


def draw_keypoints(img_bgr: np.ndarray, kps, scale: float = 1.0) -> np.ndarray:
    """Ключові точки: коло = масштаб, лінія = орієнтація."""
    work = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else img_bgr
    out = cv2.drawKeypoints(work, kps, None, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    _label(out, f"keypoints: {len(kps)}", (8, 28), (0, 255, 0), 0.7, 2)
    return out


def draw_matches(template_bgr, scene_bgr, res: MatchResult, inliers_only=True, max_lines=150) -> np.ndarray:
    """Лінії між відповідними точками зразка і сцени."""
    t_work, _ = _resize_max_side(template_bgr, int(round(max(template_bgr.shape[:2]) * res.template_scale)))
    s_work = cv2.resize(scene_bgr, None, fx=res.scene_scale, fy=res.scene_scale,
                        interpolation=cv2.INTER_AREA) if res.scene_scale != 1.0 else scene_bgr
    def thin(ms):
        if len(ms) > max_lines:
            idx = np.linspace(0, len(ms) - 1, max_lines).astype(int)
            ms = [ms[i] for i in idx]
        return ms

    if inliers_only and res.detections:
        # кожен знайдений екземпляр — своїм кольором
        groups = [(thin(d.inlier_matches), COLORS[i % len(COLORS)]) for i, d in enumerate(res.detections)]
        color = COLORS[0]
        title = f"RANSAC inliers: {sum(d.inliers for d in res.detections)}"
    else:
        color = (0, 200, 255)
        groups = [(thin(list(res.good_matches)), color)]
        title = f"good matches (ratio test): {len(res.good_matches)}"
    out = cv2.drawMatches(t_work, res.kp_template, s_work, res.kp_scene, groups[0][0], None,
                          matchColor=groups[0][1], singlePointColor=None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    for ms, c in groups[1:]:
        out = cv2.drawMatches(t_work, res.kp_template, s_work, res.kp_scene, ms, out, matchColor=c,
                              flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if inliers_only and res.detections:
        off = np.float32([t_work.shape[1], 0])
        for i, d in enumerate(res.detections):
            pts = (d.corners * res.scene_scale + off).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, COLORS[i % len(COLORS)], 2, cv2.LINE_AA)
    _label(out, title, (8, 28), color, 0.7, 2)
    return out


def side_by_side(a: np.ndarray, b: np.ndarray, height: int = 480) -> np.ndarray:
    def fit(x):
        s = height / x.shape[0]
        return cv2.resize(x, (int(x.shape[1] * s), height))
    return np.hstack([fit(a), fit(b)])
