"""Генератор тестового набору (програмна симуляція замість фотокамери).

Створює:
  data/templates/*.png       — зразки: «обкладинки книг», «плакат», «упаковка», проблемні зразки
  data/scenes/*.jpg          — сцени, на які зразок вклеєно з ВІДОМОЮ гомографією
  data/ground_truth.json     — справжні координати кутів об'єкта на кожній сцені
  data/video/demo_input.mp4  — тестове відео з рухомою обкладинкою

Завдяки відомій «правильній відповіді» точність можна виміряти числом
(середня похибка кутів у пікселях), а не лише «на око».

Запуск (потрібен scikit-image лише для вихідних фото):
    pip install scikit-image
    python tools/make_dataset.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import shutil

import cv2
import numpy as np
from skimage import data as skd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RNG = np.random.default_rng(18)  # варіант 18 :)


def rgb2bgr(im):
    if im.dtype != np.uint8:
        im = (im.astype(np.float32) * (255.0 if im.max() <= 1 else 1.0)).clip(0, 255).astype(np.uint8)
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:
        im = im[:, :, :3]
    return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)


def text_center(img, text, y, scale, color, thick=2, font=cv2.FONT_HERSHEY_TRIPLEX):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(img, text, ((img.shape[1] - tw) // 2, y), font, scale, color, thick, cv2.LINE_AA)


def fit_crop(img, w, h):
    """Масштабує й обрізає фото під розмір w x h (як object-fit: cover)."""
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (int(math.ceil(iw * s)), int(math.ceil(ih * s))), interpolation=cv2.INTER_AREA)
    y0 = (r.shape[0] - h) // 2
    x0 = (r.shape[1] - w) // 2
    return r[y0:y0 + h, x0:x0 + w]


# ---------------------------------------------------------------------------
# Зразки
# ---------------------------------------------------------------------------

def make_book(photo, title, subtitle, author, band_color, text_color, size=(360, 520)):
    w, h = size
    cover = np.full((h, w, 3), band_color, np.uint8)
    cover[120:h - 90, 18:w - 18] = fit_crop(photo, w - 36, h - 210)
    cv2.rectangle(cover, (18, 120), (w - 19, h - 91), text_color, 2)
    text_center(cover, title, 62, 1.25, text_color, 2)
    text_center(cover, subtitle, 100, 0.6, text_color, 1, cv2.FONT_HERSHEY_SIMPLEX)
    text_center(cover, author, h - 45, 0.8, text_color, 1, cv2.FONT_HERSHEY_DUPLEX)
    cv2.rectangle(cover, (0, 0), (w - 1, h - 1), (30, 30, 30), 2)
    return cover


def make_templates():
    out = {}
    out["book_astronaut"] = make_book(rgb2bgr(skd.astronaut()), "SPACE WALK", "a novel about the orbit",
                                      "E. COLLINS", (40, 30, 120), (240, 240, 255))
    out["book_coffee"] = make_book(rgb2bgr(skd.coffee()), "MORNING", "recipes & stories",
                                   "A. BARISTA", (200, 225, 240), (40, 50, 90))
    out["book_cat"] = make_book(rgb2bgr(skd.chelsea()), "CHELSEA", "the curious cat",
                                "M. WHISKER", (60, 120, 60), (230, 255, 230))

    # плакат (альбомна орієнтація)
    poster = np.full((420, 640, 3), (20, 20, 20), np.uint8)
    poster[:, :400] = fit_crop(rgb2bgr(skd.rocket()), 400, 420)
    p2 = poster[:, 400:]
    cv2.putText(p2, "LAUNCH", (14, 80), cv2.FONT_HERSHEY_TRIPLEX, 1.4, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(p2, "DAY 2026", (14, 130), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    for i, line in enumerate(["Kennedy Space", "Center, pad 39A", "live stream", "start 09:30"]):
        cv2.putText(p2, line, (14, 200 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(poster, (0, 0), (639, 419), (0, 200, 255), 3)
    out["poster_rocket"] = poster

    # упаковка (квадратна коробка)
    pack = np.full((420, 420, 3), (230, 230, 230), np.uint8)
    pack[70:330, 40:380] = fit_crop(rgb2bgr(skd.hubble_deep_field()), 340, 260)
    text_center(pack, "GALAXY TEA", 52, 1.1, (90, 20, 60), 2)
    text_center(pack, "earl grey  |  100 g  |  40 bags", 370, 0.6, (60, 60, 60), 1, cv2.FONT_HERSHEY_SIMPLEX)
    text_center(pack, "LOT 18-2026", 402, 0.5, (60, 60, 60), 1, cv2.FONT_HERSHEY_SIMPLEX)
    cv2.rectangle(pack, (0, 0), (419, 419), (90, 20, 60), 4)
    out["pack_tea"] = pack

    # ПРОБЛЕМНИЙ зразок 1: малотекстурний (однотонний логотип)
    plain = np.full((400, 400, 3), (200, 120, 40), np.uint8)
    cv2.circle(plain, (200, 200), 120, (240, 240, 240), -1, cv2.LINE_AA)
    out["plain_logo"] = plain

    # ПРОБЛЕМНИЙ зразок 2: періодичний візерунок (як пакувальний папір / шпалери):
    # одна й та сама плитка повторюється 7x7 разів -> багато однакових дескрипторів
    tile = np.full((60, 60, 3), (235, 225, 250), np.uint8)
    cv2.circle(tile, (18, 18), 9, (40, 40, 200), -1, cv2.LINE_AA)
    cv2.rectangle(tile, (34, 34), (52, 50), (30, 120, 30), -1)
    cv2.putText(tile, "Q", (32, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.line(tile, (6, 50), (24, 36), (0, 0, 0), 2, cv2.LINE_AA)
    out["pattern_wrap"] = np.tile(tile, (7, 7, 1))

    os.makedirs(os.path.join(DATA, "templates"), exist_ok=True)
    for k, v in out.items():
        cv2.imwrite(os.path.join(DATA, "templates", f"{k}.png"), v)
    return out


# ---------------------------------------------------------------------------
# Фони та сцени
# ---------------------------------------------------------------------------

# Сторонні предмети. Фото, що входять у зразки (кіт, кава, астронавт...), сюди НЕ додаємо,
# інакше перевірка хибних спрацювань була б нечесною.
DISTRACTORS = [skd.coins, skd.clock, skd.horse, skd.moon, skd.camera, skd.page, skd.text,
               skd.immunohistochemistry]


def make_background(w=1280, h=960, seed=0):
    """Захаращений фон: текстура «столу» + випадкові сторонні зображення та фігури."""
    rng = np.random.default_rng(seed)
    tex = [skd.gravel, skd.grass][seed % 2]()  # цеглу не беремо: вона була б «справжнім» збігом
    tex = cv2.resize(tex, (w, h))
    tint = rng.uniform(0.5, 1.0, 3)
    bg = (cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR).astype(np.float32) * tint * 0.7 + 40).clip(0, 255).astype(np.uint8)
    for _ in range(5):
        src = rgb2bgr(DISTRACTORS[rng.integers(len(DISTRACTORS))]())
        s = rng.uniform(0.35, 0.7)
        src = cv2.resize(src, None, fx=s, fy=s)
        ang = rng.uniform(-40, 40)
        M = cv2.getRotationMatrix2D((src.shape[1] / 2, src.shape[0] / 2), ang, 1.0)
        M[:, 2] += [rng.uniform(0, w - src.shape[1]), rng.uniform(0, h - src.shape[0])]
        warped = cv2.warpAffine(src, M, (w, h))
        mask = cv2.warpAffine(np.full(src.shape[:2], 255, np.uint8), M, (w, h))
        bg[mask > 0] = warped[mask > 0]
    for _ in range(12):
        c = tuple(int(x) for x in rng.integers(0, 255, 3))
        p1 = rng.integers(0, [w, h])
        p2 = p1 + rng.integers(-150, 150, 2)
        cv2.line(bg, tuple(map(int, p1)), tuple(map(int, p2)), c, int(rng.integers(2, 6)), cv2.LINE_AA)
    return bg


def make_homography(tw, th, center, scale=1.0, rot=0.0, tilt_x=0.0, tilt_y=0.0, f=1400.0):
    """Гомографія з «фізичних» параметрів: масштаб, поворот у площині, нахили (3D) — як рух камери."""
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    pts = np.c_[src - [tw / 2, th / 2], np.zeros(4)] * scale
    rz, rx, ry = np.radians([rot, tilt_x, tilt_y])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    P = pts @ (Ry @ Rx @ Rz).T
    P[:, 2] += f
    dst = np.float32(np.c_[f * P[:, 0] / P[:, 2], f * P[:, 1] / P[:, 2]] + center)
    return cv2.getPerspectiveTransform(src, dst), dst


def paste(scene, tmpl, H):
    h, w = scene.shape[:2]
    warped = cv2.warpPerspective(tmpl, H, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full(tmpl.shape[:2], 255, np.uint8), H, (w, h), flags=cv2.INTER_LINEAR)
    a2 = mask.astype(np.float32) / 255
    a = a2[..., None]
    # легка тінь під об'єктом — для реалістичності
    shadow = cv2.GaussianBlur(np.roll(np.roll(a2, 8, 0), 8, 1), (31, 31), 0)[..., None]
    scene = (scene * (1 - 0.35 * shadow)).astype(np.float32)
    return (warped * a + scene * (1 - a)).clip(0, 255).astype(np.uint8)


def photometric(img, brightness=1.0, contrast_shift=0, gamma=1.0, noise=0.0, blur=0, motion=0, jpeg=92):
    out = img.astype(np.float32) * brightness + contrast_shift
    out = np.clip(out, 0, 255)
    if gamma != 1.0:
        out = 255 * (out / 255) ** gamma
    if noise > 0:
        out = out + RNG.normal(0, noise, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    if motion > 0:
        k = np.zeros((motion, motion), np.float32)
        k[motion // 2, :] = 1.0 / motion
        out = cv2.filter2D(out, -1, k)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# Опис тестових випадків: (id, зразок, опис, очікування, категорія, геометрія(и), фотометрія, оклюзія)
CASES = [
    ("01_frontal", "book_astronaut", "Фронтально, масштаб 0.8", "found", "базовий",
     [dict(scale=0.8, rot=5)], {}, None),
    ("02_scale_small", "book_coffee", "Зменшення (масштаб 0.4)", "found", "масштаб",
     [dict(scale=0.4, rot=-10)], {}, None),
    ("03_scale_large", "book_cat", "Збільшення (масштаб 1.5)", "found", "масштаб",
     [dict(scale=1.5, rot=8)], {}, None),
    ("04_rotation_60", "poster_rocket", "Поворот у площині на 60°", "found", "поворот",
     [dict(scale=0.9, rot=60)], {}, None),
    ("05_rotation_180", "pack_tea", "Поворот на 180° (догори дриґом)", "found", "поворот",
     [dict(scale=0.9, rot=180)], {}, None),
    ("06_perspective_35", "book_astronaut", "Перспектива: нахил 35°", "found", "ракурс",
     [dict(scale=0.9, rot=-15, tilt_x=35, tilt_y=10)], {}, None),
    ("07_dark_noise", "book_coffee", "Темно + шум (освітлення)", "found", "освітлення",
     [dict(scale=0.8, rot=20, tilt_y=20)], dict(brightness=0.35, noise=8, gamma=1.2), None),
    ("08_occlusion_40", "poster_rocket", "Перекрито ~40% об'єкта", "found", "оклюзія",
     [dict(scale=0.9, rot=-12)], {}, 0.4),
    ("09_two_copies", "pack_tea", "Дві копії об'єкта на сцені", "found x2", "кілька об'єктів",
     [dict(scale=0.55, rot=15, center=(380, 470)), dict(scale=0.6, rot=-25, tilt_y=25, center=(930, 470))], {}, None),
    ("10_negative", "book_cat", "Об'єкта НЕМАЄ (на сцені інша книга)", "not_found", "негативний",
     [], {}, None),
    ("11_perspective_65", "book_astronaut", "ПРОБЛЕМА: дуже сильний нахил 65°", "found", "проблемний",
     [dict(scale=0.9, rot=10, tilt_x=65)], {}, None),
    ("12_motion_blur", "book_cat", "ПРОБЛЕМА: сильне розмиття в русі", "found", "проблемний",
     [dict(scale=0.7, rot=-5)], dict(motion=25), None),
    ("13_tiny", "poster_rocket", "ПРОБЛЕМА: дуже малий об'єкт (масштаб 0.13)", "found", "проблемний",
     [dict(scale=0.13, rot=0)], {}, None),
    ("14_low_texture", "plain_logo", "ПРОБЛЕМА: малотекстурний зразок", "found", "проблемний",
     [dict(scale=0.8, rot=30)], {}, None),
    ("15_repetitive", "pattern_wrap", "ПРОБЛЕМА: періодичний візерунок (повторювана плитка)", "found", "проблемний",
     [dict(scale=0.8, rot=20, tilt_y=15)], {}, None),
]


def make_scenes(templates):
    os.makedirs(os.path.join(DATA, "scenes"), exist_ok=True)
    gt = {}
    for i, (cid, tname, desc, expect, cat, geoms, photo, occl) in enumerate(CASES):
        scene = make_background(seed=i + 3)
        if cid == "10_negative":
            # інша книга на сцені — схожий за типом, але інший об'єкт
            H, _ = make_homography(360, 520, (640, 480), scale=0.8, rot=12)
            scene = paste(scene, templates["book_coffee"], H)
        objects = []
        tmpl = templates[tname]
        th, tw = tmpl.shape[:2]
        for g in geoms:
            g = dict(g)
            center = g.pop("center", (640 + RNG.uniform(-60, 60), 480 + RNG.uniform(-40, 40)))
            H, dst = make_homography(tw, th, np.float32(center), **g)
            scene = paste(scene, tmpl, H)
            objects.append({"corners": np.round(dst, 2).tolist(), "H": H.tolist()})
        if occl:
            # перекриваємо праву частину об'єкта «аркушем паперу»
            c = np.float32(objects[0]["corners"])
            x0 = int(c[:, 0].min() + (1 - occl) * (c[:, 0].max() - c[:, 0].min()))
            cv2.rectangle(scene, (x0, int(c[:, 1].min()) - 30), (x0 + 600, int(c[:, 1].max()) + 30),
                          (225, 228, 232), -1)
            cv2.putText(scene, "sheet of paper", (x0 + 20, int(c[:, 1].mean())), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (120, 120, 120), 2, cv2.LINE_AA)
        scene = photometric(scene, **photo)
        fname = f"{cid}.jpg"
        cv2.imwrite(os.path.join(DATA, "scenes", fname), scene, [cv2.IMWRITE_JPEG_QUALITY, 92])
        present = [f"templates/{tname}.png"] if objects else []
        if cid == "10_negative":
            present = ["templates/book_coffee.png"]
        gt[cid] = {"scene": f"scenes/{fname}", "template": f"templates/{tname}.png", "description": desc,
                   "expected": expect, "category": cat, "objects": objects,
                   "present_templates": present}
        print("сцена", cid, "-", desc)
    with open(os.path.join(DATA, "ground_truth.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)


def make_video(templates, n=300, w=960, h=540, fps=25):
    """Відео 12 с: обкладинка рухається, обертається, наближається, нахиляється і виходить з кадру."""
    os.makedirs(os.path.join(DATA, "video"), exist_ok=True)
    bg = make_background(w, h, seed=7)
    tmpl = templates["book_astronaut"]
    th, tw = tmpl.shape[:2]
    tmp = os.path.join(DATA, "video", "_raw.avi")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for i in range(n):
        t = i / (n - 1)
        cx = w / 2 + 260 * math.sin(2 * math.pi * t)
        cy = h / 2 + 60 * math.sin(4 * math.pi * t)
        if t > 0.82:                                   # наприкінці об'єкт виїжджає за кадр
            cx += (t - 0.82) / 0.18 * 900
        scale = 0.45 + 0.35 * (0.5 + 0.5 * math.sin(2 * math.pi * t * 1.5))
        H, _ = make_homography(tw, th, np.float32([cx, cy]), scale=scale * 0.85,
                               rot=40 * math.sin(2 * math.pi * t), tilt_x=25 * math.sin(3 * math.pi * t),
                               tilt_y=20 * math.cos(2 * math.pi * t))
        fr = paste(bg.copy(), tmpl, H)
        # «камера»: легкий шум і зміна яскравості
        fr = photometric(fr, brightness=0.85 + 0.15 * math.sin(6 * t), noise=3, jpeg=90)
        vw.write(fr)
    vw.release()
    final = os.path.join(DATA, "video", "demo_input.mp4")
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "23", "-pix_fmt", "yuv420p", final], check=True)
        os.remove(tmp)
    else:
        os.replace(tmp, final.replace(".mp4", ".avi"))
    print("відео", final)


if __name__ == "__main__":
    T = make_templates()
    make_scenes(T)
    make_video(T)
