"""Ядро системи: пошук об'єкта за фотографією-зразком.

Конвеєр обробки (pipeline):

    1. Попередня обробка   — переведення в відтінки сірого, (опційно) CLAHE, зменшення розміру
    2. Детекція            — пошук ключових точок (ORB / AKAZE / SIFT)
    3. Опис                — обчислення дескрипторів для кожної ключової точки
    4. Зіставлення         — k-NN пошук (k=2) + тест відношення Лоу (ratio test)
    5. Оцінка моделі       — гомографія 3x3 методом RANSAC (відкидає хибні пари — outliers)
    6. Перевірка           — достатньо inliers, вони покривають зразок, чотирикутник правдоподібний
    7. Результат           — кути об'єкта на сцені, кількість inliers, причина відмови тощо

Зразок (template) — фото обкладинки / плаката / упаковки.
Сцена (scene)     — зображення, на якому шукаємо цей об'єкт.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Параметри алгоритму
# ---------------------------------------------------------------------------

@dataclass
class MatchParams:
    """Усі налаштування алгоритму в одному місці (їх можна міняти на захисті)."""

    detector: str = "ORB"          # "ORB" | "AKAZE" | "SIFT"
    n_features: int = 3000         # макс. кількість ключових точок (ORB, SIFT)
    ratio: float = 0.75            # поріг тесту Лоу: d1 < ratio * d2
    cross_check: bool = False      # взаємна перевірка: A->B і B->A мають збігатися
    ransac_method: str = "RANSAC"  # "RANSAC" | "MAGSAC" (USAC, якщо є в OpenCV)
    ransac_thresh: float = 5.0     # макс. похибка перепроєкції (px), щоб пара була inlier
    ransac_iters: int = 3000       # макс. кількість ітерацій RANSAC
    min_matches: int = 10          # мінімум «добрих» пар, щоб узагалі запускати RANSAC
    min_inliers: int = 15          # мінімум inliers, щоб вважати об'єкт знайденим
    min_inlier_ratio: float = 0.15 # частка inliers серед добрих пар
    min_coverage: float = 0.3      # частка клітинок сітки 4x4 зразка, де є хоча б один inlier
    use_clahe: bool = False        # вирівнювання контрасту (допомагає при поганому освітленні)
    max_scene_side: int = 1280     # сцену зменшуємо до цього розміру (швидкість)
    max_template_side: int = 640   # зразок зменшуємо до цього розміру
    max_instances: int = 1         # скільки копій об'єкта шукати на сцені

    @classmethod
    def from_dict(cls, d: dict) -> "MatchParams":
        p = cls()
        for k, v in (d or {}).items():
            if hasattr(p, k):
                cur = getattr(p, k)
                if isinstance(cur, bool):
                    v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
                elif isinstance(cur, int):
                    v = int(float(v))
                elif isinstance(cur, float):
                    v = float(v)
                setattr(p, k, v)
        return p


# ---------------------------------------------------------------------------
# Структури результату
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Один знайдений екземпляр об'єкта."""
    corners: np.ndarray            # 4x2 — кути зразка, перенесені на сцену (порядок: ЛВ, ПВ, ПН, ЛН)
    H: np.ndarray                  # 3x3 гомографія: точка зразка -> точка сцени (у вихідних пікселях)
    inliers: int                   # кількість пар, що узгоджуються з H
    candidates: int                # скільки пар подавали в RANSAC
    coverage: float = 0.0          # наскільки рівномірно inliers покривають зразок (0..1)
    inlier_matches: List[cv2.DMatch] = field(default_factory=list, repr=False)

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.candidates if self.candidates else 0.0


@dataclass
class MatchResult:
    found: bool
    detections: List[Detection]
    reason: str                    # пояснення, чому знайдено / не знайдено
    n_kp_template: int = 0
    n_kp_scene: int = 0
    n_raw_matches: int = 0         # пар після k-NN (до ratio test)
    n_good_matches: int = 0        # пар після ratio test
    timings_ms: dict = field(default_factory=dict)
    match_direction: str = ""
    # для візуалізації (у робочому масштабі)
    kp_template: list = field(default_factory=list, repr=False)
    kp_scene: list = field(default_factory=list, repr=False)
    good_matches: List[cv2.DMatch] = field(default_factory=list, repr=False)
    template_scale: float = 1.0    # робочий розмір / вихідний
    scene_scale: float = 1.0

    def summary(self) -> dict:
        """Короткий словник для друку / JSON."""
        return {
            "found": self.found,
            "instances": len(self.detections),
            "reason": self.reason,
            "keypoints_template": self.n_kp_template,
            "keypoints_scene": self.n_kp_scene,
            "raw_matches": self.n_raw_matches,
            "good_matches": self.n_good_matches,
            "match_direction": self.match_direction,
            "detections": [
                {
                    "corners": np.round(d.corners, 1).tolist(),
                    "inliers": d.inliers,
                    "candidates": d.candidates,
                    "inlier_ratio": round(d.inlier_ratio, 3),
                    "coverage": round(d.coverage, 3),
                    "H": np.round(d.H, 5).tolist(),
                }
                for d in self.detections
            ],
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
        }


# ---------------------------------------------------------------------------
# Допоміжні функції
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Читає зображення (працює і з кириличними шляхами у Windows)."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не вдалося прочитати зображення: {path}")
    return img


def _resize_max_side(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Зменшує зображення так, щоб більша сторона була <= max_side. Повертає (img, scale)."""
    h, w = img.shape[:2]
    s = max(h, w)
    if max_side <= 0 or s <= max_side:
        return img, 1.0
    scale = max_side / float(s)
    out = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def preprocess(img: np.ndarray, use_clahe: bool) -> np.ndarray:
    """BGR -> сірий (+ CLAHE). Ключові точки шукаються лише за яскравістю."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if use_clahe:
        # CLAHE — локальне вирівнювання гістограми з обмеженням контрасту
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return gray


def create_detector(p: MatchParams):
    """Створює детектор+дескриптор і повертає (detector, norm_type)."""
    name = p.detector.upper()
    if name == "ORB":
        # ORB = Oriented FAST + Rotated BRIEF. Бінарний дескриптор 256 біт -> відстань Геммінга
        det = cv2.ORB_create(nfeatures=p.n_features, scaleFactor=1.2, nlevels=8,
                             edgeThreshold=31, patchSize=31, fastThreshold=20)
        return det, cv2.NORM_HAMMING
    if name == "AKAZE":
        # AKAZE — нелінійний масштабний простір, бінарний дескриптор M-LDB -> Геммінг
        # OpenCV 4.x: cv2.AKAZE_create; OpenCV 5.x: перенесено в cv2.xfeatures2d (пакет contrib)
        akaze_create = getattr(cv2, "AKAZE_create", None) or \
            getattr(getattr(cv2, "xfeatures2d", None), "AKAZE_create", None)
        if akaze_create is None:
            raise ValueError("AKAZE недоступний у цій версії OpenCV (встановіть opencv-contrib-python)")
        det = akaze_create(threshold=0.0008)
        return det, cv2.NORM_HAMMING
    if name == "SIFT":
        # SIFT — DoG-детектор, дескриптор 128 float -> евклідова відстань (L2)
        det = cv2.SIFT_create(nfeatures=p.n_features)
        return det, cv2.NORM_L2
    raise ValueError(f"Невідомий детектор: {p.detector}")


def _ransac_flag(method: str) -> int:
    if method.upper() == "MAGSAC" and hasattr(cv2, "USAC_MAGSAC"):
        return cv2.USAC_MAGSAC
    return cv2.RANSAC


def inlier_coverage(points: np.ndarray, w: int, h: int, grid: int = 4) -> float:
    """Частка клітинок сітки grid x grid на зразку, в які потрапив хоча б один inlier.

    Захищає від хибних спрацювань, коли збігаються лише маленькі однакові фрагменти
    (напр. однаковий шрифт заголовка на різних обкладинках однієї серії): inliers тоді
    скупчені в 1–4 клітинках, а при справжньому збігу розкидані по всьому зразку.
    """
    if len(points) == 0:
        return 0.0
    gx = np.clip((points[:, 0] / w * grid).astype(int), 0, grid - 1)
    gy = np.clip((points[:, 1] / h * grid).astype(int), 0, grid - 1)
    return len(set(zip(gx.tolist(), gy.tolist()))) / float(grid * grid)


def check_homography(H: np.ndarray, tw: int, th: int, scene_shape) -> Tuple[bool, str, np.ndarray]:
    """Перевіряє, чи гомографія дає правдоподібний чотирикутник.

    RANSAC завжди поверне *якусь* модель, навіть на сміттєвих парах. Тому перевіряємо:
      * орієнтація не дзеркальна (det > 0) — реальний плоский об'єкт не «перевертається»;
      * чотирикутник опуклий — перспективна проєкція прямокутника завжди опукла;
      * площа не надто мала й не надто велика;
      * чотирикутник не «сплющений» у лінію.
    """
    corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    if not np.all(np.isfinite(proj)):
        return False, "гомографія вироджена (нескінченні координати)", proj
    if np.linalg.det(H[:2, :2]) <= 0:
        return False, "дзеркальне відображення (det<=0)", proj
    if not cv2.isContourConvex(proj.astype(np.float32).reshape(-1, 1, 2)):
        return False, "чотирикутник неопуклий (самоперетин)", proj

    area = cv2.contourArea(proj.astype(np.float32))
    sh, sw = scene_shape[:2]
    if area < 100:
        return False, f"занадто мала площа ({area:.0f} px²)", proj
    if area > 4.0 * sw * sh:
        return False, "площа значно більша за сцену", proj

    sides = [np.linalg.norm(proj[i] - proj[(i + 1) % 4]) for i in range(4)]
    tmpl_aspect = min(tw, th) / max(tw, th)
    quad_aspect = min(sides) / max(sides)
    if quad_aspect < 0.15 * tmpl_aspect:
        return False, "чотирикутник сплющений (надто сильна перспектива)", proj
    return True, "ok", proj


# ---------------------------------------------------------------------------
# Основний клас
# ---------------------------------------------------------------------------

class ObjectFinder:
    """Шукає об'єкт-зразок на сценах.

    Використання:
        finder = ObjectFinder(MatchParams(detector="ORB"))
        finder.set_template(template_bgr)       # дескриптори зразка рахуються один раз
        result = finder.find(scene_bgr)
    """

    def __init__(self, params: Optional[MatchParams] = None):
        self.params = params or MatchParams()
        self.detector, self.norm = create_detector(self.params)
        self.template = None
        self.tmpl_work = None
        self.kp_t = []
        self.des_t = None
        self.t_scale = 1.0
        self.t_size = (0, 0)      # (w, h) вихідного зразка
        self.t_detect_ms = 0.0

    # -- зразок -------------------------------------------------------------
    def set_template(self, template_bgr: np.ndarray) -> None:
        p = self.params
        self.template = template_bgr
        h, w = template_bgr.shape[:2]
        self.t_size = (w, h)
        work, self.t_scale = _resize_max_side(template_bgr, p.max_template_side)
        self.tmpl_work = work
        gray = preprocess(work, p.use_clahe)
        t0 = time.perf_counter()
        self.kp_t, self.des_t = self.detector.detectAndCompute(gray, None)
        self.t_detect_ms = (time.perf_counter() - t0) * 1000

    # -- зіставлення дескрипторів --------------------------------------------
    def _match(self, des_s) -> Tuple[List[cv2.DMatch], int, str]:
        """Повертає «добрі» пари (queryIdx = точка зразка, trainIdx = точка сцени)."""
        p = self.params
        bf = cv2.BFMatcher(self.norm, crossCheck=False)

        # Напрям зіставлення:
        #  * 1 екземпляр: зразок -> сцена (класичний варіант);
        #  * кілька екземплярів: сцена -> зразок. Якщо на сцені 2 однакові обкладинки,
        #    то для точки зразка два найближчі сусіди на сцені майже однакові, і тест Лоу
        #    відкине обидві пари. У напрямку сцена->зразок такої проблеми немає.
        multi = p.max_instances > 1
        direction = "scene→template" if multi else "template→scene"

        if multi:
            knn = bf.knnMatch(des_s, self.des_t, k=2)
        else:
            knn = bf.knnMatch(self.des_t, des_s, k=2)
        raw = len(knn)

        good: List[cv2.DMatch] = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            # Тест відношення Лоу: найкращий збіг має бути помітно кращим за другий.
            if m.distance < p.ratio * n.distance:
                good.append(m)

        if p.cross_check and good:
            # Взаємна перевірка: для точки сцени найкраща точка зразка має бути тією самою.
            if multi:
                back = bf.match(self.des_t, des_s)
                best_for_t = {b.queryIdx: b.trainIdx for b in back}
                good = [m for m in good if best_for_t.get(m.trainIdx) == m.queryIdx]
            else:
                back = bf.match(des_s, self.des_t)
                best_for_s = {b.queryIdx: b.trainIdx for b in back}
                good = [m for m in good if best_for_s.get(m.trainIdx) == m.queryIdx]

        if multi:
            # приводимо до єдиного формату: queryIdx — зразок, trainIdx — сцена
            good = [cv2.DMatch(m.trainIdx, m.queryIdx, m.distance) for m in good]
        return good, raw, direction

    # -- пошук ---------------------------------------------------------------
    def find(self, scene_bgr: np.ndarray) -> MatchResult:
        if self.des_t is None and self.template is None:
            raise RuntimeError("Спочатку викличте set_template()")
        p = self.params
        timings = {"detect_template": self.t_detect_ms}

        scene_work, s_scale = _resize_max_side(scene_bgr, p.max_scene_side)
        gray = preprocess(scene_work, p.use_clahe)

        t0 = time.perf_counter()
        kp_s, des_s = self.detector.detectAndCompute(gray, None)
        timings["detect_scene"] = (time.perf_counter() - t0) * 1000

        res = MatchResult(found=False, detections=[], reason="",
                          n_kp_template=len(self.kp_t), n_kp_scene=len(kp_s),
                          timings_ms=timings, kp_template=self.kp_t, kp_scene=kp_s,
                          template_scale=self.t_scale, scene_scale=s_scale)

        timings["total"] = timings["detect_scene"]
        if self.des_t is None or len(self.kp_t) < 4:
            res.reason = (f"на зразку замало ключових точок ({len(self.kp_t)}) — "
                          "малотекстурне зображення")
            return res
        if des_s is None or len(kp_s) < 4:
            res.reason = "на сцені замало ключових точок"
            return res

        t0 = time.perf_counter()
        good, raw, direction = self._match(des_s)
        timings["match"] = (time.perf_counter() - t0) * 1000
        res.n_raw_matches, res.n_good_matches = raw, len(good)
        res.good_matches, res.match_direction = good, direction

        # Матриці масштабування між робочими й вихідними координатами
        S_t = np.diag([self.t_scale, self.t_scale, 1.0])          # вихідний зразок -> робочий
        S_s_inv = np.diag([1 / s_scale, 1 / s_scale, 1.0])         # робоча сцена -> вихідна
        tw_work, th_work = self.tmpl_work.shape[1], self.tmpl_work.shape[0]

        t0 = time.perf_counter()
        remaining = list(good)
        reasons = []
        while len(res.detections) < p.max_instances:
            if len(remaining) < p.min_matches:
                reasons.append(f"замало добрих пар ({len(remaining)} < {p.min_matches})")
                break
            src = np.float32([self.kp_t[m.queryIdx].pt for m in remaining]).reshape(-1, 1, 2)
            dst = np.float32([kp_s[m.trainIdx].pt for m in remaining]).reshape(-1, 1, 2)

            # RANSAC: багаторазово бере 4 випадкові пари, рахує H, рахує скільки пар
            # укладаються в поріг ransac_thresh; залишає модель з найбільшою підтримкою.
            H, mask = cv2.findHomography(src, dst, _ransac_flag(p.ransac_method),
                                         p.ransac_thresh, maxIters=p.ransac_iters,
                                         confidence=0.995)
            if H is None or mask is None:
                reasons.append("RANSAC не знайшов модель")
                break
            mask = mask.ravel().astype(bool)
            n_in = int(mask.sum())
            ratio_in = n_in / len(remaining)

            if n_in < p.min_inliers:
                reasons.append(f"замало inliers ({n_in} < {p.min_inliers})")
                break
            if ratio_in < p.min_inlier_ratio:
                reasons.append(f"мала частка inliers ({ratio_in:.2f} < {p.min_inlier_ratio})")
                break
            cov = inlier_coverage(src.reshape(-1, 2)[mask], tw_work, th_work)
            if cov < p.min_coverage:
                reasons.append(f"inliers скупчені на малій ділянці зразка (покриття {cov:.2f} < {p.min_coverage})")
                break
            ok, why, quad = check_homography(H, tw_work, th_work, scene_work.shape)
            if not ok:
                reasons.append("відхилено перевіркою геометрії: " + why)
                break

            # Перехід до вихідних координат: H_full = S_s^-1 · H · S_t
            H_full = S_s_inv @ H @ S_t
            H_full /= H_full[2, 2]
            w0, h0 = self.t_size
            corners = cv2.perspectiveTransform(
                np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]]).reshape(-1, 1, 2), H_full).reshape(-1, 2)
            inl = [m for m, k in zip(remaining, mask) if k]
            res.detections.append(Detection(corners=corners, H=H_full, inliers=n_in,
                                            candidates=len(remaining), coverage=cov,
                                            inlier_matches=inl))

            # для пошуку наступного екземпляра прибираємо пари, що потрапили в знайдену область
            poly = quad.astype(np.float32).reshape(-1, 1, 2)
            remaining = [m for m, k in zip(remaining, mask)
                         if not k and cv2.pointPolygonTest(poly, kp_s[m.trainIdx].pt, True) < -3]

        timings["homography"] = (time.perf_counter() - t0) * 1000
        timings.pop("total", None)
        timings["total"] = sum(v for k, v in timings.items() if k != "detect_template")

        res.found = len(res.detections) > 0
        if res.found:
            d = res.detections[0]
            res.reason = (f"знайдено {len(res.detections)} екз.; inliers={d.inliers}/{d.candidates} "
                          f"({d.inlier_ratio:.0%}), покриття {d.coverage:.2f}")
        else:
            res.reason = reasons[0] if reasons else "не знайдено"
        return res


def find_object(template_bgr: np.ndarray, scene_bgr: np.ndarray,
                params: Optional[MatchParams] = None) -> MatchResult:
    """Зручна функція «в один рядок»."""
    f = ObjectFinder(params)
    f.set_template(template_bgr)
    return f.find(scene_bgr)


def params_to_dict(p: MatchParams) -> dict:
    return asdict(p)
