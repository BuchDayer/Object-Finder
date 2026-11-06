"""Модульні тести (стандартний unittest, без додаткових бібліотек).

    python -m unittest discover tests -v
"""

import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from finder import MatchParams, ObjectFinder, load_image          # noqa: E402
from finder.matcher import check_homography, inlier_coverage      # noqa: E402

DATA = os.path.join(ROOT, "data")


def run(case_id, **kw):
    with open(os.path.join(DATA, "ground_truth.json"), encoding="utf-8") as f:
        c = json.load(f)[case_id]
    f = ObjectFinder(MatchParams(**kw))
    f.set_template(load_image(os.path.join(DATA, c["template"])))
    return f.find(load_image(os.path.join(DATA, c["scene"]))), c


class TestFinder(unittest.TestCase):
    def test_frontal_found_accurately(self):
        res, c = run("01_frontal")
        self.assertTrue(res.found)
        err = np.linalg.norm(res.detections[0].corners - np.float32(c["objects"][0]["corners"]), axis=1).mean()
        self.assertLess(err, 3.0)

    def test_rotation_180(self):
        res, _ = run("05_rotation_180")
        self.assertTrue(res.found)

    def test_negative_not_found(self):
        res, _ = run("10_negative")
        self.assertFalse(res.found)

    def test_two_instances(self):
        res, _ = run("09_two_copies", max_instances=2)
        self.assertEqual(len(res.detections), 2)

    def test_low_texture_reports_reason(self):
        res, _ = run("14_low_texture")
        self.assertFalse(res.found)
        self.assertIn("ключових точок", res.reason)


class TestChecks(unittest.TestCase):
    def test_identity_homography_ok(self):
        ok, _, _ = check_homography(np.eye(3), 100, 150, (500, 500))
        self.assertTrue(ok)

    def test_mirrored_rejected(self):
        H = np.diag([-1.0, 1.0, 1.0])
        H[0, 2] = 200
        ok, why, _ = check_homography(H, 100, 150, (500, 500))
        self.assertFalse(ok)
        self.assertIn("дзеркальне", why)

    def test_coverage(self):
        pts = np.float32([[5, 5], [6, 6], [7, 5]])           # усі в одній клітинці
        self.assertAlmostEqual(inlier_coverage(pts, 100, 100), 1 / 16)
        grid = np.float32([[x, y] for x in range(5, 100, 25) for y in range(5, 100, 25)])
        self.assertAlmostEqual(inlier_coverage(grid, 100, 100), 1.0)


if __name__ == "__main__":
    unittest.main()
