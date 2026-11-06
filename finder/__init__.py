"""Пакет finder — пошук об'єкта на зображенні за фотографією-зразком.

Основні складові:
    matcher.py   — алгоритм: ключові точки -> дескриптори -> зіставлення -> RANSAC -> гомографія
    visualize.py — малювання результатів (ключові точки, зіставлення, рамка знайденого об'єкта)
"""

from .matcher import MatchParams, Detection, MatchResult, ObjectFinder, load_image

__all__ = ["MatchParams", "Detection", "MatchResult", "ObjectFinder", "load_image"]
