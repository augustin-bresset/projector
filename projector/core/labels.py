"""`LabelSet` — the set of labeling classes (id, name, color).

Generic: no class is imposed. A "traversability" default is provided, but any class set
can be loaded/saved as JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class LabelClass:
    id: int
    name: str
    color: RGB


class LabelSet:
    def __init__(self, classes: list[LabelClass], ignore_id: int = 0) -> None:
        self.classes = list(classes)
        self.ignore_id = ignore_id
        self._by_id = {c.id: c for c in self.classes}

    @property
    def max_id(self) -> int:
        return max((c.id for c in self.classes), default=0)

    @property
    def paintable(self) -> list[LabelClass]:
        """Assignable classes (all but `ignore`)."""
        return [c for c in self.classes if c.id != self.ignore_id]

    def color_of(self, class_id: int) -> RGB:
        c = self._by_id.get(class_id)
        return c.color if c else (0, 0, 0)

    def name_of(self, class_id: int) -> str:
        c = self._by_id.get(class_id)
        return c.name if c else str(class_id)

    def lut(self, alpha: int = 255, max_id: int | None = None) -> np.ndarray:
        """RGBA LUT `(K, 4)` uint8 indexed by id. `ignore_id` -> alpha 0."""
        top = self.max_id if max_id is None else max(max_id, self.max_id)
        lut = np.zeros((top + 1, 4), dtype=np.uint8)
        for c in self.classes:
            if c.id == self.ignore_id or c.id < 0 or c.id > top:
                continue
            lut[c.id, :3] = c.color
            lut[c.id, 3] = alpha
        return lut

    def colorize(self, raster: np.ndarray, alpha: int = 255) -> np.ndarray:
        """Id raster `(rows, cols)` -> RGBA image `(rows, cols, 4)` uint8."""
        max_id = int(raster.max()) if raster.size else 0
        return self.lut(alpha=alpha, max_id=max_id)[raster]

    # --- (de)serialization ------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "ignore_id": self.ignore_id,
            "classes": [{"id": c.id, "name": c.name, "color": list(c.color)} for c in self.classes],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def from_dict(cls, d: dict) -> LabelSet:
        classes = [LabelClass(c["id"], c["name"], tuple(c["color"])) for c in d["classes"]]
        return cls(classes, ignore_id=d.get("ignore_id", 0))

    @classmethod
    def load(cls, path: str | Path) -> LabelSet:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def default(cls) -> LabelSet:
        """Minimal default set (fully editable). 0 = unlabeled, then a couple of classes."""
        return cls(
            [
                LabelClass(0, "unlabeled", (0, 0, 0)),
                LabelClass(1, "traversable", (60, 200, 70)),
                LabelClass(2, "obstacle", (220, 50, 45)),
            ],
            ignore_id=0,
        )
