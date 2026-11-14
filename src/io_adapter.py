"""数据读取：把 CSV 流读成 numpy 数组，并提供按段/按窗口切分。"""
from __future__ import annotations

import csv

import numpy as np


class Stream:
    """一条按时间排序的数据流。特征列约定为 f0..f{n-1}，标签列 label。"""

    def __init__(self, t, segment, X, y):
        self.t = t
        self.segment = segment
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def slice(self, start: int, end: int) -> "Stream":
        return Stream(self.t[start:end], self.segment[start:end],
                      self.X[start:end], self.y[start:end])

    def segments(self) -> list[str]:
        seen = []
        for s in self.segment:
            if s not in seen:
                seen.append(s)
        return seen

    def by_segment(self, name: str) -> "Stream":
        mask = np.array([s == name for s in self.segment])
        return Stream(self.t[mask], [s for s, m in zip(self.segment, mask) if m],
                      self.X[mask], self.y[mask])

    def windows(self, size: int) -> list[tuple[int, "Stream"]]:
        """按固定大小切窗，用于滚动评估。返回 [(起始下标, 窗口流)]。"""
        out = []
        for start in range(0, len(self), size):
            s = self.slice(start, min(start + size, len(self)))
            if len(s) > 0:
                out.append((start, s))
        return out


def load_stream(path: str) -> Stream:
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        feat_idx = [i for i, h in enumerate(header) if h.startswith("f")]
        t_idx = header.index("t")
        seg_idx = header.index("segment")
        y_idx = header.index("label")
        t, seg, rows, ys = [], [], [], []
        for r in rd:
            t.append(int(r[t_idx]))
            seg.append(r[seg_idx])
            rows.append([float(r[i]) for i in feat_idx])
            ys.append(int(r[y_idx]))
    return Stream(np.asarray(t), seg, np.asarray(rows, dtype=float), np.asarray(ys, dtype=int))
