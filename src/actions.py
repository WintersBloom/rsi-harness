"""动作实现：校准（便宜）与重训（贵），以及每次动作的成本计时。

校准为什么排在重训前面：
  评测是"单位增益成本"的比值。一次全量重训要几十秒 CPU，一次阈值校准只要毫秒级，
  而"重新校准决策阈值"往往能拿回大部分因漂移丢掉的 F1。先用满便宜动作，是这题的核心算计。
"""
from __future__ import annotations

import time

import numpy as np

from src.predictor import NumpyLR


class ActionRunner:
    def __init__(self, model: NumpyLR, threshold: float = 0.5, ref_pos_rate: float = 0.08):
        self.model = model
        self.threshold = threshold
        self.ref_pos_rate = ref_pos_rate

    # ---------- 廉价动作：只动阈值，不动权重 ----------
    def calibrate(self, X_cur: np.ndarray) -> dict:
        """无监督阈值校准：让当前窗口的预测正例率回到参考正例率附近。

        不需要标签：只调整"多少分以上算故障"这条线。对协变量漂移造成的阈值失配特别有效。
        """
        t0 = time.perf_counter()
        p = self.model.predict_proba(X_cur)
        old = self.threshold
        if len(p) == 0:
            return {"action": "calibrate", "changed": False, "threshold": old, "cpu_s": 0.0}
        # 取分位数：使 p >= new_thr 的比例 ≈ ref_pos_rate
        k = int(round((1.0 - self.ref_pos_rate) * (len(p) - 1)))
        new_thr = float(np.sort(p)[k])
        # 幅度限制：单次校准不超过 ±0.25，避免过冲（对应 plan 的"受控"要求）
        new_thr = float(np.clip(new_thr, old - 0.25, old + 0.25))
        new_thr = float(np.clip(new_thr, 0.02, 0.98))
        self.threshold = new_thr
        return {
            "action": "calibrate",
            "changed": abs(new_thr - old) > 1e-4,
            "threshold_before": round(old, 4),
            "threshold_after": round(new_thr, 4),
            "cpu_s": round(time.perf_counter() - t0, 4),
        }

    # ---------- 昂贵动作：重训 ----------
    def retrain(self, X: np.ndarray, y: np.ndarray, lr: float = 0.05,
                epochs: int = 300, l2: float = 1e-3, seed: int = 0,
                sample_weight: np.ndarray | None = None,
                anchor: tuple | None = None, anchor_strength: float = 0.0,
                scaler: tuple | None = None) -> dict:
        t0 = time.perf_counter()
        m = NumpyLR(lr=lr, epochs=epochs, l2=l2, class_weight=True, seed=seed)
        if anchor is not None and scaler is not None:
            # 沿用旧模型的标准化统计量，保证新旧权重在同一量纲，弹簧才拉得准
            m.scaler.mean = np.asarray(scaler[0], dtype=float).copy()
            m.scaler.std = np.asarray(scaler[1], dtype=float).copy()
        m.fit(X, y, sample_weight, anchor=anchor, anchor_strength=anchor_strength)
        return {"action": "retrain", "model": m, "cpu_s": round(time.perf_counter() - t0, 4),
                "config": {"lr": lr, "epochs": epochs, "l2": l2, "seed": seed,
                           "weighted": sample_weight is not None,
                           "anchor_strength": anchor_strength}}
