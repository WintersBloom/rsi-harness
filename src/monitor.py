"""漂移监测：两条证据线，数据侧（特征分布）+ 性能侧（预测分布/代理指标）。

为什么必须两条：
  - 只看数据侧（PSI）→ 协变量漂移会误报（分布变了但规律没变，模型其实没坏）；
  - 只看性能侧 → 概念漂移早期没有标签，看不出来。
  两条同时异常才升级到"重训"，这是 plan §1 判断 J2 的"双证据门"。

阈值（PSI 的通用经验值，写进 config/monitor.yaml 可改）：
  < 0.1  基本没变
  0.1~0.25  中度
  > 0.25  显著漂移
"""
from __future__ import annotations

import math

import numpy as np


# ---------- 基础统计量 ----------
def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index。分箱边界取自参考数据的分位数，保证每箱样本量均衡。"""
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    qs = np.linspace(0, 100, bins + 1)
    edges = np.percentile(ref, qs)
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    p = np.histogram(ref, bins=edges)[0] / len(ref)
    q = np.histogram(cur, bins=edges)[0] / len(cur)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """两样本 KS：返回 (D 统计量, 近似 p 值)。p 值用 Kolmogorov 分布的渐近公式。"""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0, 1.0
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / n
    cdf_b = np.searchsorted(b, grid, side="right") / m
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))


def psi_level(v: float) -> int:
    if v >= 0.25:
        return 2
    if v >= 0.10:
        return 1
    return 0


# ---------- 监测器 ----------
class Monitor:
    """维护一个参考窗口（通常是上线时的训练数据），每来一个新窗口就对比一次。"""

    def __init__(self, ref_X: np.ndarray, ref_pred: np.ndarray | None = None,
                 bins: int = 10, feature_warn: float = 0.10, feature_alert: float = 0.25,
                 pred_warn: float = 0.10, pred_alert: float = 0.25):
        self.ref_X = np.asarray(ref_X, dtype=float)
        self.ref_pred = None if ref_pred is None else np.asarray(ref_pred, dtype=float)
        self.bins = bins
        self.feature_warn = feature_warn
        self.feature_alert = feature_alert
        self.pred_warn = pred_warn
        self.pred_alert = pred_alert

    def check(self, cur_X: np.ndarray, cur_pred: np.ndarray,
              cur_pred_ref: np.ndarray | None = None) -> dict:
        """对比一个新窗口。

        `cur_pred_ref` 是**参考模型**（冻结基线）在这个窗口上的打分。为什么需要它：
        如果我们用"当前模型"的打分去算 pred_psi，那么模型被漂移带歪之后，它自己的
        坏预测就会把 pred_psi 顶高，于是监测器一直报警、永远判不出"漂移已经结束"。
        实测过：D 段（数据已恢复正常）用冻结模型看 PSI 只有 0.047，跟平稳期 0.045
        几乎一样；可用自适应模型看却一直在报警——监测器被自己污染了。

        所以 pred_psi 一律用**固定参考模型**算：它衡量的是"数据变了没有"，
        而不是"我们现在这个模型好不好"。后者是评估门的事，不该混进来。
        """
        cur_X = np.asarray(cur_X, dtype=float)
        cur_pred = np.asarray(cur_pred, dtype=float)

        # 数据侧：逐特征 PSI
        fps = []
        for j in range(self.ref_X.shape[1]):
            fps.append(psi(self.ref_X[:, j], cur_X[:, j], self.bins))
        fps_arr = np.asarray(fps)

        # 性能侧：预测分数分布漂移（无标签时唯一可用的"模型视角"信号）
        pred_psi = 0.0
        if self.ref_pred is not None:
            src = cur_pred if cur_pred_ref is None else np.asarray(cur_pred_ref, dtype=float)
            pred_psi = psi(self.ref_pred, src, max(5, self.bins // 2))

        feat_level = 2 if fps_arr.max() >= self.feature_alert else (
            1 if fps_arr.max() >= self.feature_warn else 0)
        pred_level = 2 if pred_psi >= self.pred_alert else (
            1 if pred_psi >= self.pred_warn else 0)

        # 双证据门：两侧都不低于"警告"才升到等级 2（=可重训）；
        # 只有一侧强烈异常时最多给等级 1（只报警/做廉价校准），避免"分布变了但规律没变"的误报。
        if feat_level >= 1 and pred_level >= 1:
            level = 2
        elif max(feat_level, pred_level) >= 2:
            level = 1
        else:
            level = max(feat_level, pred_level)

        return {
            "level": int(level),
            "feature_psi": [round(float(v), 5) for v in fps],
            "feature_psi_max": round(float(fps_arr.max()), 5),
            "feature_psi_mean": round(float(fps_arr.mean()), 5),
            "pred_psi": round(float(pred_psi), 5),
            "feat_level": feat_level,
            "pred_level": pred_level,
            "n": int(len(cur_X)),
        }
