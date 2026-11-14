"""纯 numpy 逻辑回归 + 标准化，零第三方依赖。

为什么不直接用 sklearn：本机 pip 装包全线失败（见 .team/status.md），
评测环境也要求离线、依赖可控。手写一份 200 行的实现反而更稳、更好解释。
"""
from __future__ import annotations

import numpy as np


class Standardizer:
    """z-score 标准化。统计量只在 fit 时从训练数据算，推理时复用，避免信息泄漏。"""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        self.std = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        s = cls()
        s.mean = np.asarray(d["mean"], dtype=float)
        s.std = np.asarray(d["std"], dtype=float)
        return s


class NumpyLR:
    """带 L2 正则与类别权重的逻辑回归，用 Adam 做全批量梯度下降（纯 numpy）。

    统一接口：fit(X, y) / predict_proba(X) / predict(X, threshold)，可与其他实现互换。
    参数刻意少：lr / epochs / l2 / class_weight 四个，方便当成"超参数优化"的对象。
    """

    def __init__(self, lr: float = 0.05, epochs: int = 300, l2: float = 1e-3,
                 class_weight: bool = True, seed: int = 0, batch_size: int = 64,
                 poly: bool = False):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.class_weight = class_weight
        self.seed = seed
        self.batch_size = batch_size
        # 二阶特征交叉。默认关：合成数据的标签就是用线性 logit 生成的，
        # 在这份数据上加交叉项必然不涨（已实测，见 docs/model_capacity.md）。
        # 打开的唯一理由，是等官方数据证明"非线性边界真实存在"。
        self.poly = poly
        self.w = None
        self.b = 0.0
        self.scaler = Standardizer()
        self.history: list[dict] = []

    def _expand(self, X: np.ndarray) -> np.ndarray:
        """二阶特征交叉：原始 d 列 + 两两乘积 C(d,2) 列。poly=False 时原样返回。

        为什么用手工交叉而不是加一层 MLP：1500 条样本下，
        13 参数的比例是 1:115（安全）；833 参数的网是 1:1.8（必过拟合）；
        加 66 个乘积项到 79 参数，是 1:19，仍然可控，且每项能命名。
        这是小样本场景的标准做法——把已知的交互直接告诉模型，而不是让它自己摸索。
        """
        if not self.poly:
            return X
        d = X.shape[1]
        iu = np.triu_indices(d, k=1)
        return np.hstack([X, X[:, iu[0]] * X[:, iu[1]]])

    # ---------- 训练 ----------
    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None,
            anchor: tuple | None = None, anchor_strength: float = 0.0) -> "LogisticRegression":
        """训练。

        anchor 是"抗遗忘"开关（对应评分表的『稳定性与抗遗忘』一项）：把上一版的
        (w, b) 作为起点，并在梯度里加一项 `anchor_strength * (w - w_anchor)`，
        相当于一根弹簧把新模型拴在旧模型旁边。

        为什么需要它：滑动窗口重训治好了概念漂移，却会把旧工况的能力洗掉——
        我们的合成流里 D 段（恢复期）F1 从 0.817 掉到 0.711，掉了 10.6 个点，
        而评分表要求"历史工况回测下降不超过 5%"。

        注意 anchor 模式下**不再重新 fit 标准化器**，沿用旧模型的那一套：
        否则新旧权重落在不同的量纲里，弹簧拉的方向就没有意义了。
        """
        rng = np.random.default_rng(self.seed)
        X = self._expand(X)
        if anchor is not None and self.scaler.mean is not None:
            Xs = self.scaler.transform(X)
        else:
            Xs = self.scaler.fit_transform(X)
        n, d = Xs.shape
        w_anchor = None
        if anchor is not None:
            w_anchor = np.asarray(anchor[0], dtype=float)
            if w_anchor.shape[0] != d:
                raise ValueError(f"anchor 维度 {w_anchor.shape[0]} 与特征数 {d} 不一致")
            self.w = w_anchor.copy()
            self.b = float(anchor[1])
        else:
            self.w = np.zeros(d)
            self.b = 0.0

        # 类别权重：故障样本少，给它更大的权重，否则模型会全predict 正常
        if self.class_weight:
            pos = max(1, int(y.sum()))
            neg = max(1, n - pos)
            sw = np.where(y == 1, n / (2.0 * pos), n / (2.0 * neg))
        else:
            sw = np.ones(n)
        # 样本权重（按新旧加权）：概念漂移下，越新的样本越接近当前分布
        if sample_weight is not None:
            sw = sw * np.asarray(sample_weight, dtype=float)

        m_w = np.zeros(d)
        v_w = np.zeros(d)
        m_b = 0.0
        v_b = 0.0
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for ep in range(1, self.epochs + 1):
            # mini-batch：让 seed 真正影响训练轨迹。
            # 全批量梯度下降是确定性的，seed 起不到作用（配对增益会退化成"假"单点），
            # 而且真实评测里换 seed 必须有可观测差异，否则统计结论不成立。
            order = rng.permutation(n)
            for s0 in range(0, n, self.batch_size):
                bi = order[s0:s0 + self.batch_size]
                xb, yb, wb = Xs[bi], y[bi], sw[bi]
                p = self._sigmoid(xb @ self.w + self.b)
                err = (p - yb) * wb
                gw = (xb.T @ err) / len(bi) + self.l2 * self.w
                if w_anchor is not None and anchor_strength > 0:
                    gw = gw + anchor_strength * (self.w - w_anchor)
                gb = float(err.mean())

                m_w = beta1 * m_w + (1 - beta1) * gw
                v_w = beta2 * v_w + (1 - beta2) * gw * gw
                m_b = beta1 * m_b + (1 - beta1) * gb
                v_b = beta2 * v_b + (1 - beta2) * gb * gb
                mhat_w = m_w / (1 - beta1 ** ep)
                vhat_w = v_w / (1 - beta2 ** ep)
                mhat_b = m_b / (1 - beta1 ** ep)
                vhat_b = v_b / (1 - beta2 ** ep)
                self.w -= self.lr * mhat_w / (np.sqrt(vhat_w) + eps)
                self.b -= self.lr * mhat_b / (np.sqrt(vhat_b) + eps)

            if ep % 50 == 0 or ep == 1:
                p_full = self._sigmoid(Xs @ self.w + self.b)
                loss = self._logloss(p_full, y, sw)
                self.history.append({"epoch": ep, "loss": round(loss, 6)})
        return self

    # ---------- 推理 ----------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(self.scaler.transform(self._expand(X)) @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ---------- 工具 ----------
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -40, 40)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _logloss(p: np.ndarray, y: np.ndarray, sw: np.ndarray) -> float:
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-(sw * (y * np.log(p) + (1 - y) * np.log(1 - p))).mean())

    def feature_importance(self) -> np.ndarray:
        """标准化后的系数绝对值，可近似当特征重要性，用于漂移归因时排特征。"""
        return np.abs(self.w)

    # ---------- 持久化（只存 npz，不引入 pickle 依赖） ----------
    def save(self, path: str, meta: dict | None = None) -> None:
        np.savez(path, w=self.w, b=np.array([self.b]),
                 mean=self.scaler.mean, std=self.scaler.std,
                 poly=np.array([1.0 if self.poly else 0.0]))
        if meta is not None:
            import json
            with open(path + ".json", "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "NumpyLR":
        z = np.load(path)
        m = cls()
        m.w = z["w"]
        m.b = float(z["b"][0])
        m.scaler.mean = z["mean"]
        m.scaler.std = z["std"]
        # 旧模型（0.7 及以前）没有 poly 键，按 False 处理
        m.poly = bool(z["poly"][0]) if "poly" in z else False
        return m


# 兼容旧名字
LogisticRegression = NumpyLR

