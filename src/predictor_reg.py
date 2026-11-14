"""岭回归（闭式解）—— 回归任务的模型，对应分类里的 NumpyLR。

为什么单独一个文件而不是给 NumpyLR 加个开关：逻辑回归的 fit 是梯度下降 +
sigmoid + 类别权重，改造成回归会把那些分支全搅在一起（样本权重、batch、
Adam 状态都得各来一套），出错面太大。岭回归有闭式解，三十行就能写对，
而且解是唯一确定的——没有随机性，反而方便做对照实验。

σ 的取法：特征先标准化，惩罚项不含截距。l2 直接叫正则强度，和分类那边
的 l2 语义一致（都是"每样本损失 + λ‖w‖²"这一族的参数）。
"""

import numpy as np


class NumpyRidge:
    def __init__(self, l2: float = 1.0):
        self.l2 = float(l2)
        self.mean_ = None
        self.scale_ = None
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape

        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-8] = 1.0
        self.scale_ = sd
        Xs = (X - self.mean_) / self.scale_

        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float).ravel()
            sw = sw / sw.mean()
            Xs = Xs * sw[:, None]
            y = y * sw

        A = np.hstack([Xs, np.ones((n, 1))])
        reg = np.eye(d + 1) * self.l2
        reg[-1, -1] = 0.0  # 截距不惩罚，否则预测会整体偏向 0
        try:
            w = np.linalg.solve(A.T @ A + reg, A.T @ y)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A.T @ A + reg, A.T @ y, rcond=None)[0]
        self.w = w[:d]
        self.b = float(w[d])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return ((X - self.mean_) / self.scale_) @ self.w + self.b

    def save(self, path: str):
        np.savez(path, w=self.w, b=self.b, mean_=self.mean_, scale_=self.scale_,
                 l2=self.l2)

    @classmethod
    def load(cls, path: str) -> "NumpyRidge":
        z = np.load(path)
        m = cls(l2=float(z["l2"]))
        m.w, m.b, m.mean_, m.scale_ = z["w"], float(z["b"]), z["mean_"], z["scale_"]
        return m
