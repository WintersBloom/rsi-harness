"""梯度提升模型（sklearn 版）—— 表格数据 + 漂移场景下，它比线性模型强得多。

为什么值得多一个依赖：
  线性模型在概念漂移时只能整体旋转边界，而 GBDT 能表达"特征之间的交互突变"，
  在 C 段这种"同样读数、故障含义变了"的场景里，重训后恢复得更快。
  代价是单次训练从毫秒级变成百毫秒级——仍远低于预算，且换取的是真实的性能上限。

接口与 NumpyLR 完全一致（fit / predict_proba / predict / save / load），
所以 main.py 可以把两者当同一种东西用；没有 sklearn 的环境会自动退回 NumpyLR。
"""
from __future__ import annotations

import pickle

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_SKLEARN = True
except Exception:      # 评测环境若没装 sklearn，整个模块仍然可导入（只是不可用）
    HAS_SKLEARN = False


class SkGBDT:
    def __init__(self, max_iter: int = 120, learning_rate: float = 0.08,
                 max_leaf_nodes: int = 15, l2: float = 1e-3, seed: int = 0):
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn 不在当前环境里；请用 NumpyLR，或 pip install scikit-learn")
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.l2 = l2
        self.seed = seed
        self._m = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2, random_state=seed, early_stopping=False,
        )
        self._fitted = False

    def fit(self, X, y, sample_weight=None) -> "SkGBDT":
        self._m.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int),
                    sample_weight=sample_weight)
        self._fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._m.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    # ---- 与 NumpyLR 相同的持久化接口（registry 用） ----
    def save(self, path: str, meta: dict | None = None) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"model": self._m, "meta": meta or {}}, fh, protocol=4)

    @staticmethod
    def load(path: str) -> "SkGBDT":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        obj = SkGBDT.__new__(SkGBDT)
        obj._m = blob["model"]
        obj._fitted = True
        return obj
