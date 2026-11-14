"""模型注册表：每次重训产出一个版本，上线前必须过"评估门"，上线后能一键回滚。

为什么单独做一个模块：
  门槛 ③ 是"综合性能不低于静态基线"。最容易被自己搞砸的方式是——重训完直接替换，
  结果新模型因为样本太少/漂移判断错反而更差。有个注册表 + 回滚，才有"保险丝"。
"""
from __future__ import annotations

import json
import os

from src.predictor import NumpyLR


def _suffix(model) -> str:
    """NumpyLR 存 npz（纯 numpy、可读），其余（如 SkGBDT）存 pickle。"""
    return ".pkl" if type(model).__name__ != "NumpyLR" else ".npz"


def _load_any(path: str):
    if path.endswith(".pkl"):
        from src.predictor_sk import SkGBDT
        return SkGBDT.load(path)
    return NumpyLR.load(path)


class Registry:
    def __init__(self, root: str = "registry"):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.state_path = os.path.join(root, "state.json")
        self.state = {"active": None, "versions": []}
        if os.path.exists(self.state_path):
            with open(self.state_path, encoding="utf-8") as fh:
                self.state = json.load(fh)

    def _save_state(self) -> None:
        with open(self.state_path, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, ensure_ascii=False, indent=2)

    def register(self, model, threshold: float, meta: dict | None = None) -> int:
        ver = len(self.state["versions"]) + 1
        path = os.path.join(self.root, f"v{ver}{_suffix(model)}")
        model.save(path, meta={"version": ver, "threshold": threshold, **(meta or {})})
        self.state["versions"].append({
            "version": ver, "path": path, "threshold": threshold, "meta": meta or {},
            "active": False,
        })
        self._save_state()
        return ver

    def activate(self, ver: int) -> None:
        for v in self.state["versions"]:
            v["active"] = (v["version"] == ver)
        self.state["active"] = ver
        self._save_state()

    def rollback(self) -> int | None:
        """回到上一个版本。返回回滚后的版本号；没有可回滚的返回 None。"""
        if not self.state["versions"]:
            return None
        prev = [v for v in self.state["versions"] if not v["active"]]
        if not prev:
            return None
        target = prev[-1]["version"]
        self.activate(target)
        return target

    def active(self):
        ver = self.state.get("active")
        if ver is None:
            return None
        rec = next(v for v in self.state["versions"] if v["version"] == ver)
        model = _load_any(rec["path"])
        return model, float(rec["threshold"]), ver

    def list_versions(self) -> list[dict]:
        return [{k: v[k] for k in ("version", "threshold", "active")} for v in self.state["versions"]]
