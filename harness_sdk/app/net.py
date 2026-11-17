"""Pure-numpy forward pass for the official checkpoints.

The image ships only this package, so ``baselines`` cannot be imported here.
The forward computation is reproduced exactly from ``baselines/model.py``:

    x  = clip((x - mean) / scale, -10, 10)
    h  = tanh(x @ w1 + b1)
    z  = h @ w2 + b2

    classification : softmax(z)
    regression     : max(0, z * target_scale + target_mean)

Feature extraction is reproduced from ``baselines/data.py::flatten`` so that
the flattened path list matches ``meta["paths"]`` byte for byte; the checkpoint
rejects any other ordering.
"""
import json
import math
import re

import numpy as np

MAX_FEATURES = 4096
MAX_DEPTH = 16

# Names that would leak the target if they ever appeared in an input object.
FORBIDDEN_INPUTS = {
    "truerul",
    "normalizedrul",
    "remainingusefullife",
    "rul",
    "totalsnapshots",
}

_SQUEEZE = re.compile(r"[^a-z0-9]")


def flatten(inputs):
    """Flatten nested inputs into (paths, values), matching the official loader."""
    paths, values = [], []

    def visit(value, path, depth):
        if depth > MAX_DEPTH:
            raise ValueError("input nesting exceeds limit")
        if isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str) or _SQUEEZE.sub("", key.lower()) in FORBIDDEN_INPUTS:
                    raise ValueError("target-derived input field is not allowed")
                visit(value[key], path + [("key", key)], depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [("index", index)], depth + 1)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value):
                raise ValueError("model inputs must contain finite numbers")
            paths.append(json.dumps(path, ensure_ascii=False, separators=(",", ":")))
            values.append(float(value))
        else:
            raise ValueError("model inputs must contain finite numeric features only")

    visit(inputs, [], 0)
    if not values or len(values) > MAX_FEATURES:
        raise ValueError(f"expected 1..{MAX_FEATURES} numeric features")
    return paths, values


class Model:
    """One task's frozen weights, loaded from an official ``checkpoint.npz``."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as archive:
            self.meta = json.loads(str(archive["metadata"]))
            self.w1 = np.array(archive["w1"], dtype=np.float64)
            self.b1 = np.array(archive["b1"], dtype=np.float64)
            self.w2 = np.array(archive["w2"], dtype=np.float64)
            self.b2 = np.array(archive["b2"], dtype=np.float64)
            self.mean = np.array(archive["mean"], dtype=np.float64)
            self.scale = np.array(archive["scale"], dtype=np.float64)

        if np.any(self.scale == 0):
            raise ValueError(f"{path}: zero scale in checkpoint")
        self.task = self.meta["task"]
        self.classes = list(self.meta.get("classes") or [])
        self.paths = list(self.meta.get("paths") or [])

    def forward(self, x):
        """Return (hidden, logits) for a batch of raw features."""
        normalized = np.clip((x - self.mean) / self.scale, -10.0, 10.0)
        hidden = np.tanh(normalized @ self.w1 + self.b1)
        return hidden, hidden @ self.w2 + self.b2

    def predict(self, x):
        """Return (n,) RUL values or (n, n_classes) probabilities."""
        _, logits = self.forward(x)
        if self.task == "regression":
            scaled = logits[:, 0] * self.meta["target_scale"] + self.meta["target_mean"]
            return np.maximum(0.0, scaled)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def check_schema(self, paths):
        if paths != self.paths:
            raise ValueError(
                f"{self.task}: prediction feature schema differs from checkpoint "
                f"({len(paths)} values vs {len(self.paths)} expected)"
            )
