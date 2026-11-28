"""Submission entry: predict is a straight forward pass, train delegates to the loop.

Rules baked in here:
  * predict() returns exactly what the checkpoint computes — no post-processing,
    no rescaling, no calibration against request metadata.
  * a missing/broken checkpoint or a schema mismatch raises with the concrete
    reason instead of substituting a constant, so a loading bug can never be
    disguised as a valid prediction.
  * classification returns a 6-item list, regression returns {"value": float}
  * pick the checkpoint by dataset_key when present, else by input schema
"""
import math
import os
from pathlib import Path

import numpy as np

from net import Model, flatten

PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]

_HERE = Path(__file__).resolve().parent
_CHECKPOINT_DIRS = (
    Path("/app/checkpoints"),
    _HERE / "checkpoints",
)


# ----------------------------------------------------------------- prediction


def _pick_key(sample):
    key = getattr(sample, "dataset_key", None)
    if key in ("A", "B", "C"):
        return key
    inputs = sample.inputs
    if isinstance(inputs, dict):
        if "settings" in inputs:
            return "C"
        features = inputs.get("features")
        if isinstance(features, dict):
            return "B"
        if isinstance(features, list) and len(features) == 128:
            return "A"
    return None


def _candidate_paths(key):
    artifact = os.environ.get("CONTEST_ARTIFACT_PATH")
    if artifact:
        yield Path(artifact)
    for base in _CHECKPOINT_DIRS:
        yield base / ("%s.npz" % key)


_MODELS = {}


def _load(key):
    if key is None:
        raise RuntimeError("cannot tell which dataset this sample belongs to")
    if key in _MODELS:
        return _MODELS[key]
    errors = []
    for path in _candidate_paths(key):
        try:
            model = Model(str(path))
        except Exception as exc:                       # noqa: BLE001 - reported below
            errors.append("%s: %s: %s" % (path, type(exc).__name__, exc))
            continue
        _MODELS[key] = model
        return model
    raise RuntimeError("no usable checkpoint for dataset %s (%s)"
                       % (key, "; ".join(errors) or "no path tried"))


def _apply_regime(x, table, slack=1.5):
    """按该行 settings 所属工况做 z-score（只用到输入里的 settings，不涉及标签）。

    工况可能不在训练集的 7 个里（隐藏流换了一个工况、或 settings 差一点），
    此时若硬用最近的簇统计量，量纲会算错、预测整体偏移——那比不标准化更差，
    而且只发生在我们看不到的数据上。所以超出该簇 radius*slack 时退回全局统计量。
    """
    keys = np.asarray(table["keys"], dtype=np.float64)
    mean = np.asarray(table["mean"], dtype=np.float64)
    std = np.asarray(table["std"], dtype=np.float64)
    radius = np.asarray(table.get("radius") or [np.inf] * len(mean), dtype=np.float64)
    idx = table.get("settings_idx") or [-3, -2, -1]
    idx = [i if i >= 0 else x.shape[1] + i for i in idx]
    settings = np.round(np.asarray(x, dtype=np.float64)[:, idx], 1)
    distance = ((((settings[:, None, :] - keys[None, :, :]) ** 2).sum(axis=2)) ** 0.5)
    nearest = distance.argmin(axis=1)
    out = (x - mean[nearest]) / std[nearest]
    far = distance[np.arange(len(nearest)), nearest] > (radius[nearest] * slack)
    if far.any():
        gmean = np.asarray(table["gmean"], dtype=np.float64)
        gstd = np.asarray(table["gstd"], dtype=np.float64)
        out[far] = (x[far] - gmean) / gstd
    return out


def _answer(sample, model):
    paths, values = flatten(sample.inputs)
    model.check_schema(paths)
    x = np.asarray([values], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("non-finite feature")
    table = model.meta.get("regime_table")
    if table:
        x = _apply_regime(x, table)
    out = model.predict(x)[0]
    if getattr(model, "task", None) == "regression":
        value = float(out)
        if not math.isfinite(value) or value < 0:
            raise ValueError("bad regression value: %r" % (out,))
        return {"outputs": {"value": value}}
    return {"outputs": [{"label": label, "score": float(score)}
                        for label, score in zip(model.classes, out)]}


def predict(sample, model_dir):
    """Return the model's own prediction. Raises on any problem, by design."""
    del model_dir  # the artifact path comes from CONTEST_ARTIFACT_PATH
    model = _load(_pick_key(sample))
    result = _answer(sample, model)
    if sample.task == "regression" and "value" not in result["outputs"]:
        raise RuntimeError("regression result without a value")
    if sample.task == "classification" and not isinstance(result["outputs"], list):
        raise RuntimeError("classification result is not a list")
    return result


# -------------------------------------------------------------------- training


def train(rnd):
    """Delegate to the policy loop: it owns model calls, candidate and finish."""
    from contest_loop import run_round

    run_round(rnd)
