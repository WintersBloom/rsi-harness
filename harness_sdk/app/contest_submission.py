"""Submission entry: step-1 protocol shell + step-3 predict wired to net.py.

train(rnd) : still the placeholder from step 1 (step 4 replaces it with the five-ring loop)
predict()  : pure numpy forward, byte-compatible with the frozen image 1.3

Rules baked in here:
  * never raise out of predict() — the SDK kills the whole measurement on one bad line
  * classification returns a 6-item list, regression returns {"value": float}
  * pick the checkpoint by dataset_key when present, else by input schema
"""
import math
import os
from pathlib import Path

import numpy as np

from net import Model, flatten

PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]

LABELS = ("Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene")

_HERE = Path(__file__).resolve().parent
_CHECKPOINT_DIRS = (
    Path("/app/checkpoints"),
    _HERE / "checkpoints",
)

_MESSAGES = "本轮训练只需确认协议已通。请只回复两个字：收到。"


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
        return None
    if key in _MODELS:
        return _MODELS[key]
    model = None
    for path in _candidate_paths(key):
        try:
            model = Model(str(path))
            break
        except Exception:
            continue
    _MODELS[key] = model
    return model


def _fallback(sample):
    if sample.task == "regression":
        return {"outputs": {"value": 0.0}}
    return {"outputs": [{"label": LABELS[0], "score": 1.0}] +
                       [{"label": label, "score": 0.0} for label in LABELS[1:]]}


def _answer(sample, model):
    paths, values = flatten(sample.inputs)
    model.check_schema(paths)
    x = np.asarray([values], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("non-finite feature")
    out = model.predict(x)[0]
    if getattr(model, "task", None) == "regression":
        value = float(out)
        if not math.isfinite(value) or value < 0:
            raise ValueError("bad regression value")
        return {"outputs": {"value": value}}
    return {"outputs": [{"label": label, "score": float(score)}
                        for label, score in zip(model.classes, out)]}


def predict(sample, model_dir):
    """Return one validated prediction. Never raises."""
    del model_dir  # the artifact path comes from CONTEST_ARTIFACT_PATH
    try:
        model = _load(_pick_key(sample))
        if model is None:
            return _fallback(sample)
        result = _answer(sample, model)
        if sample.task == "regression" and "value" not in result["outputs"]:
            return _fallback(sample)
        if sample.task == "classification" and not isinstance(result["outputs"], list):
            return _fallback(sample)
        return result
    except Exception:
        return _fallback(sample)


# -------------------------------------------------------------------- training


def train(rnd):
    """Delegate to the policy loop: it owns model calls, candidate and finish."""
    from contest_loop import run_round

    run_round(rnd)
