"""JSONL prediction adapter for the shared A/B/C image.

Protocol (byte-compatible with ``baselines/predict.py``):

    stdin   one JSON object per line: {"task": ..., "inputs": {...}}
    stdout  one JSON object per line, in the same order, flushed immediately

Only ``inputs`` is read.  ``answer`` and ``meta`` may be present in the file
the platform feeds us; they are dropped by whitelist and never inspected.

Deliberate differences from the official single-task adapter:
  * one process serves all three tasks, routed by input schema (see ``router``);
  * a numeric failure on a routed line emits a shaped fallback answer and
    continues, because a single bad line must not zero out a whole submission
    (use ``--strict`` to reproduce the official crash-on-error behaviour).
"""
import argparse
import json
import math
import os
import select
import sys

import numpy as np

try:
    from .net import Model, flatten
    from .router import route
except ImportError:  # running as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from net import Model, flatten
    from router import route

TASKS = ("A", "B", "C")


def _reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def load_models(directory):
    models = {}
    for task in TASKS:
        path = os.path.join(directory, f"{task}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing checkpoint: {path}")
        models[task] = Model(path)
    return models


def bucket(state, task):
    """Per-task state.  A and C must never share counters or windows."""
    return state.setdefault(task, {"n": 0, "labels": {}})


def fallback(model):
    """Shaped but uninformative answer, used only when a routed line fails."""
    if model.task == "regression":
        return {"outputs": {"value": float(model.meta["target_mean"])}}
    share = 1.0 / max(len(model.classes), 1)
    return {"outputs": [{"label": label, "score": share} for label in model.classes]}


def answer(model, inputs):
    paths, values = flatten(inputs)
    model.check_schema(paths)
    x = np.asarray([values], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("non-finite feature in routed line")
    out = model.predict(x)[0]
    if model.task == "regression":
        value = float(out)
        if not math.isfinite(value):
            raise ValueError("non-finite regression output")
        return {"outputs": {"value": value}}
    return {
        "outputs": [
            {"label": label, "score": float(score)}
            for label, score in zip(model.classes, out)
        ]
    }


def iter_lines(idle_exit):
    """Yield stdin lines; if ``idle_exit`` > 0, stop after that long with no input.

    Default is 0 = read until EOF, which is exactly what the official
    ``baselines/predict.py`` does (``for line in sys.stdin``).  That is also the
    safest choice: if the judge closed stdin for the baseline it must close it
    for us, so adding our own timer only creates a way to die first.  A timer
    would also truncate a run where the judge streams in batches with gaps
    longer than the timeout — incomplete output scores worse than slow output.

    The knob is kept for the case where the judge is confirmed to write
    everything and leave the pipe open: then ``--idle-exit 60`` lets us finish
    instead of blocking, which the rules would score as our failure.

    ``select`` works on pipes under Linux (the judged environment).  On Windows
    local runs read files with a real EOF, so the plain loop is used there.
    """
    if idle_exit <= 0 or os.name == "nt":
        for line in sys.stdin:
            yield line
        return
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], idle_exit)
        if not ready:
            print("warn: stdin idle for %gs, exiting" % idle_exit,
                  file=sys.stderr, flush=True)
            return
        line = sys.stdin.readline()
        if not line:
            return
        yield line


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="route and answer A/B/C predictions")
    parser.add_argument("--checkpoints", default=os.path.join(here, "checkpoints"))
    parser.add_argument("--strict", action="store_true", help="crash on the first bad line")
    parser.add_argument(
        "--idle-exit",
        type=float,
        default=0.0,
        help="exit after this many seconds with no stdin input; 0 (default) "
             "reads until EOF, exactly like the official baselines/predict.py",
    )
    args = parser.parse_args(argv)

    models = load_models(args.checkpoints)
    state = {}
    last_task = None

    for line in iter_lines(args.idle_exit):
        # Windows 上导出的文件常带 UTF-8 BOM（\ufeff），它不是空白字符，
        # strip() 删不掉，会让首行 json.loads 直接失败。这里先摘掉。
        line = line.lstrip("\ufeff")
        if not line.strip():
            continue

        # 解析失败也必须吐一行：行数对不上会让平台判整份结果无效，
        # 那比某一行的答案是兜底值严重得多。所以这里不 raise，而是记录后兜底。
        try:
            request = json.loads(line, parse_constant=_reject_constant)
            if not isinstance(request, dict):
                raise ValueError("prediction request must be a JSON object")
            inputs = request.get("inputs")
            if not isinstance(inputs, dict):
                raise ValueError("prediction request requires an inputs object")
            task = route(inputs)      # schema only; never trust the task field
        except Exception as exc:      # noqa: BLE001 - a bad line must not kill the run
            print("warn: unparsable line, emitting fallback: %s" % exc,
                  file=sys.stderr, flush=True)
            task = last_task or "A"
            inputs = None

        model = models[task]
        stats = bucket(state, task)
        stats["n"] += 1
        last_task = task

        try:
            payload = answer(model, inputs) if inputs is not None else fallback(model)
        except Exception as exc:      # noqa: BLE001 - one bad line must not kill the run
            if args.strict:
                raise
            print(f"warn: {task} line {stats['n']}: {exc}", file=sys.stderr, flush=True)
            payload = fallback(model)

        outputs = payload["outputs"]
        if isinstance(outputs, list) and outputs:
            # outputs are emitted in a fixed class order, not sorted by score,
            # so the top class must be found by max(score) rather than outputs[0].
            top = max(outputs, key=lambda o: o["score"])["label"]
            stats["labels"][top] = stats["labels"].get(top, 0) + 1

        print(json.dumps(payload, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
