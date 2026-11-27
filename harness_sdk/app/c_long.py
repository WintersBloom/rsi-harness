"""C 包的长程续训：低学习率 + 逐轮验证 + 回滚到最好那一轮 + 塌陷护栏。

为什么需要它（都是实测，不是推测）：
- 官方 15 轮 52.87；我们 300 轮（lr 0.006）50.65；lr 0.002 跑 900 轮 50.29，
  其中第 997 轮最好 50.07 —— 最后一轮并不等于最好那一轮，所以必须能回滚。
- 同一配置换个 seed（7）或把轮数拉到 1800，会塌成"常数预测器"（min==max，
  验证集 RMSE 79.4）。这种产物本身合法，但没有任何模型价值，必须拦在提交之前。
- C 的训练损失、优化器、打乱顺序全部照抄官方 `Model.fit`（MSE、Adam、
  梯度裁剪 5.0、epoch 索引打乱），只是多了"每轮验证 + 留最优 + 护栏"。
"""
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from baselines.data import read_dataset
from baselines.model import Model, checkpoint_digest

DEFAULT_EPOCHS = 900
DEFAULT_LR = 0.002
# 预测标准差低于这个值就算"塌成常数"，本轮不提交。
MIN_PRED_STD = 1.0
# 同一个发动机的预测值应随周期单调不增；违反时罚一项，权重见 MONO_LAMBDA。
MONO_LAMBDA = 0.5
MONO_PAIRS = 256


def _consecutive_pairs(ids):
    """返回同一台发动机相邻两个周期的样本下标对（前, 后）。"""
    unit = np.array([int(re.search(r"_u(\d+)_c(\d+)$", i).group(1)) for i in ids])
    cycle = np.array([int(re.search(r"_u(\d+)_c(\d+)$", i).group(2)) for i in ids])
    head = np.where((unit[1:] == unit[:-1]) & (cycle[1:] == cycle[:-1] + 1))[0]
    return head, head + 1


def _fit_epoch(model, x, y, early=None, late=None, lam=0.0):
    """一个 epoch 的官方训练逻辑（MSE + Adam + 裁剪 5.0），原样照抄。"""
    x = model.normalized(x)
    target = (y - model.meta["target_mean"]) / model.meta["target_scale"]
    rng = np.random.default_rng(np.random.SeedSequence([model.meta["seed"], model.meta["epochs"]]))
    order = rng.permutation(len(x))
    for start in range(0, len(order), model.meta["batch_size"]):
        indices = order[start:start + model.meta["batch_size"]]
        batch, expected = x[indices], target[indices]
        hidden, logits = model.forward(batch)

        error = logits[:, 0] - expected
        delta = (2 * error / len(indices))[:, None]

        gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
        hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
        gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))

        if lam > 0 and early is not None and len(early):
            pick = rng.choice(len(early), size=min(MONO_PAIRS, len(early)), replace=False)
            ei, li = early[pick], late[pick]
            pair_x = x[np.concatenate([ei, li])]
            pair_hidden, pair_logits = model.forward(pair_x)
            count = len(pick)
            z_early, z_late = pair_logits[:count, 0], pair_logits[count:, 0]
            violated = (z_late - z_early) > 0
            weight = violated.astype(float) * (lam / count)
            penalty_delta = np.concatenate([-weight, weight])[:, None]
            gradients["w2"] = gradients["w2"] + pair_hidden.T @ penalty_delta
            gradients["b2"] = gradients["b2"] + penalty_delta.sum(axis=0)
            pair_delta = (penalty_delta @ model.params["w2"].T) * (1 - pair_hidden ** 2)
            gradients["w1"] = gradients["w1"] + pair_x.T @ pair_delta
            gradients["b1"] = gradients["b1"] + pair_delta.sum(axis=0)

        norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
        if not math.isfinite(norm):
            raise ValueError("nonfinite gradient")
        model.meta["steps"] += 1
        for name, gradient in gradients.items():
            gradient *= min(1.0, 5.0 / max(norm, 1e-12))
            model.first[name] = .9 * model.first[name] + .1 * gradient
            model.second[name] = .999 * model.second[name] + .001 * gradient ** 2
            first = model.first[name] / (1 - .9 ** model.meta["steps"])
            second = model.second[name] / (1 - .999 ** model.meta["steps"])
            model.params[name] -= model.meta["learning_rate"] * first / (np.sqrt(second) + 1e-8)
        if not all(np.all(np.isfinite(p)) for p in model.params.values()):
            raise ValueError("nonfinite model state")
    model.meta["epochs"] += 1


def _metrics(model, data):
    predicted = model.predict(data["x"])
    error = predicted - data["y"]
    return {"mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "pred_std": float(np.std(predicted))}


def _snapshot(model):
    return {"params": {k: v.copy() for k, v in model.params.items()},
            "first": {k: v.copy() for k, v in model.first.items()},
            "second": {k: v.copy() for k, v in model.second.items()},
            "epochs": int(model.meta["epochs"]), "steps": int(model.meta["steps"])}


def _restore(model, snap):
    for key in model.params:
        model.params[key] = snap["params"][key].copy()
        model.first[key] = snap["first"][key].copy()
        model.second[key] = snap["second"][key].copy()
    model.meta["epochs"] = snap["epochs"]
    model.meta["steps"] = snap["steps"]


def train_run(profile, train_path, validation_path, output, *, resume=None, epochs=None,
              learning_rate=None, batch_size=None, seed=None, stage_id="baseline",
              keep_best=True, min_pred_std=MIN_PRED_STD, label_provenance=None,
              monotonic_lambda=MONO_LAMBDA):
    """与官方 `baselines.train.train_run` 同签名、同产物；内部多做了逐轮验证与回滚。"""
    output = Path(output)
    if output.exists():
        raise ValueError("output must be a new directory; previous rounds are never overwritten")

    here = Path(__file__).resolve().parent
    config_path = here / "baselines" / "configs" / ("%s.json" % profile)
    config = json.loads(config_path.read_bytes())
    if config.get("task") != "regression":
        raise ValueError("c_long 只用于回归包")

    train = read_dataset(train_path, "regression", [])
    validation = read_dataset(validation_path, "regression", [], train["paths"])
    if set(train["ids"]) & set(validation["ids"]):
        raise ValueError("training and validation sample IDs overlap")

    parent = checkpoint_digest(resume) if resume else None
    if resume:
        model = Model.load(resume)
    else:
        model = Model("regression", train["paths"], [], train["x"], train["y"],
                      hidden=config["hidden"], seed=config["seed"],
                      learning_rate=config["learning_rate"], batch_size=config["batch_size"])
    if learning_rate is not None:
        model.meta["learning_rate"] = float(learning_rate)
    if batch_size is not None:
        model.meta["batch_size"] = int(batch_size)
    if seed is not None:
        model.meta["seed"] = int(seed)
    model.meta["profile"] = profile
    model.meta["target_unit"] = config.get("target_unit", "unit")
    model.validate()

    epochs = DEFAULT_EPOCHS if epochs is None else int(epochs)
    if not 1 <= epochs <= 10000:
        raise ValueError("epochs must be in 1..10000")

    lam = float(monotonic_lambda or 0.0)
    early, late = _consecutive_pairs(train["ids"]) if lam > 0 else (None, None)
    initial_steps, initial_epochs = int(model.meta["steps"]), int(model.meta["epochs"])
    before = _metrics(model, validation)
    started = time.monotonic()

    best = _snapshot(model)
    best_metrics = dict(before)
    best_epoch = initial_epochs
    records = []
    for _ in range(epochs):
        _fit_epoch(model, train["x"], train["y"], early, late, lam)
        current = _metrics(model, validation)
        records.append({"epoch": int(model.meta["epochs"]), "step": int(model.meta["steps"]),
                        "validation": {"mae": current["mae"], "rmse": current["rmse"],
                                       "pred_std": current["pred_std"]}})
        if current["rmse"] < best_metrics["rmse"]:
            best, best_metrics, best_epoch = _snapshot(model), dict(current), int(model.meta["epochs"])

    final = _metrics(model, validation)
    if keep_best:
        _restore(model, best)

    kept = _metrics(model, validation)
    collapsed = kept["pred_std"] < float(min_pred_std)
    model.meta["best_epoch"] = int(best_epoch)
    model.meta["best_validation_rmse"] = float(best_metrics["rmse"])
    model.validate()

    output.mkdir(parents=True)
    model.save(output / "checkpoint.npz")
    elapsed = time.monotonic() - started
    report = {
        "format": "contest.cpu-baseline-report.v1",
        "profile": profile,
        "task": "regression",
        "purpose": "development_reference",
        "official_scoring_ready": False,
        "device": "cpu",
        "stage_id": stage_id,
        "seed": int(model.meta["seed"]),
        "parameter_count": int(model.meta["parameter_count"]),
        "input_features": len(model.meta["paths"]),
        "train_rows": int(len(train["x"])),
        "validation_rows": int(len(validation["x"])),
        "train_sha256": train["sha256"],
        "validation_sha256": validation["sha256"],
        "parent_checkpoint_sha256": parent,
        "checkpoint_sha256": checkpoint_digest(output / "checkpoint.npz"),
        "initial_steps": initial_steps,
        "final_steps": int(model.meta["steps"]),
        "initial_epochs": initial_epochs,
        "final_epochs": int(model.meta["epochs"]),
        "training_seconds": round(elapsed, 3),
        "learning_rate": float(model.meta["learning_rate"]),
        "batch_size": int(model.meta["batch_size"]),
        "validation_before": {"mae": before["mae"], "rmse": before["rmse"]},
        "validation_after": {"mae": kept["mae"], "rmse": kept["rmse"]},
        "validation_final_epoch": {"mae": final["mae"], "rmse": final["rmse"]},
        "kept_best_epoch": int(best_epoch),
        "trainer": "c_long",
        "monotonic_lambda": float(lam),
        "monotonic_pairs": int(0 if early is None else len(early)),
        "guard": {"collapsed": bool(collapsed), "pred_std": float(kept["pred_std"]),
                  "min_pred_std": float(min_pred_std)},
    }
    if label_provenance is not None:
        report["label_provenance"] = label_provenance
    (output / "training.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "history.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(dict(record, stage_id=stage_id, seed=report["seed"]),
                                    sort_keys=True) + "\n")
    return report
