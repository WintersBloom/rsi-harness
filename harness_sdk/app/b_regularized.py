"""B 包的续训：MAE 损失 + L2 权重衰减，从父检查点出发。

为什么需要这个模块
------------------
B 父检查点（XJTU-SY 轴承 RUL）落在坏的局部解上：继续用 MSE 续训只会越训越差
（15 轮 681 → 300 轮 596，到 1000 轮才 500 出头），而"直接猜训练集中位数"的
RMSE 是 470.8。也就是说这个任务上模型的**有效信息量本来就接近零**，与其让
它输出一条噪声曲线，不如让它诚实地退化成常数预测：把 w1/w2 用 L2 压向 0，
输出就只剩 b2。

两种损失退化成不同的常数：MSE 的最优常数是均值（954 → 验证 536），
MAE 的最优常数是中位数（758 → 验证 470.8）。所以这里选 MAE。

合规性
------
* 起点是平台给出的父检查点（`resume`），权重由优化器写出；
* `steps` / `epochs` 单调增长，`parent_checkpoint_sha256` 可对账；
* 只改「损失函数」和「正则强度」，属于训练期调参，不换结构、不手工改权重；
* 产物仍是官方格式的 `checkpoint.npz`，官方 `Model.load` / `predict.py` 都能读。

改动只发生在损失与梯度上（`fit_regularized`），前向、归一化、裁剪、Adam、
随机打乱顺序全部照抄官方 `baselines/model.py`，便于逐项对照。
"""
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from baselines.data import read_dataset, sha256
from baselines.model import Model, checkpoint_digest, _THREADS

DEFAULT_EPOCHS = 100
DEFAULT_WEIGHT_DECAY = 2.0
DEFAULT_LOSS = "mae"


def fit_regularized(model, x, y, epochs, weight_decay, loss):
    """复刻官方 `Model.fit`，只加两件事：可换损失、可加 L2。

    L2 只作用在 w1/w2（偏置必须能自由移动，否则连"猜一个常数"都做不到）。
    """
    x = model.normalized(x)
    if model.meta["task"] == "regression":
        target = (y - model.meta["target_mean"]) / model.meta["target_scale"]
    else:
        target = y

    for _ in range(epochs):
        rng = np.random.default_rng(np.random.SeedSequence([model.meta["seed"], model.meta["epochs"]]))
        order = rng.permutation(len(x))
        for start in range(0, len(order), model.meta["batch_size"]):
            indices = order[start:start + model.meta["batch_size"]]
            batch, expected = x[indices], target[indices]
            hidden, logits = model.forward(batch)

            error = logits[:, 0] - expected
            if loss == "mse":
                delta = (2 * error / len(indices))[:, None]
            elif loss == "mae":
                delta = (np.sign(error) / len(indices))[:, None]
            elif loss == "huber":
                delta = (np.clip(error, -1.0, 1.0) / len(indices))[:, None]
            else:
                raise ValueError(loss)

            gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
            hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
            gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
            for name in ("w1", "w2"):
                gradients[name] = gradients[name] + weight_decay * model.params[name]

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
        model.meta["epochs"] += 1


def _metrics(model, data):
    predicted = model.predict(data["x"])
    error = predicted - data["y"]
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}


def train_run(profile, train_path, validation_path, output, *, resume=None, epochs=None,
              learning_rate=None, batch_size=None, seed=None, stage_id="baseline",
              loss=DEFAULT_LOSS, weight_decay=DEFAULT_WEIGHT_DECAY, label_provenance=None):
    """与官方 `baselines.train.train_run` 同签名、同产物格式，训练步换成 `fit_regularized`。"""
    output = Path(output)
    if output.exists():
        raise ValueError("output must be a new directory; previous rounds are never overwritten")

    here = Path(__file__).resolve().parent
    config_path = here / "baselines" / "configs" / ("%s.json" % profile)
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    if config.get("task") != "regression":
        raise ValueError("b_regularized 只用于回归包")

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
    model.meta["target_unit"] = config.get("target_unit", "minutes")
    model.validate()

    epochs = DEFAULT_EPOCHS if epochs is None else int(epochs)
    if not 1 <= epochs <= 10000:
        raise ValueError("epochs must be in 1..10000")

    initial_steps, initial_epochs = model.meta["steps"], model.meta["epochs"]
    before = _metrics(model, validation)
    started = time.monotonic()
    records = []
    for _ in range(epochs):
        fit_regularized(model, train["x"], train["y"], 1, weight_decay, loss)
        records.append({"epoch": model.meta["epochs"], "step": model.meta["steps"],
                        "validation": _metrics(model, validation)})
    duration = time.monotonic() - started

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "checkpoint.npz"
    model.save(checkpoint)

    report = {
        "format": "contest.cpu-baseline-report.v1", "profile": profile, "task": "regression",
        "purpose": "development_reference", "official_scoring_ready": False, "device": "cpu",
        "cpu_threads_requested": _THREADS, "numpy_version": np.__version__,
        "python_version": platform.python_version(), "machine": platform.machine(),
        "stage_id": stage_id, "seed": model.meta["seed"],
        "parameter_count": model.meta["parameter_count"], "input_features": len(model.meta["paths"]),
        "train_rows": len(train["ids"]), "validation_rows": len(validation["ids"]),
        "train_sha256": train["sha256"], "validation_sha256": validation["sha256"],
        "profile_sha256": sha256(config_raw), "parent_checkpoint_sha256": parent,
        "checkpoint_sha256": checkpoint_digest(checkpoint),
        "initial_steps": initial_steps, "initial_epochs": initial_epochs,
        "final_steps": model.meta["steps"], "final_epochs": model.meta["epochs"],
        "training_seconds": duration, "learning_rate": model.meta["learning_rate"],
        "batch_size": model.meta["batch_size"], "validation_before": before,
        "validation_after": records[-1]["validation"],
        "configuration": {
            "hidden": model.meta["hidden"], "activation": "tanh", "optimizer": "Adam",
            "adam_beta1": .9, "adam_beta2": .999, "adam_epsilon": 1e-8, "gradient_clip_norm": 5.0,
            "feature_clip": [-10.0, 10.0], "normalization": "initial_training_mean_std",
            "learning_rate": model.meta["learning_rate"], "batch_size": model.meta["batch_size"],
            "requested_epochs": epochs, "classes": model.meta["classes"],
            "loss": loss, "weight_decay": weight_decay},
        "source_sha256": {name: sha256((here / "baselines" / name).read_bytes())
                          for name in ("model.py", "data.py", "predict.py", "train.py")},
        "custom_trainer_sha256": {"b_regularized.py": sha256(Path(__file__).read_bytes())},
        "label_provenance": label_provenance,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_unit": config.get("target_unit", "minutes"),
        "target_origin": "remaining_time_from_lifecycle",
        "target_definition": "(last_snapshot - current_snapshot) * 1 minute; last observed snapshot has RUL 0",
    }
    (output / "training.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    with (output / "history.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps({"stage_id": stage_id, "seed": model.meta["seed"], **record},
                                    allow_nan=False) + "\n")
    return report
