"""静态基线：模拟"部署后不再更新"的模型。

赛题门槛是"综合性能不低于组委会静态基线"。要证明自己更强，先得有一个可复现的静态基线。
这里用最朴素的设定：只用最初一段有标签的数据训练一次，之后全程不再更新，看它在漂移下怎么垮。

用法：
    python src/baseline_static.py --data data/synth_stream.csv --out experiments/baseline_static.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.io_adapter import load_stream
from src.metrics import evaluate
from src.predictor import NumpyLR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_stream.csv")
    ap.add_argument("--train-n", type=int, default=1500, help="初始训练集大小（模拟上线时拿到的历史数据）")
    ap.add_argument("--out", default="experiments/baseline_static.json")
    args = ap.parse_args()

    stream = load_stream(args.data)
    train = stream.slice(0, args.train_n)
    model = NumpyLR(lr=0.05, epochs=300, l2=1e-3, class_weight=True, seed=0)
    model.fit(train.X, train.y)

    result = {
        "config": {"train_n": args.train_n, "lr": model.lr, "epochs": model.epochs,
                   "l2": model.l2, "class_weight": model.class_weight},
        "train_loss_tail": model.history[-3:],
        "overall": evaluate(stream.y, model.predict_proba(stream.X)),
        "by_segment": {},
        "windows": [],
    }

    # 分段评估：静态模型在哪一段开始崩，是后面"漂移监测要不要触发"的直接依据
    for name in stream.segments():
        seg = stream.by_segment(name)
        result["by_segment"][name] = evaluate(seg.y, model.predict_proba(seg.X))

    # 滚动窗口：更接近评测时的"隐藏测试流逐段喂入"
    for start, win in stream.windows(500):
        r = evaluate(win.y, model.predict_proba(win.X))
        r["start"] = start
        r["segment"] = win.segment[0]
        result["windows"].append(r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(f"{'segment':<12}{'n':>6}{'pos':>6}{'F1':>8}{'AUC':>8}{'AP':>8}")
    for name, r in result["by_segment"].items():
        print(f"{name:<12}{r['n']:>6}{r['pos']:>6}{r['f1']:>8}{r['auc']:>8}{r['ap']:>8}")
    print(f"\noverall F1={result['overall']['f1']} AUC={result['overall']['auc']} AP={result['overall']['ap']}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
