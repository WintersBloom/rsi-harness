"""Full validation run of the SDK image: sdk5, --network none, all rows.

Produces the numbers that go into README. Read-only w.r.t. repo.
"""

import json
import math
import os
import subprocess
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = "zhixun-harness:sdk6"

# 官方数据不在本仓库里（版权与体积），需要外部提供：
#   CONTEST_DATA=<含 official/ 的目录>  python tools/verify_sdk5_full.py
# 或把三包解到 ./data/official/ 下。找不到时给一条明确的报错，而不是 FileNotFoundError。
DATA = os.environ.get("CONTEST_DATA") or os.path.join(WS, "data")
PACKS = {
    "A": ("official/cpu-a-guide-20261110-v1/public/validation.jsonl", "classification"),
    "B": ("official/cpu-b-rul-20261110-v1/public/validation.jsonl", "regression"),
    "C": ("official/cpu-c-guide-20261110-v1/public/validation.jsonl", "regression"),
}


def macro_f1(y_true, y_pred, labels):
    f1s = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec))
    return sum(f1s) / len(f1s)


def main():
    for key, (rel, task) in PACKS.items():
        path = os.path.join(DATA, rel)
        if not os.path.exists(path):
            print(
                f"{key}: 找不到验证集 {path}\n"
                f"    官方数据不在本仓库里。请设置 CONTEST_DATA 指向含 official/ 的目录：\n"
                f"    CONTEST_DATA=/path/to/data python tools/verify_sdk5_full.py"
            )
            continue
        rows, answers = [], []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                o = json.loads(line)
                rows.append(json.dumps({"task": task, "inputs": o["inputs"]}, ensure_ascii=False))
                answers.append(o["answer"])

        p = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "-i", IMG, "predict"],
            input="\n".join(rows) + "\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if p.returncode != 0:
            print(f"{key} FAIL rc={p.returncode}\n{p.stderr[:800]}")
            continue
        outs = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
        if len(outs) != len(rows):
            print(f"{key} 行数不符 in={len(rows)} out={len(outs)}")
            continue

        if task == "classification":
            labels = sorted(set(answers))
            pred = [max(o["outputs"], key=lambda x: x["score"])["label"] for o in outs]
            acc = sum(1 for a, b in zip(answers, pred) if a == b) / len(answers)
            print(f"{key}: {len(rows)} 行  Macro-F1={macro_f1(answers, pred, labels):.4f}  acc={acc:.4f}")
        else:
            vals = [float(o["outputs"]["value"]) for o in outs]
            rmse = math.sqrt(sum((float(a) - v) ** 2 for a, v in zip(answers, vals)) / len(vals))
            mae = sum(abs(float(a) - v) for a, v in zip(answers, vals)) / len(vals)
            print(f"{key}: {len(rows)} 行  RMSE={rmse:.4f}  MAE={mae:.4f}")


if __name__ == "__main__":
    main()
