"""把 UCI gas sensor array drift（libsvm、128 维、6 类）转成本项目的流格式。

为什么用这份数据
----------------
它是目前能找到的、最接近赛题设定的**真实**漂移流：
  * 13910 条测量、128 维（16 个传感器 × 8 个特征）
  * 按时间分成 10 个 batch（2007-10 至 2011-04，跨 36 个月）
  * 漂移来源是传感器老化——同一浓度的同一气体，几年后读数整体偏移

这属于**协变量漂移**（P(X) 变、P(Y|X) 不变），正好和我们合成流里的 B 段同类，
但它是真实测量而不是公式生成的。用它只能验证"代码能不能吃真实脏数据"和
"监测器在真实漂移上还灵不灵"，**证明不了方案在赛题数据上的好坏**——后者
必须等官方 A 包。

正例怎么定
----------
原始是 6 分类。这里取 class 3 为正例、其余为负例：
  * 全局占比 11.8%，量级接近合成流的 8%
  * 各 batch 间从 7.5% 波动到 25%，天然带一层标签漂移（这一点比合成流更真实）
代价是把它从"6 分类"降成了"二分类"，标签语义变了，所以**不要拿它的 F1
和合成流的 F1 横向比**。

为什么每个 batch 内要打乱
------------------------
第一次直接按文件顺序拼，结果 25 个窗里有 10 个窗**一条正例都没有**（F1=0、
AUC=nan）。原因是原始文件在一个 batch 内是**按类别连续排列**的（先 90 条
class1、再 98 条 class2……），而一个 batch 是同一个月的测量，类别之间没有
时间先后。按 500 条切窗，切出来的自然是一整块同类。

所以：**batch 内部打乱（固定 seed 保证可复现），batch 之间保持时间顺序**。
这才是这条流真实的形态——漂移发生在"月与月之间"，不是"条与条之间"。

输出列：t,segment,f0..f127,label（与 src/io_adapter.py 的约定一致）
"""
from __future__ import annotations

import argparse
import csv
import random
import zipfile

POSITIVE_CLASS = 3
N_FEATURES = 128
SHUFFLE_SEED = 20261110


def parse_batch(raw: bytes) -> list[tuple[int, list[float]]]:
    """libsvm 一行：`label idx:val idx:val ...`，idx 从 1 开始，缺失特征补 0。"""
    rows = []
    for line in raw.decode("utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        label = int(parts[0])
        feats = [0.0] * N_FEATURES
        for tok in parts[1:]:
            idx_s, val_s = tok.split(":")
            feats[int(idx_s) - 1] = float(val_s)
        rows.append((label, feats))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/uci_gas.zip")
    ap.add_argument("--out", default="data/uci_gas_stream.csv")
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    header = ["t", "segment"] + [f"f{i}" for i in range(N_FEATURES)] + ["label"]

    t = 0
    stats = []
    rng = random.Random(SHUFFLE_SEED)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        for b in range(1, 11):
            seg = f"batch{b}"
            rows = parse_batch(zf.read(f"Dataset/batch{b}.dat"))
            rng.shuffle(rows)
            pos = 0
            for label, feats in rows:
                y = 1 if label == POSITIVE_CLASS else 0
                pos += y
                wr.writerow([t, seg] + feats + [y])
                t += 1
            stats.append((seg, len(rows), pos))

    print(f"{'segment':10s} {'n':>6s} {'pos':>5s} {'rate':>7s}")
    for seg, n, pos in stats:
        print(f"{seg:10s} {n:6d} {pos:5d} {pos / n:7.3f}")
    print(f"{'TOTAL':10s} {t:6d}")


if __name__ == "__main__":
    main()
