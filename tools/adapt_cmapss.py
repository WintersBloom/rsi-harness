"""把 NASA C-MAPSS FD001（涡扇发动机退化）转成本项目的流格式。

这份数据的性质与 gas 完全不同，接它之前必须先说清楚
--------------------------------------------------
C-MAPSS 是**退化**数据，不是**漂移**数据：
  * 100 台发动机，每台从投运跑到失效（128~362 个 cycle）
  * 单一工况（FD001），传感器读数的分布在整个数据集里基本不变
  * 变的是"这台机器处在寿命的哪个阶段"，不是"世界的规律变了"

所以在它上面跑我们的系统，测的**不是漂移检测**，而是两件别的事：
  1. 读取层能不能处理"回归任务切成二分类"这种标签定义（抽象层验证）
  2. 在没有漂移的数据上，我们的机制会不会**乱动**（误报验证）

如果它在 C-MAPSS 上比冻结基线低，那反而是坏消息——说明我们在没有漂移时
也在瞎折腾。理想结果是两组几乎相同。

标签定义
--------
RUL（剩余寿命）= 该发动机的总 cycle 数 − 当前 cycle。
`label = 1 if RUL < THRESHOLD else 0`（THRESHOLD=30，PHM08 挑战赛的通用做法）。
这个 30 是人为选的，换 50 会得到不同的正例率——这类选择必须在报告里写明。

时间顺序
--------
按 unit 1→100、每台内按 cycle 升序。相邻窗因此是"不同发动机的相同寿命阶段"，
不是"同一条流在时间上的漂移"。这也是它测不了漂移的原因。

输出列：t,segment,f0..f20,label
"""
from __future__ import annotations

import argparse
import csv

RUL_THRESHOLD = 30
N_SENSORS = 21


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/train_FD001.txt")
    ap.add_argument("--out", default="data/cmapss_stream.csv")
    ap.add_argument("--threshold", type=int, default=RUL_THRESHOLD)
    args = ap.parse_args()

    # 先按 unit 收齐，才能算每台的总 cycle 数
    by_unit: dict[int, list[list[float]]] = {}
    with open(args.src, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 26:
                continue
            u = int(parts[0])
            # 第 2 列是 cycle，第 3-5 列是 3 个工况设定（FD001 下近似常数，丢掉），
            # 第 6 列起是 21 个传感器
            by_unit.setdefault(u, []).append([float(x) for x in parts[1:]])

    header = ["t", "segment"] + [f"f{i}" for i in range(N_SENSORS)] + ["label"]
    t = 0
    stats = []
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        for u in sorted(by_unit):
            rows = sorted(by_unit[u], key=lambda r: r[0])  # 按 cycle 升序
            total = len(rows)
            seg = f"unit{u:03d}"
            pos = 0
            for r in rows:
                cycle = int(r[0])
                rul = total - cycle
                y = 1 if rul < args.threshold else 0
                pos += y
                wr.writerow([t, seg] + r[4:4 + N_SENSORS] + [y])
                t += 1
            stats.append((seg, total, pos))

    print(f"{'segment':10s} {'n':>5s} {'pos':>4s} {'rate':>6s}")
    for seg, n, pos in stats[:5]:
        print(f"{seg:10s} {n:5d} {pos:4d} {pos / n:6.3f}")
    print("...")
    print(f"{'TOTAL':10s} {t:5d}")
    tot_pos = sum(p for _, _, p in stats)
    print(f"positive rate overall = {tot_pos / t:.3f}")


if __name__ == "__main__":
    main()
