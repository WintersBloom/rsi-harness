"""合成工业漂移数据流（占位用，等官方数据包到手后由真实数据替代）。

设计目标：让"静态基线会失准"这件事在数据上真实发生，用来验证闭环是否真的有用。

形态：表格数据，二分类（0=正常，1=故障），12 个特征，按时间顺序分 4 段：
  A 平稳期      —— 训练集与测试集同分布，任何模型都能学
  B 渐变漂移    —— 协变量漂移：特征均值缓慢平移，分布变了但规律没变
  C 突变概念漂移 —— 决策边界改变：同样的特征读数，故障含义变了
  D 回归期      —— 分布回到 A，考验系统能否"恢复"而不是被 C 段带偏

用法：
    python src/synth.py --out data/synth_stream.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import random

N_FEATURES = 12
SEGMENTS = [("A_stable", 3000), ("B_gradual", 3000), ("C_concept", 3000), ("D_recover", 3000)]


def _make_weights(rng: random.Random, n=N_FEATURES):
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _rotate(w, rng: random.Random, strength: float):
    """决策边界旋转：模拟"同样的读数，故障含义变了"的概念漂移。"""
    out = []
    for wi in w:
        out.append(wi * (1.0 - strength) + rng.uniform(-1.0, 1.0) * strength)
    return out


def _write(rows, out_path):
    header = ["t", "segment"] + [f"f{j}" for j in range(N_FEATURES)] + ["label"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)
    pos = sum(r[-1] for r in rows)
    return len(rows), pos


def _calibrate(raw, pos_rate):
    """全局分位阈值：把正例比例压到 pos_rate 附近（故障是稀有事件）。"""
    probs = sorted(r[3] for r in raw)
    cut = probs[int(len(probs) * (1.0 - pos_rate))]
    return [[t_, seg] + [round(f, 5) for f in feats] + [1 if p >= cut else 0]
            for t_, seg, feats, p in raw]


def generate_gradual(seed=11, out_path="data/g_gradual.csv", pos_rate=0.08,
                     n_seg=12, seg_len=1000, step=0.12, covariate=0.15):
    """渐进概念漂移：决策边界每段只转一小步，连续累积。

    存在的意义：**造一条"全量重训明显优于冻结基线"的流**。
    原 synth_stream 的 C 段是突变式（旋转 0.75 + 翻符号），旧数据完全反向，
    全量重训反而被毒化，于是官方 15 分要的归一化增益 G 根本没有分母
    （G = (ours − baseline) / (全量重训 − baseline)）。

    每段转 0.12 弧度、共 12 段，总漂移约 1.44 弧度但不突变。

    `covariate`：每段特征均值平移 0.15σ（交替正负）。
    **这一项是必须的，不是修饰**：实测发现只转决策边界、特征分布保持不变时，
    PSI 全程 ≤ 0.07（level 0），监测器完全看不出问题，系统一次都不更新。
    纯概念漂移下边缘分布不变是数学必然——PSI 看不到 P(Y|X) 的变化。
    真实工业里传感器漂移与工艺变更通常同时发生，所以这里也同时建模。
    代价：这条流证明的是"两者同时发生时我们能跟上"，不是"纯概念漂移也行"。
    """
    rng = random.Random(seed)
    w = _make_weights(rng)
    raw, t = [], 0
    for s in range(n_seg):
        seg_name = f"G{s + 1:02d}"
        shift = covariate * s
        for i in range(seg_len):
            feats = [rng.gauss(0.0, 1.0) + (shift if j % 2 == 0 else -shift)
                     for j in range(N_FEATURES)]
            raw.append([t, seg_name, feats, _sigmoid(sum(wi * f for wi, f in zip(w, feats)) - 0.4)])
            t += 1
        w = _rotate(w, rng, step)
    return _write(_calibrate(raw, pos_rate), out_path)


def generate_recurring(seed=12, out_path="data/g_recurring.csv", pos_rate=0.08,
                       n_patterns=3, rounds=3, seg_len=1400):
    """周期性工况：同一批"模式"循环出现（季节/批次/产线切换）。

    存在的意义：这是全量重训最占优的形态——它见过所有模式，而窗口重训只能
    记住最近一两个。如果我们的方案在这种流上也能接近天花板，说明"遗忘"确实
    被控制在可接受范围；如果不能，就是方案的真实短板。
    """
    rng = random.Random(seed)
    # 模式之间共享主干、只在少数特征上换方向：让"记住多种模式"成为可能
    base = _make_weights(rng)
    patterns = []
    for k in range(n_patterns):
        w = list(base)
        for j in range(k, N_FEATURES, n_patterns):
            w[j] = -w[j] * 1.2
        patterns.append(w)

    raw, t = [], 0
    for r in range(rounds):
        for k, w in enumerate(patterns):
            seg_name = f"R{k + 1}"
            for i in range(seg_len):
                feats = [rng.gauss(0.0, 1.0) for _ in range(N_FEATURES)]
                raw.append([t, seg_name, feats,
                            _sigmoid(sum(wi * f for wi, f in zip(w, feats)) - 0.4)])
                t += 1
    return _write(_calibrate(raw, pos_rate), out_path)


def generate(seed: int = 7, out_path: str = "data/synth_stream.csv", pos_rate: float = 0.08,
             drift_strength: float = 1.0):
    """drift_strength：漂移剧烈程度。
    0.4 ≈ 温和（真实工业的缓慢退化，边界只挪一点）
    1.0 ≈ 当前基准（突变式概念漂移，AP 从 0.99 掉到 0.08）
    1.5 ≈ 剧烈（边界几乎重画）
    """
    rng = random.Random(seed)
    rotate_deg = min(0.95, 0.75 * drift_strength)
    flip_every = max(2, int(round(3 / max(drift_strength, 0.1))))
    w_base = _make_weights(rng)
    raw = []
    t = 0
    for seg_name, n in SEGMENTS:
        w = w_base
        if seg_name == "C_concept":
            # 概念漂移：边界旋转 + 部分特征符号翻转，幅度由 drift_strength 控制
            w = _rotate(w_base, rng, rotate_deg)
            w = [wi * (-1.0 if i % flip_every == 0 else 1.0) for i, wi in enumerate(w)]
        for i in range(n):
            progress = i / max(1, n - 1)
            feats = []
            for j in range(N_FEATURES):
                base = rng.gauss(0.0, 1.0)
                if seg_name == "B_gradual":
                    # 渐变协变量漂移：均值与方差逐渐平移
                    base = base * (1.0 + 0.6 * progress) + 0.9 * progress * (1 if j % 2 == 0 else -1)
                elif seg_name == "C_concept":
                    base = base * 1.15 + 0.35
                elif seg_name == "D_recover":
                    base = base * 0.98
                feats.append(base)
            logit = sum(wi * f for wi, f in zip(w, feats)) - 0.4
            raw.append([t, seg_name, feats, _sigmoid(logit)])
            t += 1

    # 全局分位阈值校准：把正例比例压到 pos_rate 附近（故障是稀有事件）
    probs = sorted(r[3] for r in raw)
    cut = probs[int(len(probs) * (1.0 - pos_rate))]

    rows = []
    for t_, seg_name, feats, p in raw:
        rows.append([t_, seg_name] + [round(f, 5) for f in feats] + [1 if p >= cut else 0])

    header = ["t", "segment"] + [f"f{j}" for j in range(N_FEATURES)] + ["label"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)
    pos = sum(r[-1] for r in rows)
    return len(rows), pos


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth_stream.csv")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--drift-strength", type=float, default=1.0, help="漂移剧烈程度：0.4 温和 / 1.0 基准 / 1.5 剧烈")
    ap.add_argument("--mode", default="step", choices=["step", "gradual", "recurring"],
                    help="step=突变概念漂移（原版）/ gradual=渐进漂移 / recurring=周期工况")
    ap.add_argument("--segments", type=int, default=12, help="gradual 模式的段数")
    ap.add_argument("--seg-len", type=int, default=1000, help="每段样本数")
    ap.add_argument("--step", type=float, default=0.12, help="gradual 模式每段旋转弧度")
    ap.add_argument("--covariate", type=float, default=0.15,
                    help="gradual 模式每段特征均值平移量（单位：σ）。0 = 纯概念漂移，"
                         "但实测 PSI 全程 ≤0.07、监测器看不见，系统一次都不更新")
    args = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.mode == "gradual":
        n, pos = generate_gradual(seed=args.seed, out_path=args.out,
                                  n_seg=args.segments, seg_len=args.seg_len, step=args.step,
                                  covariate=args.covariate)
    elif args.mode == "recurring":
        n, pos = generate_recurring(seed=args.seed, out_path=args.out, seg_len=args.seg_len)
    else:
        n, pos = generate(seed=args.seed, out_path=args.out, drift_strength=args.drift_strength)
    print(f"wrote {args.out}: {n} rows, {pos} positives ({pos / n:.2%})")
