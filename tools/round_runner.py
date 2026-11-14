"""十轮自适应训练流水线：每轮一个假设、一次改动、一次评估、一次记账。

设计原则
--------
1. **只用官方 CLI 参数**（lr / batch_size / epochs / 续训 vs 重训），不改 `baselines/` 源码。
   理由：`training.json` 会记录 `source_sha256`，改源码会让证据链说不清"是不是官方 baseline"。
2. **不用 validation 拟合任何参数**。validation 只读、只算分、只用于"选动作"（官方
   `TRAINING_GUIDE.md:9` 明确允许"用于衡量改动"）。
3. **每轮必须留下一条可验证的因果链**：decision（为什么改）→ config（改了什么）
   → before/after（改完怎样）→ accepted（接受还是回退）。

动作空间（都不改维度、都能续训）
------------------------------
    continue20   续训 20 epoch（同 lr）
    continue10   小步续训 10 epoch
    restart20    从官方 checkpoint 重新初始化训练 20 epoch（同一 seed，不同起点）
    lr_lo        重训 20 epoch，lr=0.0015
    lr_hi        重训 20 epoch，lr=0.006
    bs64         重训 20 epoch，batch_size=64
    stop         宣布收敛，不训练

用法
----
    python tools/round_runner.py --profile A --rounds 10 --outdir out_chain
    python tools/round_runner.py --profile A --rounds 10 --outdir out_chain --dry-run
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG = {
    "A": os.path.join(ROOT, "data", "official", "cpu-a-guide-20261110-v1"),
    "B": os.path.join(ROOT, "data", "official", "cpu-b-rul-20261110-v1"),
    "C": os.path.join(ROOT, "data", "official", "cpu-c-guide-20261110-v1"),
}
PY = sys.executable


def read_report(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def metric_of(rep, key="validation_after"):
    """从 training.json 取主指标。分类用 macro_f1（越大越好），回归用 rmse（越小越好）。

    返回 (value, higher_better, name)。
    """
    v = rep[key]
    if "macro_f1" in v:
        return float(v["macro_f1"]), True, "macro_f1"
    return float(v["rmse"]), False, "rmse"


def gain_of(before, after, higher_better):
    """统一成"增益"：正数=变好，负数=变差。分类回归共用一套判断。"""
    return (after - before) if higher_better else (before - after)


def run_train(pkg_dir, train, val, parent, out, epochs, lr=None, bs=None,
              stage_id="round", use_resume=True, seed=None):
    """调用官方 train.py。use_resume=True 用 --resume（保留 Adam/epoch），否则 --from-baseline。"""
    flag = "--resume" if use_resume else "--from-baseline"
    cmd = [PY, "-m", "baselines.train",
           "--profile", os.path.basename(pkg_dir).split("-")[1].upper()[:1],
           "--train", train, "--validation", val,
           flag, parent, "--epochs", str(epochs),
           "--output", out, "--stage-id", stage_id]
    if lr is not None:
        cmd += ["--learning-rate", str(lr)]
    if bs is not None:
        cmd += ["--batch-size", str(bs)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=pkg_dir, capture_output=True, text=True)
    dt = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError("train failed: %s\n%s" % (p.stdout[-2000:], p.stderr[-2000:]))
    return read_report(os.path.join(out, "training.json")), dt


def decide(history, actions_tried):
    """规则化诊断：看最近几轮的边际变化，决定下一步。

    返回 (action, reason)。

    规则顺序（先排除，再尝试）：
      1. 上一轮持平（|delta| < 0.01pp）→ 收敛，停
      2. 连续两轮 |delta| < 0.2pp → 收敛，停
      3. 上一轮掉分 → 换一族动作（restart → lr_lo → lr_hi → bs64）
      4. 上一轮有增益但很小（< 0.2pp）→ 换起点
      5. 上一轮有效（≥ 0.2pp）→ 继续同方向，但连续同向超过 2 次就换
    """
    if not history:
        return "continue20", "首轮：从官方 checkpoint 续训，观察是否还有提升空间"

    last = history[-1]
    delta = last.get("gain", 0.0)

    # 关注：分类的增益是 macro_f1 差（0.x 量级），回归的是 rmse 差（几十到几百）。
    # 所以这里的"小"必须用相对量，不能用绝对阈值。
    rel = abs(last.get("rel_gain", 0.0))

    if rel < 1e-4:
        return "stop", "上一轮相对变化 <0.01%%，续训已饱和"
    if len(history) >= 2:
        d1 = history[-1].get("rel_gain", 0.0)
        d2 = history[-2].get("rel_gain", 0.0)
        if abs(d1) < 0.002 and abs(d2) < 0.002:
            return "stop", "连续两轮相对增益 <0.2%%，判定收敛"

    if last["accepted"] is False:
        for act in ("lr_lo", "continue10", "lr_hi", "bs64", "bs256"):
            if act not in actions_tried:
                return act, "上一轮掉分 %.2f%%，换动作 %s" % (abs(rel) * 100, act)
        return "stop", "动作空间耗尽"

    # 同方向连续 2 次成功就不再继续（续训必然会遇到饱和点）
    same_dir = sum(1 for h in history[-2:] if h["action"] == last["action"] and h["accepted"])
    if rel < 0.002 or same_dir >= 2:
        for act in ("lr_lo", "continue10", "lr_hi", "bs64", "bs256"):
            if act not in actions_tried:
                return act, "本轮增益 %+.2f%% 偏小，换动作 %s" % (rel * 100, act)
        return "stop", "动作空间耗尽"
    return "continue20", "上一轮有效（%+.2fpp），继续同方向" % (delta * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="A", choices=["A", "B", "C"])
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--outdir", default="out_chain")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pkg = PKG[args.profile]
    outdir = os.path.join(ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    train_f = os.path.join(pkg, "public", "train.jsonl")
    val_f = os.path.join(pkg, "public", "validation.jsonl")
    best_ckpt = os.path.join(pkg, "checkpoint.npz")

    # 第 0 轮：官方 checkpoint 的基线分（不训练，只读官方 training.json）
    with open(os.path.join(pkg, "training.json"), encoding="utf-8") as f:
        base = json.load(f)
    base_val, higher_better, metric_name = metric_of(base)
    arrow = "↑越大越好" if higher_better else "↓越小越好"
    print("第 0 轮（官方 checkpoint）%s = %.4f  (%s)" % (metric_name, base_val, arrow))
    print()

    history = []
    actions_tried = set()
    best_f1 = base_val
    best_round = 0
    rounds_log = []

    for r in range(1, args.rounds + 1):
        action, reason = decide(history, actions_tried)
        actions_tried.add(action)
        print("[轮 %d] 动作=%s | %s" % (r, action, reason))
        if action == "stop":
            print("      ⇒ 停止，不进入下一轮")
            rounds_log.append({"round": r, "action": "stop", "reason": reason})
            break

        rounds_log.append({"round": r, "action": action, "reason": reason})
        if args.dry_run:
            print("      (dry-run，跳过实际训练)")
            continue

        out = os.path.join(outdir, "r%02d" % r)
        if os.path.isdir(out):
            shutil.rmtree(out)

        try:
            # 全部走 --resume：官方 train.py:74 硬性要求"续训必须保留 checkpoint 的 seed"，
            # 换 seed 等于声明新模型第 1 轮，会切断证据链。所以动作只在 lr / bs / epochs 上做。
            epochs = {"continue10": 10}.get(action, 20)
            lr = {"lr_lo": 0.0015, "lr_hi": 0.006}.get(action)
            bs = {"bs64": 64, "bs256": 256}.get(action)
            rep, dt = run_train(pkg, train_f, val_f, best_ckpt, out,
                                epochs, lr=lr, bs=bs,
                                stage_id="round%d_%s" % (r, action), use_resume=True)
        except RuntimeError as exc:
            print("      !! 训练失败：%s" % exc)
            rounds_log.append({"round": r, "action": action, "error": str(exc)})
            continue

        before, _, _ = metric_of(rep, "validation_before")
        after, _, _ = metric_of(rep, "validation_after")
        gain = gain_of(before, after, higher_better)
        accepted = gain > 1e-9
        print("      before %.4f → after %.4f  (增益 %+.4f)  耗时 %.1fs  %s"
              % (before, after, gain, dt, "接受" if accepted else "回退"))

        rel = (gain / abs(before)) if abs(before) > 1e-12 else 0.0
        history.append({"round": r, "action": action, "before": before,
                        "after": after, "gain": gain, "rel_gain": rel,
                        "accepted": accepted})

        if accepted:
            best_ckpt = os.path.join(out, "checkpoint.npz")
            if gain_of(base_val, after, higher_better) > gain_of(base_val, best_f1, higher_better):
                best_f1 = after
                best_round = r

        with open(os.path.join(outdir, "state.json"), "w", encoding="utf-8") as f:
            json.dump({"metric": metric_name, "higher_better": higher_better,
                       "best": best_f1, "best_round": best_round,
                       "history": history, "rounds": rounds_log}, f,
                      ensure_ascii=False, indent=2)

    print()
    print("=" * 62)
    total_gain = gain_of(base_val, best_f1, higher_better)
    print("最佳 %s = %.4f（第 %d 轮）" % (metric_name, best_f1, best_round))
    print("相对官方基线 %.4f：增益 %+.4f（%s）"
          % (base_val, total_gain, arrow))
    print("总轮数 %d，接受 %d 轮，回退 %d 轮"
          % (len(history), sum(1 for h in history if h["accepted"]),
             sum(1 for h in history if not h["accepted"])))
    with open(os.path.join(outdir, "state.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": metric_name, "higher_better": higher_better,
                   "best": best_f1, "best_round": best_round, "baseline": base_val,
                   "total_gain": total_gain,
                   "history": history, "rounds": rounds_log}, f,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
