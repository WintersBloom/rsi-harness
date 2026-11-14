"""自进化闭环主程序：监测 → 归因 → 决策 → 执行 → 评估上线/回滚。

一条命令跑完，全程无人工干预：
    python src/main.py --data data/synth_stream.csv --outdir results

设计原则（对应 .team/plan.md §1）：
  - 决策链只用无标签信号（特征 PSI + 预测分布 PSI），标签只用于训练与事后评估；
  - 每个动作都记成本（ledger），能算"每 1% 增益花了多少"；
  - 重训必须过"评估门"：新模型当窗表现不优于旧模型就不采纳（等效回滚）；
  - 预算/次数硬闸永远优先于"想不想更新"。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.actions import ActionRunner
from src.attribute import attribute
from src.io_adapter import load_stream
from src.ledger import Ledger
from src.metrics import evaluate, expected_calibration_error
from src.monitor import Monitor
from src.policy import build_evidence, make_policy
from src.predictor import NumpyLR
from src.registry import Registry


def fresh_workdir(workdir: str) -> list[str]:
    """清掉上一次运行的残留（registry / ledger / checkpoint / predictions）。

    为什么必须清：registry 会复用 v1、ledger 会追加、checkpoint 会接着旧窗口，
    三者叠加会让"同一个命令跑两次"得到不同结果。要么做完整断点续训，要么
    保证幂等——这里选后者，因为幂等才是评测场景真正需要的（重跑必得同分）。
    """
    removed = []
    if not os.path.isdir(workdir):
        return removed
    for name in ("ledger.jsonl", "checkpoint.json", "predictions.csv"):
        p = os.path.join(workdir, name)
        if os.path.exists(p):
            os.remove(p)
            removed.append(name)
    reg = os.path.join(workdir, "registry")
    if os.path.isdir(reg):
        shutil.rmtree(reg)
        removed.append("registry/")
    return removed


def run(args) -> dict:
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.workdir, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)
        print(msg)

    if args.resume:
        log("[resume] 保留上次状态（未实现断点续训，等价于从零重跑）")
    else:
        wiped = fresh_workdir(args.workdir)
        if wiped:
            log(f"[fresh] 清理上次残留: {', '.join(wiped)}")

    stream = load_stream(args.data)
    train = stream.slice(0, args.train_n)
    model = NumpyLR(lr=args.lr, epochs=args.epochs, l2=args.l2, seed=args.seed,
                    poly=args.poly)
    model.fit(train.X, train.y)

    # 初始阈值。0.5 是"概率过半"的数学惯例，但这个模型带类别权重（样本不平衡时
    # 人为抬高正例权重），输出概率不是校准过的概率——0.5 切下去预测正例率远高于真实
    # 的 8%。quantile 口径直接取训练集打分分布上对应真实正例率的分位点，只改起点、
    # 不改"什么时候动"。
    if args.init_threshold == "quantile":
        _rate = float(train.y.mean())
        threshold = float(np.quantile(model.predict_proba(train.X), 1.0 - _rate))
    else:
        threshold = 0.5
    registry = Registry(os.path.join(args.workdir, "registry"))
    v0 = registry.register(model, threshold, {"note": "initial model, trained on first window"})
    registry.activate(v0)
    # 基线守护哨兵：保存一份永不更新的副本。漂移过去之后，模型很可能被"上一段概念"带偏，
    # 这时能把它拉回来的只有基线。这是门槛③（不低于静态基线）的保险丝。
    baseline_model = NumpyLR.load(os.path.join(args.workdir, "registry", f"v{v0}.npz"))

    mon = Monitor(train.X, model.predict_proba(train.X))
    # --max-updates 用来放开"自我迭代次数"硬上限：算天花板（每窗都全量重训）时需要
    # 无限次更新，否则测出来的不是天花板而是"只能改 6 次的方案"。
    # 决策者：默认 rule（与全部历史实验一致）；llm/hybrid 需要显式传 --llm-base-url。
    # 传了模型但连不上时会自动退回 rule 并打日志——评测机上调不通，系统也要能跑完。
    policy_cfg = {}
    if args.max_updates:
        policy_cfg["max_updates"] = args.max_updates
    if args.calib_cooldown is not None:
        policy_cfg["calib_cooldown"] = args.calib_cooldown
    policy_cfg = policy_cfg or None
    policy = make_policy(
        args.policy, policy_cfg,
        base_url=args.llm_base_url, model=args.llm_model,
        api_key=os.environ.get(args.llm_api_key_env) if args.llm_api_key_env else None,
        timeout=args.llm_timeout, no_think=not args.llm_think, warn=log,
    )
    if args.policy != "rule":
        log(f"[policy] mode={args.policy} model={args.llm_model} url={args.llm_base_url or '(未配置)'}")
    ledger = Ledger(os.path.join(args.workdir, "ledger.jsonl"))
    runner = ActionRunner(model, threshold, ref_pos_rate=float(train.y.mean()))

    # 第三条信号的参考点：初始模型在训练集上的 AUC。
    # 用它当"排序本来有多好"的基线，后面每窗比"掉没掉"。
    ref_auc = float(evaluate(train.y, model.predict_proba(train.X), 0.5)["auc"])
    if args.third_signal == "auc":
        log(f"[third-signal] auc 参考值 {ref_auc:.4f}（初始模型在训练集上），掉超 {policy.cfg['auc_drop']*100:.0f}pp 判排序已坏")

    labeled_X, labeled_y = [train.X], [train.y]
    y_all_init = train.y.copy()      # 参考正例率（无标签守护模式也要用）
    from collections import deque
    recent = deque(maxlen=args.train_window)   # 最近 N 个窗口的数据，用于"滑动窗口重训"
    recent.append((train.X, train.y))
    n_updates = 0
    n_calib = 0
    last_update_at = -99
    last_calib_at = -99
    used_cpu = 0.0
    zero_streak = 0          # 分布已恢复的连续窗口数
    last_level = 0           # 上一窗的告警等级（edge 守护用）
    last_action = "none"     # 上一窗实际执行的动作（喂给决策器，避免连续同动作）
    n_f1_triggers = 0        # F1 兜底触发器触发的次数（只在 --f1-trigger > 0 时可能非零）

    pred_rows = []
    base_rows = []          # 冻结基线在同一批样本上的打分，用于算"真实增益"的分母
    window_records = []
    eval_stream = stream.slice(args.train_n, len(stream))

    for idx, (start, win) in enumerate(eval_stream.windows(args.window)):
        # 严格"标签延迟一窗"：当窗标签在当窗内不可用，所以 recent 只在窗口处理完之后才追加。
        # 早期版本在循环开头就 append，导致重训用到了当窗标签、又用新模型重算当窗 F1 ——
        # 那是标签泄漏，增益是假的。
        # ---- 1) 监测（无标签，且只用参考模型的打分） ----
        # 顺序很关键：监测信号不依赖当窗标签，也不依赖当前模型的打分（pred_psi 用冻结基线算），
        # 所以它可以**在预测之前**给出结论。这让"警报刚解除"能做到零延迟——本窗就换回基线预测，
        # 而不是等窗末发现已经预测错了 500 个样本。
        p_ref = baseline_model.predict_proba(win.X)
        mr = mon.check(win.X, p_ref, cur_pred_ref=p_ref)
        prev_level, last_level = last_level, mr["level"]

        # 1.5) edge 守护提前执行：上一窗还在等级 2，这一窗回落到 0，说明漂移期结束。
        # 此刻还在用漂移期训出来的模型，先换回冻结基线再预测。
        # `not args.frozen` 是必须的：baseline_model 是从磁盘独立 load 出来的另一个对象，
        # 所以 `model is not baseline_model` 在冻结组里也恒真，会凭空产生一次 edge 回滚
        # （第三轮验收在 A_frozen/seed0/run.log 里抓到过）。
        if (args.guard_mode == "edge" and not args.frozen and prev_level >= 2 and mr["level"] == 0
                and model is not baseline_model):
            model = baseline_model
            threshold = 0.5
            runner.model = model
            runner.threshold = threshold
            registry.activate(v0)
            log(f"[w{idx:03d}] EDGE rollback to baseline v{v0}: 警报 {prev_level}->0，漂移期结束")
            ledger.log("edge_rollback", f"警报解除 {prev_level}->0，本窗即回滚基线（零延迟）",
                       extra={"window": idx, "level_before": prev_level, "level_after": 0})

        # ---- 1.8) 当窗预测（此时模型可能已被 edge 换成基线） ----
        p = model.predict_proba(win.X)
        y_true = win.y
        real = evaluate(y_true, p, threshold)

        # ---- 2) 归因（仅报警时） ----
        # 分布已恢复的连续窗口数（recover 守护用，纯监测信号、不碰标签）
        if mr["level"] == 0:
            zero_streak += 1
        else:
            zero_streak = 0

        attr = None
        if mr["level"] > 0:
            attr = attribute(mr)

        # ---- 2.5) 当窗标签还没到（窗末复判中），本窗一律不可用 ----
        # 时序约定（第二轮验收后收紧）：标签在**本窗所有工作结束后**才进入缓冲。
        # 早先版本在这里 append，导致下面评估门 recent[-1] 拿到的就是当窗——
        # predictions.csv 没被污染，但"接受哪个模型"用了当窗答案，是有系统性乐观偏差的。
        # 现在的语义是：标签到达后，最快也要到下一窗才能参与训练/评估。

        # ---- 3) 决策（规则 / LLM / 混合，由 --policy 决定） ----
        # 把本窗所有**无标签**信号压成一份 evidence，再交给当前决策者。
        # 注意 evidence 里没有 segment、没有当窗标签——那两样是答案，放进去就是泄漏。
        budget_ratio = used_cpu / args.budget_cpu_s if args.budget_cpu_s > 0 else 0.0
        pos_now = float((p >= threshold).mean()) if len(p) else 0.0
        ev = build_evidence(
            window=idx,
            level=mr["level"],
            drift_type=(attr or {}).get("drift_type", "none"),
            psi_feat=mr["feature_psi_max"],
            psi_pred=mr["pred_psi"],
            pos_rate_pred=pos_now,
            pos_rate_ref=float(y_all_init.mean()),
            recent_f1=(window_records[-1]["f1"] if window_records else None),
            ece_last_window=(window_records[-1].get("ece") if window_records else None),
            # 第三条信号：上一窗的 AUC（标签延迟一窗，所以只能用上一窗的）。
            # 只在 --third-signal auc 时传，默认 None —— evidence 结构不因此变化。
            auc_last_window=((window_records[-1].get("auc") if window_records else None)
                             if args.third_signal == "auc" else None),
            auc_ref=(ref_auc if args.third_signal == "auc" else None),
            budget_remaining=1.0 - budget_ratio,
            n_updates=n_updates,
            max_updates=policy.cfg["max_updates"],
            windows_since_update=idx - last_update_at,
            windows_since_calib=idx - last_calib_at,
            prev_action=last_action,
        )
        decision = policy.decide_evidence(ev)

        if args.frozen:
            decision = {"action": "none", "reason": "frozen: 静态基线，永不更新"}
        elif args.retrain_every and idx > 0 and idx % args.retrain_every == 0 and len(recent) >= 1:
            # 对照组：不看监测信号，按固定周期无脑重训。
            # 工业界最常见的朴素做法，也是"我们的触发机制到底值不值"的唯一诚实对照——
            # 如果不看漂移、每 N 窗就重训也能拿到同样的分，那我们的监测器就是装饰品。
            decision = {"action": "retrain",
                        "reason": f"periodic: 每 {args.retrain_every} 窗强制重训（不看监测信号）"}

        # ---- 3.5) 兜底触发：PSI 看不见的那一类漂移 ----
        # 为什么需要它：level 是 PSI 型监测器给的，而 PSI 只看 P(X)。纯概念漂移
        # （P(X) 不变、只有 P(Y|X) 变）下 PSI 全程为 0，决策第一步就返回 none，
        # 后面所有守护机制根本没机会执行。唯一能看见它的是"延迟标签到达后 F1 下滑"。
        # 所以这里用历史窗的 F1 做第二条触发线：连续 2 窗比历史均值低超过阈值才动，
        # 一窗不动（单窗抖动太常见）。默认关闭，因为正常流上它会误开、打乱已调好的动作序列。
        # --f1-trigger-quiet：只在监测器沉默（当窗 level==0）时才允许兜底触发。
        # 理由：level>=1 说明 PSI 已经看见变化，policy 自己会决定，F1 兜底再去插一脚
        # 只会占掉校准冷却——实测 seed1 就是因为在 C 段（level2）多触发一次，把 D 段
        # 那次真正有效的校准挤掉了，直接掉 1.81pp。沉默期才是这条兜底线的正确战场。
        quiet_ok = (not args.f1_trigger_quiet) or (mr["level"] == 0)
        if (args.f1_trigger > 0 and decision["action"] == "none" and quiet_ok
                and len(window_records) >= 6 and idx - last_calib_at >= 2):
            # 变量名不能叫 recent：外层 recent 是 deque（滑动窗口数据），覆盖它会让
            # 后面的重训/守护在解包里炸掉（float 不可解包）。这个坑在第一次跑就踩到了。
            hist_f1 = [r["f1"] for r in window_records[:-2]]
            tail_f1 = [r["f1"] for r in window_records[-2:]]
            mean_hist = float(np.mean(hist_f1)) if hist_f1 else 0.0
            mean_tail = float(np.mean(tail_f1)) if tail_f1 else 0.0
            if hist_f1 and (mean_hist - mean_tail) > args.f1_trigger:
                decision = {
                    "action": args.f1_action,
                    "reason": (f"f1-trigger: 最近2窗均值 {mean_tail:.3f} "
                               f"低于历史 {mean_hist:.3f} 超 {args.f1_trigger:.2f}")
                }
                n_f1_triggers += 1
                ledger.log("f1_trigger", decision["reason"], extra={"window": idx})

        # ---- 4) 执行 + 记账 ----
        metric_before = real["f1"]
        action_info = {"action": decision["action"], "reason": decision["reason"]}
        if decision["action"] == "calibrate":
            r = runner.calibrate(win.X)
            threshold = runner.threshold
            used_cpu += r["cpu_s"]
            n_calib += 1
            last_calib_at = idx
            action_info.update(r)
            real = evaluate(y_true, p, threshold)   # 阈值变了，当窗指标随之重算
            ledger.log("calibrate", decision["reason"], cpu_s=r["cpu_s"],
                       metric_before=metric_before, metric_after=real["f1"],
                       extra={"threshold": threshold})

        elif decision["action"] == "retrain":
            # 滑动窗口重训：只用最近 N 个窗口的数据。
            # 用全部历史重训是概念漂移下最常见的错误——旧概念的样本量远大于新概念，会把新规律盖住。
            X_all = np.vstack([x for x, _ in recent])
            y_all = np.concatenate([y for _, y in recent])
            sw = None
            if args.weight_mode == "recency" and len(recent) > 1:
                # 按新旧加权：窗口越新权重越大（0.5 → 1.0）。
                # 概念漂移下，最近几个窗口才是"当前任务"，更早的在拖后腿。
                k = len(recent)
                wts = [0.5 + 0.5 * (i / (k - 1)) for i in range(k)]
                sw = np.concatenate([np.full(len(yy), w) for (_, yy), w in zip(recent, wts)])
            anchor = (model.w, model.b) if args.anchor > 0 else None
            scaler = ((baseline_model.scaler.mean, baseline_model.scaler.std)
                      if args.anchor > 0 else None)
            r = runner.retrain(X_all, y_all, lr=args.lr, epochs=args.epochs, l2=args.l2,
                               seed=args.seed, sample_weight=sw,
                               anchor=anchor, anchor_strength=args.anchor, scaler=scaler)
            new_model = r["model"]
            used_cpu += r["cpu_s"]
            # 评估门：只用"标签已经到齐、且不在当窗"的历史窗来比，绝不碰当窗标签。
            gx_eval, gy_eval = recent[-1] if recent else (train.X, train.y)
            old_f1 = evaluate(gy_eval, model.predict_proba(gx_eval), threshold)["f1"]
            new_f1 = evaluate(gy_eval, new_model.predict_proba(gx_eval), 0.5)["f1"]
            accepted = new_f1 >= old_f1 - args.tolerance
            action_info.update({"cpu_s": r["cpu_s"], "old_f1": round(old_f1, 4),
                                "new_f1": round(new_f1, 4), "accepted": accepted})
            if accepted:
                n_updates += 1
                last_update_at = idx
                model = new_model
                # 换模型会改变输出分布，阈值必须重配——但"重配"不等于"退回 0.5"。
                # 之前这里写死 threshold = 0.5，等于把校准过的分数线扔掉；D 段重训后失准
                # 有一部分来自这里。--calib-after-retrain 打开时用当窗分位数重定一次。
                if args.calib_after_retrain:
                    runner.model = model
                    runner.threshold = threshold
                    cr = runner.calibrate(gx_eval)
                    threshold = cr["threshold_after"]
                    last_calib_at = idx
                    log(f"[w{idx:03d}] post-retrain calibrate: thr {cr['threshold_before']:.3f} -> {threshold:.3f}")
                else:
                    threshold = 0.5
                runner.model = model
                runner.threshold = threshold
                ver = registry.register(model, threshold, {"note": f"retrained at window {idx}"})
                registry.activate(ver)
                labeled_X.append(win.X)
                labeled_y.append(win.y)
                log(f"[w{idx:03d}] retrain accepted v{ver}: F1 {old_f1:.4f} -> {new_f1:.4f}")
            else:
                log(f"[w{idx:03d}] retrain REJECTED (rollback): F1 {old_f1:.4f} -> {new_f1:.4f}")
            ledger.log("retrain", decision["reason"], cpu_s=r["cpu_s"],
                       metric_before=old_f1, metric_after=new_f1 if accepted else old_f1,
                       extra={"accepted": accepted, "window": idx})

        else:
            ledger.log("none", decision["reason"], metric_before=metric_before, metric_after=metric_before)

        for t, pv, yy, pr in zip(win.t, p, y_true, (p >= threshold).astype(int)):
            pred_rows.append([int(t), int(yy), round(float(pv), 6), int(pr), idx])

        # 冻结基线对同一批样本的打分。用它在汇总时算 baseline overall F1，
        # 这样"本方案 − 基线"才是门槛③要证的净增益，而不是把各动作的 delta 硬加起来。
        for t, yy, pv in zip(win.t, y_true, p_ref):
            base_rows.append([int(t), int(yy), round(float(pv), 6)])

        window_records.append({
            "window": idx, "start": int(start), "segment": win.segment[0],
            "level": mr["level"], "drift_type": (attr or {}).get("drift_type"),
            "action": decision["action"], "reason": decision["reason"],
            "source": decision.get("source", "rule"),
            "evidence": ev,
            "threshold": round(threshold, 4), "f1": real["f1"], "auc": real["auc"], "ap": real["ap"],
            # 校准误差要等标签，所以在窗末算、供下一窗的思路参考。
            "ece": expected_calibration_error(y_true, p),
            "used_cpu_s": round(used_cpu, 3),
        })
        last_action = decision["action"]

        # ---- 5) 基线守护：漂移过去后，若当前模型还不如最初的静态基线，就回滚 ----
        # 三种模式，对应"标签到底能不能拿到"的三种现实：
        #   f1   : 标签延迟到达（默认，延迟 2 窗）。工业质检里缺陷标签本来就要人工复判，滞后是常态。
        #   dist : 完全无标签，只能看预测分布是否离谱（退化的备选，效果会差，但比没有强）。
        #   off  : 不做守护，等于 C 组。
        if idx >= 2 and len(recent) >= 2 and args.guard_mode != "off":
            gx = np.vstack([x for x, _ in list(recent)[-2:]])
            gy = np.concatenate([y for _, y in list(recent)[-2:]])
            if args.guard_mode == "f1":
                f1_cur = evaluate(gy, model.predict_proba(gx), threshold)["f1"]
                f1_base = evaluate(gy, baseline_model.predict_proba(gx), 0.5)["f1"]
                worse = f1_base > f1_cur + args.guard_margin
                detail = f"F1 {f1_cur:.4f} -> {f1_base:.4f}"
                before, after = f1_cur, f1_base
            elif args.guard_mode == "edge":
                # edge 已经在窗首执行过（见 1.5 步），这里不再重复判断
                worse = False
                detail = "（edge 已在窗首处理）"
                before, after = 0.0, 0.0
            elif args.guard_mode == "recover":
                # 纯监测信号，零标签：分布已经连续 2 窗回到参考水平，说明漂移结束，
                # 而当前模型是在漂移期训的，此刻基线（在参考分布上训练）反而更可信。
                worse = zero_streak >= 2 and model is not baseline_model
                detail = f"分布已恢复 {zero_streak} 窗，回退到基线"
                before, after = float(zero_streak), 0.0
            else:  # dist：只看预测正例率偏离参考正例率多少，完全不碰标签
                ref_rate = float(y_all_init.mean())
                def _dev(m, thr):
                    pr = float((m.predict_proba(gx) >= thr).mean())
                    return abs(pr - ref_rate) / max(ref_rate, 1e-6)
                d_cur = _dev(model, threshold)
                d_base = _dev(baseline_model, 0.5)
                worse = d_base + args.guard_margin < d_cur
                detail = f"正例率偏离 {d_cur:.3f} -> {d_base:.3f}"
                before, after = d_cur, d_base
            if worse:
                model = baseline_model
                threshold = 0.5
                runner.model = model
                runner.threshold = threshold
                registry.activate(v0)
                log(f"[w{idx:03d}] GUARD rollback to baseline v{v0}: {detail}")
                ledger.log("guard_rollback", f"守护[{args.guard_mode}]检测到当前模型劣于静态基线，回滚",
                           extra={"window": idx, "mode": args.guard_mode,
                                  "detail": detail, "metric_before": before, "metric_after": after})

        # ---- 5.5) 窗末：本窗标签到达，进入滑动缓冲 ----
        # 放在这里（而不是决策前）是第二轮验收的核心修正：
        # 只有把 append 推到"本窗该做的一切都做完"之后，上面评估门/守护拿到的
        # recent[-1] 才真的是上一窗，而不是当窗。差这一行，+1.67pp 就是乐观偏差。
        recent.append((win.X, win.y))

        # ---- 6) checkpoint（可断点续跑） ----
        if idx % 4 == 0:
            with open(os.path.join(args.workdir, "checkpoint.json"), "w", encoding="utf-8") as fh:
                json.dump({"window": idx, "n_updates": n_updates, "n_calib": n_calib,
                           "used_cpu": round(used_cpu, 3), "threshold": threshold}, fh, ensure_ascii=False, indent=2)

    # ---------- 汇总 ----------
    all_p = np.array([r[2] for r in pred_rows])
    all_y = np.array([r[1] for r in pred_rows])
    overall = evaluate(all_y, all_p, threshold)

    # 冻结基线在同一条流上的 overall 表现：这是门槛③的比较对象，也是"单位增益成本"的分子。
    base_p = np.array([r[2] for r in base_rows])
    base_y = np.array([r[1] for r in base_rows])
    baseline_overall = evaluate(base_y, base_p, 0.5)
    net_gain = round(float(overall["f1"]) - float(baseline_overall["f1"]), 6)

    # 三口径：评分不允许只看"全程平均"——末窗值看最终状态，最差窗值看抗打击能力
    f1s = [w["f1"] for w in window_records]
    three_views = {
        "mean": round(float(np.mean(f1s)), 4),
        "last": round(float(f1s[-1]), 4),
        "worst": round(float(np.min(f1s)), 4),
        "worst_window": int(np.argmin(f1s)),
    }
    # 分段汇总（用于诊断哪一段被治好了、哪一段被治坏了）
    seg_acc = {}
    for w in window_records:
        seg_acc.setdefault(w["segment"], []).append(w["f1"])
    by_segment = {k: round(float(np.mean(v)), 4) for k, v in seg_acc.items()}

    summary = {
        "config": vars(args),
        "overall": overall,
        "baseline_overall": baseline_overall,
        "net_gain_f1": net_gain,
        "three_views": three_views,
        "by_segment": by_segment,
        "n_windows": len(window_records),
        "n_updates": n_updates,
        "n_calibrations": n_calib,
        "used_cpu_s": round(used_cpu, 3),
        "budget_cpu_s": args.budget_cpu_s,
        "ledger": ledger.summary(baseline_f1=baseline_overall["f1"], run_f1=overall["f1"]),
        "windows": window_records,
    }
    # 决策者的成本账：token 要单列，因为评分表"单位增益消耗"里 token 是分子之一。
    # rule 模式这里是空 dict，不会多出字段，老结果文件格式不变。
    pstats = policy.stats() if hasattr(policy, "stats") else {}
    if pstats:
        summary["policy_stats"] = pstats
    # 兜底触发器的计数只在开的时候写：默认关闭 ⇒ summary 与旧版本逐字节相同。
    if args.f1_trigger > 0:
        summary["n_f1_triggers"] = n_f1_triggers

    with open(os.path.join(args.workdir, "predictions.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t", "y_true", "p", "pred", "window"])
        wr.writerows(pred_rows)

    with open(os.path.join(args.outdir, "run_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    with open(os.path.join(args.outdir, "run.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")

    log(f"\noverall F1={overall['f1']} AUC={overall['auc']} AP={overall['ap']}")
    log(f"updates={n_updates} calibrations={n_calib} cpu_s={used_cpu:.2f} / budget={args.budget_cpu_s}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="智训引擎 · 自进化闭环（无人工干预版）")
    ap.add_argument("--data", default="data/synth_stream.csv")
    ap.add_argument("--outdir", default="results", help="汇总产物目录")
    ap.add_argument("--workdir", default="results/work", help="运行态产物目录（ledger/registry/checkpoint）")
    ap.add_argument("--train-n", type=int, default=1500)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--train-window", type=int, default=3, help="重训时使用的最近窗口数（滑动窗口）")
    ap.add_argument("--budget-cpu-s", type=float, default=60.0, help="本次运行的总算力预算（秒）")
    ap.add_argument("--tolerance", type=float, default=0.005, help="评估门容差：新模型可接受的退步幅度")
    ap.add_argument("--guard-margin", type=float, default=0.02, help="基线守护：基线 F1 领先超过该幅度才回滚")
    ap.add_argument("--max-updates", type=int, default=0,
                    help="重训次数硬上限，0=用默认 6。算天花板时设大值（如 999）")
    # 校准冷却：两次 calibrate 之间至少隔几个窗口。默认 2（Policy.DEFAULT_CFG）。
    # 单独放出来是因为 mock 实验发现"每窗都校准"能到 0.8781，说明这个参数可能被低估了，
    # 需要一个能直接做对照的开关，而不是改源码。
    ap.add_argument("--calib-cooldown", type=int, default=None,
                    help="校准冷却窗数，默认 2；设 0 = 每窗都可校准")
    # 模型容量对照：默认线性。加二阶交叉是"手动加一层"，比上 MLP 便宜得多
    # （13→79 参数，1:19 样本参数比），但合成数据是线性生成的，预期不涨。
    # 留这个开关是为了官方数据到手后能立刻做"线性 vs 交叉 vs 深"的三方对照。
    ap.add_argument("--poly", action="store_true",
                    help="启用二阶特征交叉（12 维 -> 78 维），用于验证非线性边界是否存在")
    ap.add_argument("--f1-trigger", type=float, default=0.0,
                    help="兜底触发器（默认 0=关）：PSI 全程不报警、但最近 2 窗 F1 比历史均值低超过该值时，"
                         "强制做一次校准。用于纯概念漂移——那种漂移 P(X) 不变，PSI 在数学上看不见，"
                         "只有延迟标签到达后的 F1 下滑能发现它。设 0.05 即'掉 5 个百分点'。")
    ap.add_argument("--f1-action", default="calibrate", choices=["calibrate", "retrain"],
                    help="兜底触发器命中后做什么：calibrate=只挪阈值（默认，成本低、不毒化模型）；"
                         "retrain=重训（要过评估门，被拒则等效回滚）")
    ap.add_argument("--f1-trigger-quiet", action="store_true",
                    help="兜底触发只在监测器沉默（level==0）时生效。开启后它不再插进"
                         "level>=1 的决策与冷却期——实测这样才不会挤掉恢复段的关键校准")
    ap.add_argument("--calib-after-retrain", action="store_true",
                    help="重训被接受后立刻按当窗打分分位数重定阈值，而不是退回 0.5。"
                         "换模型会改变输出分布，退回 0.5 等于扔掉之前校准过的分数线")
    ap.add_argument("--init-threshold", default="half", choices=["half", "quantile"],
                    help="初始判定阈值：half=0.5（历史口径）；quantile=训练集打分分布上"
                         "对应真实正例率的分位点。模型带类别权重时 0.5 往往偏离最优")
    ap.add_argument("--retrain-every", type=int, default=0,
                    help="对照组开关：每 N 窗无脑重训一次，完全不看监测信号（0=关）。"
                         "用来回答『不看漂移、定期重训是不是也够』")
    ap.add_argument("--anchor", type=float, default=0.0,
                    help="抗遗忘弹簧强度：>0 时重训以上一版权重为起点并加 L2 拉回（对应评分表『稳定性与抗遗忘』）")
    ap.add_argument("--guard-mode", default="edge", choices=["f1", "dist", "recover", "edge", "off"],
                    help="基线守护：edge=警报解除即回退（默认，零标签零延迟）；f1=延迟标签比 F1；"
                         "recover=分布恢复 N 窗后回退；dist=只看预测正例率；off=不守护")
    ap.add_argument("--weight-mode", default="none", choices=["none", "recency"],
                    help="重训样本加权：recency=越新的窗口权重越大")
    ap.add_argument("--third-signal", default="off", choices=["off", "auc"],
                    help="第三条信号：auc=用上一窗标签算出的 AUC 判断排序有没有坏。"
                         "掉超 5pp 才重训，否则只校准（FD002 实测：只校准 +3.45pp、每窗重训 -5.50pp）。"
                         "默认 off，行为与原版逐字节一致")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frozen", action="store_true", help="冻结模式：只监测不更新，用于跑静态基线对照")
    ap.add_argument("--policy", default="rule", choices=["rule", "llm", "hybrid", "auto"],
                    help="决策者：rule=确定性规则（默认，历史实验都用它）；llm=每窗问大模型；"
                         "hybrid=规则打底、只在边界问 LLM；auto=hybrid（连不上自动熔断回规则）")
    ap.add_argument("--llm-base-url", default="",
                    help="OpenAI 兼容端点，如 http://127.0.0.1:8000/v1；留空则强制 rule")
    ap.add_argument("--llm-model", default="qwen3.8-27b")
    ap.add_argument("--llm-timeout", type=float, default=20.0,
                    help="单次调用超时（秒）。超时即降级回规则，不阻塞窗口")
    ap.add_argument("--llm-api-key-env", default="",
                    help="从这个环境变量读 API Key（不落盘、不进日志、不进 summary）")
    ap.add_argument("--llm-think", action="store_true",
                    help="允许模型先思考再回答（默认关闭：Qwen3 系列开思考单次约 450 token，"
                         "关掉只要 6 token；关掉靠 chat_template_kwargs.enable_thinking=False）")
    ap.add_argument("--resume", action="store_true",
                    help="保留 workdir 里上次的残留（默认会清空，保证重跑幂等）")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
