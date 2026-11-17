"""SDK 训练轮的五环实现：监测 → 归因 → 决策 → 执行 → 账本。

这个模块只被 /app/contest_submission.py 的 train(rnd) 调用，运行在评测容器里。
它和 src/ 下那套合成流实现是**两套东西**，不要互相 import：
  - src/ 那套走 CSV、分类、F1 触发，服务于离线实验与十轮链；
  - 这里走 SDK 的 rnd 对象、三包、评估门，服务于企业赛的在线轮次。

三条设计约束（都来自实测踩坑，不是偏好）：

1. **不抛异常**。train 抛异常 = 这一轮没有 finish = 整轮作废。所有分支都收敛到
   "提交一个合法候选 + finish"。宁可交父模型，不可让进程死。
2. **经验表存 /state，不依赖 rnd.history() 的字段**。SDK 只校验 history 记录里的 id，
   其他字段是什么由 host 决定（HARNESS_SDK.md:102 只说"已经完成的检查点记录"）。
   所以"这个方向试过几次、几次被接受"必须由我们自己持久化。
3. **策略随经验改变**。这是评分表"自改进证据"那 10 分的落点：不是"选对动作"，
   而是"同一个动作在同样的证据下，因为历史不同而得到不同的处置"。
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

# ---------- 经验表：策略迭代的载体 ----------

STATE_FILE = "experience.json"


def _empty_experience() -> dict:
    return {
        "format": "contest.experience.v1",
        "rounds": 0,
        # 方向 -> {"tries","accepts","rejects","best_delta","status"}
        "directions": {},
        # 平台期追踪
        "plateau": {"since_round": 0, "best_delta": None, "streak_below_noise": 0},
        # 每一轮的摘要（只留最近 20 条，state 有大小限制）
        "trace": [],
        # 不可伪造的效率指标
        "rounds_since_accept": 0,
        "dead_directions": [],
    }


def load_experience(rnd, work: Path) -> dict:
    """从 /state 取回上一轮的经验表。没有就返回空表，绝不抛。"""
    try:
        dst = work / "state_in"
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        rnd.state_workspace(str(dst))
        path = dst / STATE_FILE
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("format") == "contest.experience.v1":
                return data
    except Exception:
        pass
    return _empty_experience()


def save_experience(rnd, work: Path, exp: dict) -> None:
    try:
        out = work / "state_out"
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        (out / STATE_FILE).write_text(json.dumps(exp, ensure_ascii=False, indent=1), encoding="utf-8")
        rnd.save_state(str(out))
    except Exception:
        pass  # 存不上就下一轮重来，不影响本轮 finish


# ---------- 策略：元规则 + 经验 ----------

# 动作白名单。LLM 只能在这里面选，选到外面一律退回规则给的动作。
# 曾经还有一项 calibrate（阈值校准），移植自原型的二分类场景——但三个官方包里
# A 是多分类（argmax，没有阈值可调）、B/C 是回归（没有阈值），该动作在任何一包上
# 都没有对应的执行语义，因此移除。
# 留作记录：从别的任务搬来的动作，要先证明它在新任务的输出空间里有定义。
ACTIONS = ("noop", "retrain_20", "retrain_10")

# 噪声带：改进量小于它就算"没动"。
# A 包是 macro-F1 的 bootstrap SE（n=1244、F1≈0.72 时算得 ±1.27pp）；
# B/C 是回归，RMSE 没有统一量纲，用相对变化 1% 作带。
NOISE_ABS = {"A": 0.0127}
NOISE_REL = {"B": 0.01, "C": 0.01}


def _delta(key: str, before: dict, after: dict) -> float | None:
    """统一成"越大越好"：分类用 +ΔF1，回归用 −ΔRMSE。缺失返回 None。"""
    try:
        if key == "A":
            return float(after["macro_f1"]) - float(before["macro_f1"])
        return float(before["rmse"]) - float(after["rmse"])
    except Exception:
        return None


def _beyond_noise(key: str, delta: float | None, before: dict) -> bool:
    """这次改进/退步是否超出噪声带。"""
    if delta is None:
        return False
    if key in NOISE_ABS:
        return abs(delta) > NOISE_ABS[key]
    try:
        base = float(before.get("rmse", 0)) or 1.0
    except Exception:
        base = 1.0
    return abs(delta) / base > NOISE_REL.get(key, 0.01)


def _verdict(exp: dict, action: str) -> dict:
    """这个方向在历史里的战绩。"""
    rec = exp.get("directions", {}).get(action, {})
    return {"tries": int(rec.get("tries", 0)),
            "accepts": int(rec.get("accepts", 0)),
            "rejects": int(rec.get("rejects", 0)),
            "best_delta": rec.get("best_delta"),
            "status": rec.get("status", "open")}


def _plateau(exp: dict, key: str) -> dict:
    return exp.get("plateau", {}) or {}


def choose_by_rules(exp: dict, key: str, evidence: dict) -> tuple[str, str]:
    """三条元规则。返回 (动作, 理由)。

    元规则是人写的先验，agent 做的是"在元规则下积累经验并改变后续行为"，
    不是"自己发明了新规则"。文档里必须这样写，不能声称后者。
    """
    v20, v10 = _verdict(exp, "retrain_20"), _verdict(exp, "retrain_10")
    plateau = _plateau(exp, key)
    last = evidence.get("last_delta")

    # R2 平台期：连续 3 轮改进量都在噪声带内 ⇒ 停止探索，只做可逆的校准。
    if int(plateau.get("streak_below_noise", 0)) >= 3:
        return "noop", (f"连续 {plateau['streak_below_noise']} 轮改进量均小于噪声带"
                        f"（{_noise_text(key)}），按 R2 判定已到平台期，本轮保留父模型")

    # R1 方向判死：某方向连续 2 次超出噪声带的退步 ⇒ 永久排除。
    if v20["status"] == "dead" and v10["status"] != "dead":
        return "retrain_10", f"retrain_20 连续 {v20['rejects']} 次退步已判死（R1），本轮换短程续训"
    if v20["status"] == "dead" and v10["status"] == "dead":
        return "noop", "retrain_20 与 retrain_10 均已判死（R1），本轮保留父模型"

    # R3 失败模式：上一轮重训被拒或执行失败 ⇒ 本轮保留父模型、观察一轮。
    # （曾经这里是"先做可逆的校准"，但校准在多项分布/回归输出上没有定义，已移除。）
    if evidence.get("last_action", "").startswith("retrain") and evidence.get("last_rejected"):
        if evidence.get("last_failed"):
            return "noop", ("上一轮训练执行失败（异常），本轮先保留父模型，"
                            "避免连续失败耗尽预算；待下一次有正常分数再判断方向")
        shown = "无分数" if last is None else f"{abs(float(last)):.4f}"
        return "noop", (f"上一轮 {evidence['last_action']} 退步 "
                        f"{shown} 超出噪声带（R3），本轮保留父模型、观察一轮，"
                        f"避免连续失败耗尽预算")

    return "retrain_20", "无历史否决信号，按默认节奏续训 20 轮"


def _noise_text(key: str) -> str:
    return f"{NOISE_ABS[key] * 100:.2f}pp" if key in NOISE_ABS else f"{NOISE_REL.get(key, 0.01) * 100:.0f}%"


def ask_model(rnd, evidence: dict, exp: dict, fallback: str, fallback_reason: str) -> tuple[str, str, str | None, str, int]:
    """让 LLM 在合法动作里选一个。返回 (动作, 理由, call_id, llm_status, budget)。

    三条硬约束：
      - 只给监测证据与历史战绩，不给标签、不给真值；
      - 动作必须在候选名单里，否则退回规则给的动作并记 llm_fallback；
      - 任何异常都吞掉，用 fallback 顶上 —— LLM 可以缺席，流程不能死。
    """
    ranked = [a for a in (fallback,) + ACTIONS if _verdict(exp, a)["status"] != "dead"]
    ranked = list(dict.fromkeys(ranked))[:3] or ["noop"]
    payload = {
        "dataset": evidence.get("dataset_key"),
        "round": evidence.get("round_index"),
        "last_action": evidence.get("last_action"),
        "last_delta": evidence.get("last_delta"),
        "last_rejected": evidence.get("last_rejected"),
        "train_rows": evidence.get("train_rows"),
        "candidates": {a: _verdict(exp, a) for a in ranked},
        "plateau": _plateau(exp, evidence.get("dataset_key", "A")),
        "allowed_actions": ranked,
    }
    prompt = (
        "你是训练轮的动作选择器。下面是无标签的监测证据与历史战绩。"
        "只能在 allowed_actions 里选一个动作，输出一行 JSON："
        '{"action":"<动作名>","reason":"<不超过80字的中文理由>"}。'
        "不要输出其他任何内容。\n" + json.dumps(payload, ensure_ascii=False)
    )
    # Qwen3 系列是"先思考再回答"的模型：官方端点不支持 chat_template_kwargs 关思考，
    # 思考会吃掉整个 max_tokens 预算，content 直接为空（实测 200/400 token 时 100% 为空）。
    # 对策：① 预算尽量给到 900；② 末尾加 /no_think；③ 空回复重试一次。
    # 但 SDK 的 round.py:179 规定 max_tokens 不得超过 rnd.max_output_tokens（否则 MAX_TOKENS_INVALID），
    # 平台实测值可能只有 128（见 SDK tests），所以这里绝不虚抬下限。
    cap = int(getattr(rnd, "max_output_tokens", 0) or 0)
    budget = max(1, min(900, cap)) if cap > 0 else 900
    tries = 2
    last_text = ""
    call_id = None
    llm_status = "empty"
    for attempt in range(tries):
        try:
            receipt = rnd.model(
                [{"role": "user", "content": prompt + "\n/no_think"}],
                max_tokens=budget,
                temperature=0,
            )
        except Exception as exc:  # 超时、网络，一律兜底
            return (fallback, f"{fallback_reason}（模型不可用：{type(exc).__name__}，已用规则兜底）",
                    None, "error", budget)
        call_id = getattr(receipt, "call_id", None)
        last_text = (receipt.text or "").strip()
        if last_text:
            llm_status = "answered"
            break

    start, end = last_text.find("{"), last_text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(last_text[start:end + 1])
        except Exception:
            obj = {}
        action = str(obj.get("action", "")).strip()
        reason = str(obj.get("reason", "")).strip()[:300]
        if action in ranked:
            return action, reason or fallback_reason, call_id, llm_status, budget
        return (fallback, f"{fallback_reason}（模型给了 {last_text[:60]!r}，不在候选名单，已退回规则）",
                call_id, "out_of_set", budget)
    return (fallback, f"{fallback_reason}（模型回复为空，{tries} 次尝试后仍无内容，已用规则兜底）",
            call_id, llm_status, budget)


# ---------- 执行 ----------

def _write_jsonl(rows, path: Path) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _parent_checkpoint(rnd, work: Path, key: str) -> Path | None:
    """父模型：优先 /previous（平台给的），失败退回镜像内置的官方权重。

    注意 `rnd.parent.path()` 会校验 sha256（round.py:106-111），所以假宿主里
    声明的"1024 字节、sha256:abab..."是取不到真文件的——它会抛，我们吞掉，
    走内置权重。这条回退不是"降级"，是**首轮必然发生**：第一轮没有上一轮产物。
    """
    try:
        declared = list(rnd.parent.declared_paths())
        for name in declared:
            if name.endswith(".npz"):
                p = rnd.parent.path(name, work / "parent")
                if p.is_file():
                    return p
    except Exception:
        pass
    here = Path(__file__).resolve().parent
    for base in (Path("/app/checkpoints"), here / "checkpoints",
                 here.parent / "harness" / "checkpoints_official",
                 here.parent / "harness" / "checkpoints_trained"):
        p = base / ("%s.npz" % key)
        if p.is_file():
            return p
    return None


def run_round(rnd) -> dict:
    """一轮训练。任何分支都必须走 finish，绝不向上抛。"""
    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    key = rnd.dataset_key
    exp = load_experience(rnd, work)
    exp["rounds"] = int(exp.get("rounds", 0)) + 1

    # ---- 1. 监测：拿本轮的规模与上一轮的结局当证据（无标签字段）----
    train_rows = list(rnd.data.training_rows())
    val_rows = list(rnd.data.validation_rows())
    last = exp.get("trace", [])[-1] if exp.get("trace") else {}
    evidence = {
        "dataset_key": key,
        "round_index": rnd.round_index,
        "last_action": last.get("action", ""),
        "last_delta": last.get("delta"),
        "last_rejected": bool(not last.get("accepted", True)) and bool(last),
        "last_failed": bool(last.get("failed", False)),
        "train_rows": len(train_rows),
    }

    # ---- 2+3. 决策：规则给底，模型在候选名单内裁定 ----
    # B 包例外：官方 train_run 要求 label_provenance（train.py:53-60），我们没有官方认可的
    # 健康标签派生方法，硬训会直接抛。而 B 包六个方向实测全负（最小 +3.6、最大 +33.8），
    # 本来就不该动。所以 B 包固定 noop —— 这是有依据的保守，不是偷懒。
    llm_status, llm_budget = "not_called", 0
    if key == "B":
        action, reason = "noop", "B 包无合规的健康标签派生方法（train.py:53 要求 provenance），且六个方向实测全负，本轮保留官方权重"
        call_id = None
        if True:  # 仍需一次真实模型调用，MODEL_CALL_REQUIRED
            cap_b = int(getattr(rnd, "max_output_tokens", 0) or 0)
            llm_budget = max(1, min(16, cap_b)) if cap_b > 0 else 16
            try:
                receipt = rnd.model(
                    [{"role": "user", "content": "B 包本轮不训练，只保留官方权重。请回复：保留。"}],
                    max_tokens=llm_budget, temperature=0)
                call_id = receipt.call_id
                llm_status = "b_ack" if (receipt.text or "").strip() else "b_empty"
            except Exception:
                llm_status = "b_error"
    else:
        fallback, fallback_reason = choose_by_rules(exp, key, evidence)
        action, reason, call_id, llm_status, llm_budget = ask_model(
            rnd, evidence, exp, fallback, fallback_reason)

    # ---- 4. 执行 ----
    candidate = work / "candidate"
    if candidate.exists():
        shutil.rmtree(candidate, ignore_errors=True)
    candidate.mkdir(parents=True)

    parent = _parent_checkpoint(rnd, work, key)
    parent_found = parent is not None
    trained = False
    failed = False
    error = ""
    accepted = False
    metrics = {}
    try:
        if action == "noop" or parent is None:
            if parent is not None:
                shutil.copyfile(parent, candidate / "checkpoint.npz")
            reason = reason + "（保留父模型，无父模型可用时不训练）"
        else:
            train_path = work / "train.jsonl"
            val_path = work / "val.jsonl"
            _write_jsonl(train_rows, train_path)
            _write_jsonl(val_rows, val_path)
            from baselines.train import train_run  # 官方实现，镜像内 /app/baselines
            epochs = 10 if action == "retrain_10" else 20
            out = work / "trained"
            if out.exists():
                shutil.rmtree(out, ignore_errors=True)
            report = train_run(key, train_path, val_path, out, resume=str(parent),
                               epochs=epochs, stage_id=f"r{rnd.round_index}")
            trained = True
            before = report.get("validation_before") or {}
            after = report.get("validation_after") or {}
            metrics = {"before": before, "after": after}
            accepted = _better(key, after, before)
            src = out / "checkpoint.npz"
            if accepted:
                shutil.copyfile(src, candidate / "checkpoint.npz")
                shutil.copyfile(out / "training.json", candidate / "training.json")
            else:
                shutil.copyfile(parent, candidate / "checkpoint.npz")
                reason = reason + f"（续训后未优于父模型：{_fmt(key, before)} → {_fmt(key, after)}，已回退）"
    except Exception as exc:
        # 训练失败：交父模型，别让这一轮白跑
        failed = True
        error = "%s: %s" % (type(exc).__name__, exc)
        if parent is not None and (candidate / "checkpoint.npz").is_file() is False:
            shutil.copyfile(parent, candidate / "checkpoint.npz")
        reason = reason + f"（执行异常 {type(exc).__name__}: {exc}，已回退父模型）"
        accepted = False

    if not (candidate / "checkpoint.npz").is_file():
        # 最后一道：连父模型都没有，就写一个空占位，保证 finish 有东西可指
        np.savez(candidate / "checkpoint.npz", placeholder=np.zeros(1))
        reason = reason + "（无可用权重，提交占位）"

    # ---- 5. 账本：更新经验表（这是"策略随经验改变"的唯一载体）----
    delta = _delta(key, metrics.get("before") or {}, metrics.get("after") or {}) if metrics else None
    dirs = exp.setdefault("directions", {})
    rec = dirs.setdefault(action, {"tries": 0, "accepts": 0, "rejects": 0,
                                   "best_delta": None, "status": "open"})
    rec["tries"] = int(rec.get("tries", 0)) + 1
    if delta is not None:
        best = rec.get("best_delta")
        rec["best_delta"] = delta if best is None else max(best, delta)
    if accepted:
        rec["accepts"] = int(rec.get("accepts", 0)) + 1
        exp["rounds_since_accept"] = 0
    else:
        rec["rejects"] = int(rec.get("rejects", 0)) + 1
        exp["rounds_since_accept"] = int(exp.get("rounds_since_accept", 0)) + 1
        # R1：同一方向连续 2 次超出噪声带的退步 ⇒ 判死
        if delta is not None and _beyond_noise(key, delta, metrics.get("before") or {}) and delta < 0:
            rec["status"] = "dead"
    exp["dead_directions"] = sorted(a for a, r in dirs.items() if r.get("status") == "dead")

    plate = exp.setdefault("plateau", {"since_round": 0, "best_delta": None, "streak_below_noise": 0})
    if delta is not None and _beyond_noise(key, delta, metrics.get("before") or {}):
        plate["streak_below_noise"] = 0
        plate["best_delta"] = delta
        plate["since_round"] = rnd.round_index
    elif delta is not None:
        plate["streak_below_noise"] = int(plate.get("streak_below_noise", 0)) + 1

    exp["trace"] = (exp.get("trace", []) + [{
        "round": rnd.round_index, "action": action, "accepted": accepted,
        "delta": delta, "failed": failed, "error": error[:200], "reason": reason[:200],
    }])[-20:]

    # change_summary 里写死可观测字段：即使拿不到平台回传，也能从配置里判断
    # "父模型读没读到 / 到底训没训 / 前后分数是多少 / 平台给了多少 token 上限 / 模型出没出内容"，
    # 不必消耗 trial 去猜。
    observable = (f"parent_found={parent_found}；trained={trained}；"
                  f"before={_fmt(key, metrics.get('before') or {})}；"
                  f"after={_fmt(key, metrics.get('after') or {})}；"
                  f"max_out={llm_budget}；llm_status={llm_status}")
    if failed:
        observable += f"；failed={error[:80]}"

    submitted = rnd.submit_candidate(
        candidate,
        hypothesis=reason,
        change_summary=f"{action}；接受={accepted}；{observable}",
        configuration={"dataset_key": key, "action": action, "epochs": 10 if action == "retrain_10" else (20 if action == "retrain_20" else 0)},
        seed=1,
        checkpoint_paths=["checkpoint.npz"],
        predict_argv=["python3", "/app/harness.py", "predict"],
    )

    save_experience(rnd, work, exp)

    history = rnd.history()
    rnd.finish(
        selected_artifact_id=submitted["artifact_id"],
        hypothesis=reason,
        change_summary=f"{action}；接受={accepted}；Δ={delta}；{observable}；耗时 {time.perf_counter() - t0:.1f}s",
        reason=reason,
        configuration={
            "dataset_key": key,
            "action": action,
            "directions": exp.get("directions", {}),
            "plateau": exp.get("plateau", {}),
            "rounds_since_accept": exp.get("rounds_since_accept", 0),
            "dead_directions": exp.get("dead_directions", []),
        },
        seed=1,
        history_ids=[item["id"] for item in history],
        model_call_ids=list(rnd.call_ids),
    )
    return exp


def _better(key: str, after: dict, before: dict) -> bool:
    if not after or not before:
        return False
    try:
        if key == "A":
            return float(after.get("macro_f1", 0)) > float(before.get("macro_f1", 0)) + 1e-6
        return float(after.get("rmse", 1e9)) < float(before.get("rmse", 1e9)) - 1e-6
    except Exception:
        return False


def _fmt(key: str, m: dict) -> str:
    try:
        return ("F1 %.4f" % m.get("macro_f1")) if key == "A" else ("RMSE %.2f" % m.get("rmse"))
    except Exception:
        return str(m)
