"""决策器：输入"当前状态"，输出"做什么、为什么"。

同一个位置上有两个候选人，必须能互换而不改调用方：
  - RulePolicy   ：一张 if-else 规则表（默认）。可审计、可复现、零成本。
  - LLMPolicy    ：每窗把证据交给 Qwen3.8-27B，让它选动作。
  - HybridPolicy ：规则先判；只在"规则想不动、但确实报警了"这类边界上问 LLM。

评测硬要求：**LLM 必须可降级**。断网、超时、返回格式错、配额耗尽，任何一种
都必须立刻退回规则，不能卡住窗口，更不能让整条链路失败——否则评测机上调不通
就等于交了零分。所以 HybridPolicy 带熔断器（连续失败 2 次即永久降级）。

证据里刻意不放的东西：
  - segment（A/B/C/D 段名）：那是答案，等于告诉模型"现在是第几段"；
  - y_true / 当窗标签：时序上还没到（标签延迟一窗）。
放了就是标签泄漏。

策略表（config/policy.yaml 可覆盖）：
    level 0                     -> none
    level 1                     -> calibrate（有冷却，避免频繁动作）
    level 2 + 有延迟标签 AUC      -> AUC 掉超 5pp 则 retrain，否则 calibrate（--third-signal auc）
    level 2 + 协变量漂移         -> retrain（实测：警报级时只校准救不回来）
    level 2 + 概念/混合漂移      -> retrain（判断边界变了，校准救不回来）
    预算剩余 < 15%              -> 强制 none（安全闸，优先级最高）
    更新次数 >= 上限             -> 强制 none（防无限制自我迭代）
"""
from __future__ import annotations

import json

from src.llm_client import LLMClient, LLMError

DEFAULT_CFG = {
    "budget_floor": 0.15,        # 预算剩余低于这个比例就停手
    "calib_cooldown": 2,         # 两次校准之间至少隔几个窗口
    "update_cooldown": 2,        # 两次重训之间至少隔几个窗口
    "max_updates": 6,            # 单次运行最多重训几次（硬上限）
    "level2_needs_both": True,   # level 2 是否要求双证据（monitor 已保证）
    "auc_drop": 0.05,            # --third-signal auc 时，AUC 掉多少算"排序坏了"
}

ACTIONS = ("none", "calibrate", "retrain", "rollback")


class Policy:
    """规则决策器（原有实现，保持接口不变，老实验的可复现性靠它）。"""

    def __init__(self, cfg: dict | None = None):
        self.cfg = dict(DEFAULT_CFG)
        if cfg:
            self.cfg.update(cfg)

    def decide(self, level: int, drift_type: str, budget_used_ratio: float,
               windows_since_update: int, windows_since_calib: int, n_updates: int,
               auc_last: float | None = None, auc_ref: float | None = None) -> dict:
        c = self.cfg
        remaining = 1.0 - budget_used_ratio

        # 安全闸优先：无论漂移多严重，预算见底就只观察
        if remaining < c["budget_floor"]:
            return {"action": "none", "reason": f"预算剩余 {remaining:.0%} 低于安全线 {c['budget_floor']:.0%}，强制只观察"}
        if n_updates >= c["max_updates"]:
            return {"action": "none", "reason": f"重训次数已达硬上限 {c['max_updates']}，停止自我迭代"}

        if level == 0:
            return {"action": "none", "reason": "双证据线均未报警"}

        if level == 1:
            if windows_since_calib < c["calib_cooldown"]:
                return {"action": "none", "reason": f"处于校准冷却期（距上次 {windows_since_calib} 窗）"}
            return {"action": "calibrate", "reason": "单侧告警，先做零成本校准，不动模型"}

        # level == 2
        if windows_since_update < c["update_cooldown"]:
            return {"action": "none", "reason": f"处于更新冷却期（距上次 {windows_since_update} 窗）"}

        # 第三条信号（--third-signal auc）：拿上一窗标签算出的 AUC 判断"排序有没有坏"。
        #
        # 为什么它比 PSI 系列可靠：PSI 是分布距离，输入一变它就高，但它分不清
        # "输入变了"和"规律变了"。AUC 只看排序对不对，规律没变时它不动。
        #
        # 为什么用"上一窗"：标签延迟一窗，本窗算不出来。这不是妥协，是时序事实。
        #
        # FD002 实测（tools/probe_third_signal.py, 209 窗）：
        #   排序没坏时，当窗分位数校准 +3.45pp；每窗重训 -5.50pp。
        #   两者差 9 个点——动作选错的代价比"多花算力"大得多。
        if auc_last is not None and auc_ref is not None:
            drop = float(auc_ref) - float(auc_last)
            if drop > c["auc_drop"]:
                return {"action": "retrain",
                        "reason": f"延迟标签 AUC 掉 {drop*100:.1f}pp，排序能力已坏，须重训"}
            if windows_since_calib < c["calib_cooldown"]:
                return {"action": "none",
                        "reason": f"AUC 未掉（{drop*100:+.1f}pp）但处于校准冷却期（距上次 {windows_since_calib} 窗）"}
            return {"action": "calibrate",
                    "reason": f"延迟标签 AUC 稳（{drop*100:+.1f}pp），排序没坏，只校准不重训"}

        if drift_type == "covariate":
            # 实测（2026-11-07）：渐变协变量到警报级时，只调阈值救不回来。
            # B 段 w8 同一窗：校准 F1 0.818、重训 0.968（PSI 已到 0.75）。
            # 原因：输入尺度整体偏移时，阈值只能平移一点，而模型对新尺度下的
            # 排序也需要重新估计。level==1（单侧轻警）仍走校准，见上面分支。
            return {"action": "retrain",
                    "reason": "协变量漂移已达警报级：输入尺度变了，仅校准不够"}

        # concept / mixed：判断边界变了，校准救不回来
        return {"action": "retrain", "reason": f"{drift_type} 漂移：决策边界可能失效，需重训并走评估门"}


# ================= 决策接口：evidence → decision =================
#
# main.py 只依赖 `decide_evidence(ev) -> dict`，换实现不动主流程。
# 返回体固定含 action / reason / source 三个键，source ∈ {rule, llm, hybrid}。
# 这三个键会进 ledger 和 run_summary，答辩时能逐窗追溯"这步是谁拍的板"。

def build_evidence(*, window: int, level: int, drift_type: str,
                   psi_feat: float, psi_pred: float,
                   pos_rate_pred: float, pos_rate_ref: float,
                   recent_f1: float | None,
                   budget_remaining: float, n_updates: int, max_updates: int,
                   windows_since_update: int, windows_since_calib: int,
                   prev_action: str, ece_last_window: float | None = None,
                   auc_last_window: float | None = None,
                   auc_ref: float | None = None) -> dict:
    """把一窗的所有无标签信号压成一份"证据"。字段名要稳定：LLM prompt 和缓存键都靠它。"""
    ev = {
        "window": int(window),
        "level": int(level),
        "drift_type": str(drift_type),
        "psi_feature": round(float(psi_feat), 4),
        "psi_pred": round(float(psi_pred), 4),
        "pos_rate_now": round(float(pos_rate_pred), 4),
        "pos_rate_ref": round(float(pos_rate_ref), 4),
        "f1_last_window": None if recent_f1 is None else round(float(recent_f1), 4),
        # 上一窗的校准误差：纯概念漂移下 PSI 与正例率都不动，只有它会亮。
        # 需要标签，所以只能"延迟一窗"反映，不能当实时信号用。
        "ece_last_window": None if ece_last_window is None else round(float(ece_last_window), 4),
        "budget_remaining": round(float(budget_remaining), 3),
        "updates_used": int(n_updates),
        "updates_cap": int(max_updates),
        "windows_since_update": int(windows_since_update),
        "windows_since_calib": int(windows_since_calib),
        "prev_action": str(prev_action),
    }
    # 第三条信号只在 --third-signal auc 时出现。
    # 刻意做成"条件加键"而不是恒定两个 null：默认关时 evidence 逐字节不变，
    # 之前所有基于 evidence 的实验（尤其 LLM 那组）不需要重跑。
    if auc_last_window is not None and auc_ref is not None:
        ev["auc_last_window"] = round(float(auc_last_window), 4)
        ev["auc_ref"] = round(float(auc_ref), 4)
    return ev


def _blocked(action: str, ev: dict, cfg: dict) -> str | None:
    """安全闸：LLM 的建议也必须过这一关。返回拦截理由，None=放行。

    为什么单独抽出来：规则和 LLM 走同一套闸，才能说"换的是决策者，不是安全边界"。
    rollback 永远放行——退回更稳的版本在任何情况下都不该被预算拦。
    """
    if action == "none" or action == "rollback":
        return None
    if ev["budget_remaining"] < cfg["budget_floor"]:
        return f"预算剩余 {ev['budget_remaining']:.0%} 低于安全线 {cfg['budget_floor']:.0%}"
    if action == "retrain":
        if ev["updates_used"] >= cfg["max_updates"]:
            return f"重训已达上限 {cfg['max_updates']} 次"
        if ev["windows_since_update"] < cfg["update_cooldown"]:
            return f"重训冷却中（{ev['windows_since_update']} < {cfg['update_cooldown']}）"
    if action == "calibrate" and ev["windows_since_calib"] < cfg["calib_cooldown"]:
        return f"校准冷却中（{ev['windows_since_calib']} < {cfg['calib_cooldown']}）"
    return None


class RulePolicy:
    def __init__(self, cfg: dict | None = None):
        self.cfg = dict(DEFAULT_CFG)
        if cfg:
            self.cfg.update(cfg)
        self.rule = Policy(self.cfg)

    def decide_evidence(self, ev: dict) -> dict:
        d = self.rule.decide(
            level=ev["level"],
            drift_type=ev["drift_type"],
            budget_used_ratio=1.0 - ev["budget_remaining"],
            windows_since_update=ev["windows_since_update"],
            windows_since_calib=ev["windows_since_calib"],
            n_updates=ev["updates_used"],
            auc_last=ev.get("auc_last_window"),
            auc_ref=ev.get("auc_ref"),
        )
        d["source"] = "rule"
        return d

    def stats(self) -> dict:
        return {}


_SYSTEM_PROMPT = (
    "你是工业质检系统的自适应决策器。数据按窗口到达，质量会漂移。\n"
    "每窗你会收到一组**无标签**监测信号，请决定本窗动作，只输出一个 JSON：\n"
    '{"action":"none|calibrate|retrain|rollback","reason":"20字以内中文理由"}\n'
    "含义：\n"
    "  none      = 不动（分布正常，或证据不足）\n"
    "  calibrate = 只调判定阈值，不碰模型权重（便宜）\n"
    "  retrain   = 用最近几个窗口重新训练（只有当判断边界可能变了才用）\n"
    "  rollback  = 退回更稳的旧版本（当前版本被漂移带偏时）\n"
    "原则：\n"
    "  1) level=0 一律 none；\n"
    "  2) 只有 psi_feature 与 psi_pred 同时偏高（都 >= 0.10）才考虑 retrain；\n"
    "  3) 只有 psi_feature 高、psi_pred 不高，说明输入变了但规律可能没变，优先 calibrate；\n"
    "  4) budget_remaining < 0.2 时优先 none 或 calibrate；\n"
    "  5) 不确定时选 none —— 乱动的代价高于不动。\n"
    "只输出 JSON，不要任何解释文字。"
)


class LLMPolicy:
    """纯 LLM 决策：每窗都把证据交给模型（L 组，用于对照实验）。

    注意它**不是**"让 LLM 训练模型"，只是让它选那 4 个动作之一。
    """

    def __init__(self, client: LLMClient, cfg: dict | None = None):
        self.client = client
        self.cfg = dict(DEFAULT_CFG)
        if cfg:
            self.cfg.update(cfg)
        self.rule = Policy(self.cfg)
        self.fails = 0
        self.fallbacks = 0

    def _ask(self, ev: dict) -> tuple[str, str]:
        user = json.dumps({k: v for k, v in ev.items() if k != "window"}, ensure_ascii=False)
        text = self.client.chat(
            [{"role": "system", "content": _SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            temperature=0.0, max_tokens=256,
        )
        i, j = text.find("{"), text.rfind("}")
        if i < 0 or j <= i:
            raise LLMError(f"没有 JSON: {text[:120]!r}")
        obj = json.loads(text[i:j + 1])
        action = str(obj.get("action", "")).strip().lower()
        if action not in ACTIONS:
            raise LLMError(f"非法 action: {action!r}")
        return action, str(obj.get("reason", ""))[:80]

    def decide_evidence(self, ev: dict) -> dict:
        try:
            action, reason = self._ask(ev)
        except (LLMError, json.JSONDecodeError) as e:
            self.fails += 1
            self.fallbacks += 1
            d = self.rule.decide(
                level=ev["level"], drift_type=ev["drift_type"],
                budget_used_ratio=1.0 - ev["budget_remaining"],
                windows_since_update=ev["windows_since_update"],
                windows_since_calib=ev["windows_since_calib"],
                n_updates=ev["updates_used"],
                auc_last=ev.get("auc_last_window"),
                auc_ref=ev.get("auc_ref"),
            )
            d.update({"source": "rule_fallback", "llm_error": str(e)[:120]})
            return d

        blocked = _blocked(action, ev, self.cfg)
        if blocked:
            # LLM 想做动作但安全闸不许：记下来，动作降级为 none
            return {"action": "none", "source": "llm_blocked",
                    "reason": f"LLM 建议 {action}，被安全闸拦下：{blocked}"}
        return {"action": action, "source": "llm", "reason": f"[llm] {reason}"}

    def stats(self) -> dict:
        return dict(self.client.stats(), llm_failed_calls=self.fails,
                    llm_fallbacks=self.fallbacks)


class HybridPolicy:
    """规则打底 + LLM 补边界。设计目标是"token 花在最值钱的地方"。

    只在三种情况问 LLM：
      1) 规则判 none，但 level >= 1（规则想不动，可信号确实响了）；
      2) 规则判 none 是因为冷却期（不是因为没有漂移）；
      3) level == 2 且 drift_type 是 mixed/unknown（规则自己也拿不准）。
    其余情况直接用规则——省下来的 token 就是评分表上"单位增益消耗"的分母。
    """

    def __init__(self, client: LLMClient, cfg: dict | None = None, max_fails: int = 2):
        self.client = client
        self.cfg = dict(DEFAULT_CFG)
        if cfg:
            self.cfg.update(cfg)
        self.rule_policy = RulePolicy(self.cfg)
        self.llm = LLMPolicy(client, self.cfg)
        self.max_fails = max_fails
        self.circuit_open = False
        self.cache: dict[str, tuple[str, str]] = {}

    def _cache_key(self, ev: dict) -> str:
        # 刻意不含 window：状态相同就复用上一次的判断，既省 token 又提高可复现性。
        # auc_last_window 必须进键：它是 --third-signal auc 下唯一决定"校准还是重训"的字段，
        # 漏掉它会让两个 AUC 不同的窗误用同一份缓存（第 5 轮验收指出的真 bug）。
        # 用 ev.get 而不是 ev[...]：默认关时这两个键根本不存在。
        keys = ("level", "drift_type", "psi_feature", "psi_pred",
                "budget_remaining", "updates_used", "windows_since_update",
                "auc_last_window")
        return json.dumps({k: ev.get(k) for k in keys}, sort_keys=True)

    def _needs_llm(self, base: dict, ev: dict) -> bool:
        if base["action"] != "none":
            return False                      # 规则已经决定动手，不必再问
        if ev["level"] == 0:
            return False                      # 完全没信号，问也是浪费
        if ev["drift_type"] in ("mixed", "unknown"):
            return True
        return ev["level"] >= 1               # 报警了但规则选不动（冷却/预算）

    def decide_evidence(self, ev: dict) -> dict:
        base = self.rule_policy.decide_evidence(ev)
        if self.circuit_open or not self._needs_llm(base, ev):
            return base

        key = self._cache_key(ev)
        if key in self.cache:
            action, reason = self.cache[key]
            blocked = _blocked(action, ev, self.cfg)
            if blocked:
                return base
            return {"action": action, "source": "hybrid_cached",
                    "reason": f"[llm缓存] {reason}"}

        try:
            action, reason = self.llm._ask(ev)
            self.llm.fails = 0
        except (LLMError, json.JSONDecodeError) as e:
            self.llm.fails += 1
            self.llm.fallbacks += 1
            if self.llm.fails >= self.max_fails:
                self.circuit_open = True      # 熔断：剩下全走规则，不再重试拖慢窗口
            return {**base, "source": "hybrid_fallback", "llm_error": str(e)[:120]}

        self.cache[key] = (action, reason)
        blocked = _blocked(action, ev, self.cfg)
        if blocked:
            return {**base, "source": "hybrid_blocked",
                    "reason": f"{base['reason']}（LLM 建议 {action} 被闸拦：{blocked}）"}
        return {"action": action, "source": "hybrid", "reason": f"[llm] {reason}"}

    def stats(self) -> dict:
        s = dict(self.client.stats())
        s.update({"llm_failed_calls": self.llm.fails, "llm_fallbacks": self.llm.fallbacks,
                  "llm_circuit_open": self.circuit_open, "llm_cache_size": len(self.cache)})
        return s


def make_policy(mode: str, cfg: dict | None = None, *, base_url: str = "",
                model: str = "qwen3.8-27b", api_key: str | None = None,
                timeout: float = 20.0, no_think: bool = True, warn=None):
    """工厂。任何"配置不全/建不起来"的情况都退回 RulePolicy，并在日志里说清楚。

    这不是容错偷懒，是评测场景的硬要求：镜像交上去之后我们不在场，
    唯一能保证的是"没有 LLM 也能跑完"。
    """
    mode = (mode or "rule").lower()
    if mode == "rule":
        return RulePolicy(cfg)
    if not base_url:
        if warn:
            warn(f"[policy] mode={mode} 但没给 --llm-base-url，退回规则决策")
        return RulePolicy(cfg)
    client = LLMClient(base_url, model, api_key=api_key, timeout=timeout, no_think=no_think)
    if mode == "llm":
        return LLMPolicy(client, cfg)
    if mode in ("hybrid", "auto"):
        return HybridPolicy(client, cfg)
    if warn:
        warn(f"[policy] 未知 mode={mode}，退回规则决策")
    return RulePolicy(cfg)
