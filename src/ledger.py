"""成本账本：每次动作记一条，能直接算出"单位增益成本"。

赛题评分是比值——分子是性能增益，分母是算力/标注/Token。
所以账本不是"记个流水"，它是客观分的分母，必须能回答：
    这一次动作，花了多少，赚了多少，值不值。

用法：
    led = Ledger("experiments/ledger.jsonl")
    led.log("retrain", reason="drift_level=2", cpu_s=1.2, delta_metric=0.031,
            metric_before=0.71, metric_after=0.74)
    print(led.summary())   # 累计成本 / 总增益 / 每 1% 增益成本
"""
from __future__ import annotations

import json
import os
import time


class Ledger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.records: list[dict] = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self.records.append(json.loads(line))

    def log(self, action: str, reason: str = "", cpu_s: float = 0.0, gpu_s: float = 0.0,
            tokens: int = 0, labels: int = 0, metric_before: float | None = None,
            metric_after: float | None = None, extra: dict | None = None) -> dict:
        delta = None
        if metric_before is not None and metric_after is not None:
            delta = round(metric_after - metric_before, 6)
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "action": action,
            "reason": reason,
            "cost": {"cpu_s": round(cpu_s, 4), "gpu_s": round(gpu_s, 4),
                     "tokens": int(tokens), "labels": int(labels)},
            "metric_before": metric_before,
            "metric_after": metric_after,
            "delta_metric": delta,
        }
        if extra:
            rec["extra"] = extra
        self.records.append(rec)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    # ---------- 汇总 ----------
    def total_cost(self) -> dict:
        tot = {"cpu_s": 0.0, "gpu_s": 0.0, "tokens": 0, "labels": 0}
        for r in self.records:
            for k in tot:
                tot[k] += r["cost"][k]
        return tot

    def total_gain(self, baseline_f1: float | None = None, run_f1: float | None = None) -> float:
        """本次运行的净增益。

        有两种口径，必须分清：

        1. **有基线可比时**（传了 baseline_f1 / run_f1）：增益 = 本方案 overall F1 − 同一条流上
           冻结基线的 overall F1。这才是门槛③要自证的那个数，也是"单位增益成本"的分子。
        2. **没有基线时**：退化为按动作累加 delta_metric。这只是日志摘要，**不能用于客观分**——
           因为回滚/校准事件写的不是 F1（早先版本把"告警等级 2→0"当成 F1 增量求和，
           得到 total_gain = −1.3 的荒唐结果，就是这里出的错）。

        第二轮验收指出过：拿它当客观分分母，分母会变成负数。
        """
        if baseline_f1 is not None and run_f1 is not None:
            return round(float(run_f1) - float(baseline_f1), 6)
        return round(sum(r["delta_metric"] for r in self.records
                         if r.get("delta_metric") is not None), 6)

    def cost_per_gain(self, baseline_f1: float | None = None, run_f1: float | None = None) -> dict:
        """每 1% 增益的成本（分母按三项成本分别给，不用加权成单一数字，避免拍权重）。"""
        gain = self.total_gain(baseline_f1, run_f1)
        tot = self.total_cost()
        if gain is None or gain <= 0:
            return {"gain": gain, "note": "增益 <= 0，单位增益成本无定义（这正是要避免的情形）"}
        return {
            "gain": gain,
            "gain_pct": round(gain * 100, 4),
            "cpu_s_per_1pct": round(tot["cpu_s"] / (gain * 100), 4),
            "tokens_per_1pct": round(tot["tokens"] / (gain * 100), 4),
            "labels_per_1pct": round(tot["labels"] / (gain * 100), 4),
        }

    def summary(self, baseline_f1: float | None = None, run_f1: float | None = None) -> dict:
        by_action: dict[str, int] = {}
        for r in self.records:
            by_action[r["action"]] = by_action.get(r["action"], 0) + 1
        out = {
            "n_actions": len(self.records),
            "by_action": by_action,
            "total_cost": self.total_cost(),
            "total_gain": self.total_gain(baseline_f1, run_f1),
            "cost_per_gain": self.cost_per_gain(baseline_f1, run_f1),
        }
        if baseline_f1 is not None and run_f1 is not None:
            out["gain_basis"] = {
                "run_overall_f1": round(float(run_f1), 6),
                "baseline_overall_f1": round(float(baseline_f1), 6),
                "formula": "run_f1 - baseline_f1（同一条流、同一阈值口径）",
            }
        else:
            out["gain_basis"] = {"formula": "sum(delta_metric)：仅日志摘要，不可用于客观分"}
        return out

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.summary(), fh, ensure_ascii=False, indent=2)
