"""归因：漂移报警之后，回答"是哪一种漂移、由哪些特征引起"。

只用两条已经算出来的统计量做判断（不需要额外算力，也不需要标签）：

  feature_psi 高、pred_psi 低  → 协变量漂移（输入分布变了，但模型排序能力没变）
  feature_psi 中低、pred_psi 高 → 概念漂移（输入没怎么变，但模型判断整体偏移）
  两者都高                     → 混合漂移

这个判断直接决定动作选择：协变量漂移优先做"重新标准化/校准"，概念漂移才值得"重训"。
"""
from __future__ import annotations

import numpy as np


def attribute(monitor_result: dict, feature_names: list[str] | None = None, top_k: int = 5) -> dict:
    fps = np.asarray(monitor_result["feature_psi"], dtype=float)
    feat_max = float(monitor_result["feature_psi_max"])
    pred_psi = float(monitor_result["pred_psi"])

    order = np.argsort(-fps)[:top_k]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(fps))]
    top = [{"feature": feature_names[i], "psi": round(float(fps[i]), 5)} for i in order]

    # 分类判据：看**两条证据线的比值**，而不是各自过没过阈值。
    #
    # 为什么改（2026-11-07 实测）：旧判据"pred_psi>=0.10 且 feat>=0.10 就算 mixed"，
    # 在渐变协变量段会把纯协变量误判成 mixed——输入侧 PSI 0.75 时，冻结模型
    # 的输出分布自然也会漂到 0.15，这并不是"判定标准变了"，只是"输入变了"的传导。
    # 实测：B 段 w7/w8 就这样被判成 mixed（真值是 covariate）。
    #
    # 比值能分开两者：
    #   纯协变量，输出漂移只是输入的传导   → pred/feat 约 0.2–0.4
    #   概念漂移，输出漂移超过输入漂移     → pred/feat > 1.5
    ratio = pred_psi / max(feat_max, 1e-9)
    if feat_max < 0.10 and pred_psi < 0.10:
        drift_type = "none"             # 两条线都没过警告线
    elif pred_psi >= 0.25 and ratio >= 1.5:
        drift_type = "concept"          # 输出侧漂得比输入侧凶 → 边界变了
    elif ratio >= 1.0 and feat_max >= 0.10 and pred_psi >= 0.10:
        drift_type = "mixed"            # 两侧都在动，且输出侧不弱于输入侧
    elif feat_max >= 0.10:
        drift_type = "covariate"        # 输入变了但输出基本跟着走 → 规律还在
    else:
        drift_type = "covariate"

    # 建议动作：这是"证据 → 动作"的映射，policy 可以直接用，也可以覆盖
    if drift_type == "concept":
        suggestion = "retrain"
        why = "预测分数分布显著偏移而输入变化不大，判断边界大概率失效"
    elif drift_type == "mixed":
        suggestion = "calibrate_then_retrain"
        why = "输入与预测同时漂移，先校准降低阈值失配，再看是否仍需重训"
    elif drift_type == "covariate":
        suggestion = "calibrate"
        why = "输入分布漂移但预测分布基本稳定，模型排序能力仍在，校准/重标准化即可"
    else:
        suggestion = "none"
        why = "两条证据线都未越过警告线"

    return {
        "drift_type": drift_type,
        "suggestion": suggestion,
        "why": why,
        "confidence": round(min(1.0, max(feat_max, pred_psi) / 0.4), 3),
        "top_features": top,
        "feature_psi_max": round(feat_max, 5),
        "pred_psi": round(pred_psi, 5),
    }
