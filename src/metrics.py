"""指标计算，纯 numpy。故障是稀有类（8%），所以主指标用 PR 相关的 F1 / AP，而不是 accuracy。"""
from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    c = confusion(y_true, y_pred)
    denom = 2 * c["tp"] + c["fp"] + c["fn"]
    return 0.0 if denom == 0 else 2 * c["tp"] / denom


def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    c = confusion(y_true, y_pred)
    p = 0.0 if c["tp"] + c["fp"] == 0 else c["tp"] / (c["tp"] + c["fp"])
    r = 0.0 if c["tp"] + c["fn"] == 0 else c["tp"] / (c["tp"] + c["fn"])
    return p, r


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """用秩和公式算 AUC，O(n log n)，不依赖 sklearn。"""
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 处理并列：取平均秩
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    ranks = avg[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """AP（PR 曲线下面积），稀有类场景比 AUC 更能反映实际可用性。"""
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-score)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / y_true.sum()
    # 取每个正例处的精确率增量积分
    ap = 0.0
    prev_rec = 0.0
    for i in range(len(y)):
        if y[i] == 1:
            ap += prec[i] * (rec[i] - prev_rec)
            prev_rec = rec[i]
    return float(ap)


def expected_calibration_error(y_true: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    """期望校准误差（ECE）：把预测概率分箱，比较"平均预测值"与"实际正例率"。

    为什么单独算它：纯概念漂移下输入分布不变，PSI 与预测正例率都不动，
    唯一会变的是"模型说的 0.9 到底还准不准"。实测（g_pure 冻结基线）
    ECE 从 0.055 涨到 0.17–0.21，同时预测正例率只波动 0.02。
    代价是必须等标签到达，所以只能反映"上一窗"的校准状态。
    """
    y = np.asarray(y_true, dtype=float)
    s = np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
    n = len(s)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1], right=False), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        err += (cnt / n) * abs(float(s[mask].mean()) - float(y[mask].mean()))
    return round(float(err), 4)


def evaluate(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (score >= threshold).astype(int)
    p, r = precision_recall(y_true, y_pred)
    return {
        "n": int(len(y_true)),
        "pos": int(y_true.sum()),
        "f1": round(f1(y_true, y_pred), 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "auc": round(roc_auc(y_true, score), 4),
        "ap": round(average_precision(y_true, score), 4),
    }


# ---------------------------------------------------------------------------
# 回归任务指标（对应 --task reg）
# ---------------------------------------------------------------------------
# 为什么要有这一套：C-MAPSS 这类数据的原生任务是"预测剩余寿命"，输出是连续
# 值而不是概率，F1/AUC/AP 对它没有定义。五环里只有"算分"和"校准动作"两处
# 依赖任务类型，其余（监测/归因/决策/回滚/账本）原样复用，所以这里只补指标。

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return round(float(np.mean(np.abs(y_true - y_pred))), 4)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return round(1.0 - ss_res / ss_tot, 4)


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PHM08 官方评分函数（非对称）。

    为什么不能用 MAE 代替它：这个分数**惩罚晚预测远重于早预测**。
    预测"还剩 60 循环"而真实是"还剩 100"（早预测，白检修）只是浪费；
    预测"还剩 100"而真实是 60（晚预测，机器在空中坏掉）是事故。
    所以 d<0 除以 13、d>0 除以 10，再减 1 求和。

    来源：C-MAPSS 原始论文（PHM08 Data Challenge）第 8 页，
    `d = pred - true`；分数越低越好，0 是完美。

    我们用它的理由：如果官方按这个口径评回归任务，用 MAE 选模型会选错
    ——MAE 对早晚一视同仁，而这个分数不是。
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return round(float(np.sum(s)), 2)


def evaluate_reg(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """回归评估。mae 是主指标——它和 F1 在分类里的地位对应：
    单位与被预测量相同，能直接解释成"平均差了几个循环"。
    phm08 是工业口径的补充：它对晚预测罚得更重，方向和 MAE 不完全一致。"""
    return {
        "n": int(len(y_true)),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "phm08": phm08_score(y_true, y_pred),
        "y_mean": round(float(np.mean(y_true)), 4),
        "pred_mean": round(float(np.mean(y_pred)), 4),
    }
