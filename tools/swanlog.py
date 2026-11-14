"""把本地实验结果批量回放到 SwanLab 云端记录。

## 为什么不在 main.py 里直接调 swanlab

1. **评测镜像是离线的**（`docker run --network none`），不能联网，也不该为了记录多拖
   swanlab 那串依赖（pydantic / rich / protobuf …）进 308MB 的镜像。
2. **赛题要求留痕的对象是"我们做的实验"**，不是"评测时的运行"。原文：
   「全部训练、微调、调整与评估实验须按《实验记录规范》在云端实验记录平台完整记录」。
   所以记录发生在开发期，由本脚本从本地产物回放，产物（ledger.jsonl / run_summary.json）
   才是评测期真正的证据链。

这样职责是干净的：**镜像负责跑，SwanLab 负责证。**

## 用法

    # 先在本机验证（不联网，写到 ./swanlog/，之后可 swanlab sync 上传）
    python tools/swanlog.py --suite results/suite/compare.json --mode offline

    # 直接上云（需要先 swanlab login 或设置 SWANLAB_API_KEY）
    $env:SWANLAB_API_KEY="..."
    python tools/swanlog.py --suite results/suite/compare.json --mode cloud

    # 只看会记录什么，不初始化任何 run
    python tools/swanlog.py --suite results/suite/compare.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import os

# SwanLab 在 Windows 上写 config.yaml 时没指定 encoding，会落成 GBK 字节。
# 中文 desc 读回来就是「鍐荤粨鍩虹嚎」那种乱码，所以上云一律用英文描述。
DESC_EN = {
    "A_frozen": "frozen baseline: monitor only, never update (= organizer static baseline)",
    "B_full_history": "full-history retrain: old concept overwhelms the new one",
    "C_sliding": "sliding-window retrain: fixes drift, collapses after recovery",
    "D_edge_guard": "PROPOSED: sliding window + rollback when the alarm clears (zero-label, zero-delay)",
    "E_edge_window2": "shorter window (2) + edge guard: adapts faster, cheaper",
    "F_f1_guard": "F1 guard: compare delayed labels at window end (one window late)",
    "G_recover_guard": "recover guard: roll back after 2 quiet windows (two windows late)",
    "H_nolabel_dist": "no-label fallback: watch positive-rate deviation only",
}
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".pylibs"))
sys.path.insert(0, ROOT)


def load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _short_sha(path: str) -> str:
    """数据指纹。文件不在就返回 missing，不要让整个回放因为缺一份 csv 就崩。"""
    import hashlib
    if not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_commit() -> str:
    """当前代码版本。不是 git 仓库也要能跑（评测机可能就是拷进去的目录）。"""
    import subprocess
    try:
        p = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8")
        return p.stdout.strip() if p.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def flatten_suite(rec: dict) -> dict:
    """把一组实验结果摊平成 SwanLab 的指标字典。指标名带前缀，方便在面板里分组。"""
    out = {
        "f1/overall": rec["f1"],
        "auc/overall": rec["auc"],
        "ap/overall": rec["ap"],
        "f1/window_mean": rec["mean"],
        "f1/window_last": rec["last"],
        "f1/window_worst": rec["worst"],
        "cost/cpu_seconds": rec["cpu"],
        "count/updates": rec["updates"],
        "count/calibrations": rec["calib"],
    }
    for seg, v in (rec.get("segs") or {}).items():
        out[f"f1/segment/{seg}"] = v
    return out


def window_series(suite_dir: str, name: str, seed: int = 0) -> list:
    """读某一组某个 seed 的逐窗明细，用于把"一个结论"画成"一条曲线"。

    compare.json 里只有汇总量，没有逐窗值——所以这里回到 run_summary.json 取。
    路径不存在就返回空列表（旧批次可能没保留这些文件），不报错。
    """
    p = os.path.join(ROOT, suite_dir, name, f"seed{seed}", "run_summary.json")
    if not os.path.exists(p):
        return []
    return load_json(p).get("windows") or []


def flatten_window(w: dict) -> dict:
    """单窗明细 → 扁平指标。动作用 0/1 表示，方便在图上当竖线看。"""
    action = w.get("action") or "none"
    return {
        "f1/window": w.get("f1"),
        "auc/window": w.get("auc"),
        "ap/window": w.get("ap"),
        "drift/level": w.get("level"),
        "threshold": w.get("threshold"),
        "cost/cpu_window": w.get("used_cpu_s"),
        "action/retrain": 1 if action == "retrain" else 0,
        "action/calibrate": 1 if action == "calibrate" else 0,
        "action/rollback": 1 if "rollback" in action else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="results/suite3/compare.json", help="八组对照结果")
    ap.add_argument("--sensitivity", default="results/sensitivity/summary.json", help="可选，存在才记")
    ap.add_argument("--project", default="zhipin-engine")
    ap.add_argument("--mode", default=os.environ.get("SWANLAB_MODE", "offline"),
                    choices=["offline", "cloud", "disabled", "local"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    suite_path = os.path.join(ROOT, args.suite)
    records = load_json(suite_path)
    if not isinstance(records, list):
        records = records.get("runs") or records.get("records") or []

    print(f"loaded {len(records)} suite runs from {args.suite}")

    if args.dry_run:
        for rec in records:
            print(f"\n[{rec['name']}] {rec.get('desc', '')}")
            for k, v in flatten_suite(rec).items():
                print(f"  {k} = {v:.4f}" if isinstance(v, float) else f"  {k} = {v}")
        return

    import swanlab  # 延迟导入：dry-run 和纯本地流程都不需要它

    for rec in records:
        # run_suite 把首 seed 的 config 附在每行里；旧版 compare.json 没有，就退回用 name 猜。
        c = rec.get("config") or {}
        cfg = {
            "desc": DESC_EN.get(rec["name"], rec["name"]),
            "strategy": rec["name"],
            "train_window": c.get("train_window", rec.get("train_window")),
            "guard_mode": c.get("guard_mode", rec.get("guard_mode")),
            "weight_mode": c.get("weight_mode", rec.get("weight_mode")),
            "train_n": c.get("train_n"),
            "window": c.get("window"),
            "budget_cpu_s": c.get("budget_cpu_s"),
            "lr": c.get("lr"),
            "epochs": c.get("epochs"),
            "l2": c.get("l2"),
            "seed_set": "0,1,2",
            # 数据指纹：换了数据这份 run 就不再对应同一件事（第三轮验收点名要这个字段）
            "data_version": "synth_stream.csv@"
                            + _short_sha(os.path.join(ROOT, "data", "synth_stream.csv")),
            "git_commit": _git_commit(),
        }
        run = swanlab.init(project=args.project, experiment_name=rec["name"],
                           config={k: v for k, v in cfg.items() if v is not None},
                           mode=args.mode)
        swanlab.log(flatten_suite(rec))
        # 再按窗口逐点记一遍：只记汇总等于只给结论不给过程，
        # 评审看不到"漂移来了 F1 怎么掉、回滚之后怎么爬"。
        suite_dir = os.path.dirname(args.suite)
        series = window_series(suite_dir, rec["name"])
        for w in series:
            swanlab.log({k: v for k, v in flatten_window(w).items() if v is not None},
                        step=int(w.get("window", 0)))
        swanlab.finish()
        pts = len(series)
        print(f"  logged {rec['name']} -> {getattr(run, 'id', '')} ({pts} window points)")

    # 敏感性实验单独一个 run，把三档漂移强度当 step
    sens_path = os.path.join(ROOT, args.sensitivity)
    if os.path.exists(sens_path):
        sens = load_json(sens_path)
        # 注意别写成 `sens.get("rows") or sens if isinstance(sens, list) else ...`：
        # Python 会把三元表达式整体当 or 的右操作数，list 上再调 .get 就会 AttributeError。
        if isinstance(sens, list):
            rows = sens
        else:
            rows = sens.get("rows") or sens.get("grid") or []
        if rows:
            swanlab.init(project=args.project, experiment_name="drift_sensitivity",
                         config={"desc": "F1/AP gain under mild / base / severe drift strength",
                                 "drift_strengths": "0.4,1.0,1.5"},
                         mode=args.mode)
            for i, row in enumerate(rows):
                swanlab.log({k: v for k, v in row.items() if isinstance(v, (int, float))},
                            step=i)
            swanlab.finish()
            print(f"  logged drift_sensitivity ({len(rows)} rows)")

    print("\ndone. 若 mode=offline，记录在 ./swanlog/，用 `swanlab sync <dir>` 上传。")


if __name__ == "__main__":
    main()
