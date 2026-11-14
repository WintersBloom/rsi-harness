"""把十轮自适应链（out_chain_a/b/c）上传到 SwanLab。

为什么要单独一个脚本，不并进 swanlog.py：
    swanlog.py 传的是"同一份合成流上的十组横评"（A_frozen / D_edge_guard ...），
    那是"换策略比一比"；这里传的是"一条链上逐轮做了什么"，
    那是"换轮次看过程"。评分表要的"三次以上迭代"指的是后者。
    两种东西的横轴不一样（一个是策略名，一个是轮次），混在一个 run 里读不出来。

指标命名约定（与 swanlog.py 保持一致的前缀风格）：
    score/validation     当轮验证集指标（分类是 macro_f1，回归是 rmse）
    score/best_so_far    到目前为止最好的那个
    action/gain          当轮相对上一轮的增益（正=变好，已统一方向）
    action/accepted      当轮是否被接受（1/0）
    chain/baseline       官方基线
    chain/total_gain     整条链拿到的净增益

用法：
    python tools/upload_chain.py --dry-run          # 只打印，不碰 swanlab
    python tools/upload_chain.py --mode offline     # 写到本地 swanlog/，不上云
    python tools/upload_chain.py --mode cloud       # 直接上云（需已 login）

上传完仍要去网页把项目设为 Public，API 改不了可见性。
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 三个包的元信息：目录后缀、展示名、指标名、方向
PACKAGES = [
    ("a", "chain_A_gas", "macro_f1", True, "cpu-a-guide-20261110-v1", "六类气体分类"),
    ("b", "chain_B_bearing", "rmse", False, "cpu-b-rul-20261110-v1", "轴承 RUL 回归"),
    ("c", "chain_C_engine", "rmse", False, "cpu-c-guide-20261110-v1", "发动机 RUL 回归"),
]


def _short_sha(path, n=12):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load_state(tag):
    path = os.path.join(ROOT, "out_chain_%s" % tag, "state.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="zhipin-engine")
    ap.add_argument("--mode", default="offline", choices=["offline", "cloud", "disabled"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run:
        sys.path.insert(0, os.path.join(ROOT, ".pylibs"))

    commit = _git_commit()
    made = []

    for tag, run_name, metric, higher_better, pkg, desc in PACKAGES:
        st = load_state(tag)
        if st is None:
            print("skip %s: 没有 out_chain_%s/state.json（这条链还没跑）" % (tag, tag))
            continue

        hist = st.get("history") or []
        cfg = {
            "task": pkg,
            "desc": desc,
            "metric": metric,
            "higher_better": higher_better,
            "baseline": st.get("baseline"),
            "best": st.get("best"),
            "best_round": st.get("best_round"),
            "total_gain": st.get("total_gain"),
            "n_rounds": len(hist),
            "n_accepted": sum(1 for h in hist if h.get("accepted")),
            "git_commit": commit,
            # 数据指纹：换一版数据这份 run 就不再对应同一件事
            "data_version": "%s@%s" % (
                pkg,
                _short_sha(os.path.join(ROOT, "data", "official", pkg, "public", "train.jsonl")),
            ),
        }

        print("\n=== %s (%s) ===" % (run_name, desc))
        print("  baseline=%.4f best=%.4f best_round=%s total_gain=%+.4f"
              % (cfg["baseline"], cfg["best"], cfg["best_round"], cfg["total_gain"]))
        for h in hist:
            print("  r%02d %-12s %.4f -> %.4f  gain=%+.4f  %s"
                  % (h["round"], h["action"], h["before"], h["after"],
                     h["gain"], "ACCEPT" if h["accepted"] else "revert"))

        if args.dry_run:
            continue

        import swanlab

        run = swanlab.init(project=args.project, experiment_name=run_name,
                           config=cfg, mode=args.mode, reinit=True)
        best_so_far = st.get("baseline")
        for h in hist:
            # 统一成"越高越好"的方向，图上的趋势才与 accepted 的判断一致
            v = h["after"] if higher_better else -h["after"]
            metric_key = "score/validation" if higher_better else "score/validation"
            best_key = "score/best_so_far"
            payload = {
                metric_key: v,
                best_key: best_so_far if higher_better else -best_so_far,
                "action/gain": h["gain"],
                "action/accepted": int(bool(h["accepted"])),
            }
            swanlab.log(payload, step=h["round"])
            if h["accepted"]:
                best_so_far = h["after"]

        # 收尾一条：整条链的结论
        swanlab.log({
            "chain/baseline": st["baseline"] if higher_better else -st["baseline"],
            "chain/best": st["best"] if higher_better else -st["best"],
            "chain/total_gain": st["total_gain"],
            "chain/n_accepted": cfg["n_accepted"],
        }, step=len(hist) + 1)
        made.append(run)

    if args.dry_run:
        print("\n[dry-run] 没有写入任何记录。")
        return

    print("\n完成：%d 个 run（mode=%s）" % (len(made), args.mode))
    if args.mode == "offline":
        print("本地目录在 swanlog/，上传用：")
        print("  python -m swanlab sync swanlog/run-xxx -p %s" % args.project)


if __name__ == "__main__":
    main()
