"""对照组实验：一次跑完全部策略变体，输出可写进报告的对比表。

用法：
    python tools/run_suite.py                # 全部 5 组
    python tools/run_suite.py --seeds 0,1,2  # 多 seed，算配对保守增益
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
MAIN = os.path.join(ROOT, "src", "main.py")
DATA = os.path.join(ROOT, "data", "synth_stream.csv")

# name, extra_args, 说明
RUNS = [
    ("A_frozen",        ["--frozen"],                                            "冻结基线：只监测不更新（等价主办方静态基线）"),
    ("B_full_history",  ["--train-window", "999", "--guard-mode", "off"],        "全历史重训：旧概念压倒新概念"),
    ("C_sliding",       ["--train-window", "3", "--guard-mode", "off"],          "滑动窗口重训：治漂移，但漂移结束后会崩"),
    ("D_edge_guard",    ["--train-window", "3", "--guard-mode", "edge"],         "★本方案：滑动窗口 + 警报解除即回退（零标签、零延迟）"),
    ("E_edge_window2",  ["--train-window", "2", "--guard-mode", "edge"],         "更短窗口 + edge：适应更快，看会不会更抖"),
    ("F_f1_guard",      ["--train-window", "3", "--guard-mode", "f1"],           "延迟标签守护：窗末比对 F1 才回滚（慢一窗）"),
    ("G_recover_guard", ["--train-window", "3", "--guard-mode", "recover"],      "零标签守护：分布连续 2 窗恢复才回退（慢两窗）"),
    ("H_nolabel_dist",  ["--train-window", "3", "--guard-mode", "dist"],         "完全无标签降级：只看预测正例率偏离"),
    # I/J 是"诚实对照"：不看漂移、按固定周期无脑重训。工业界最常见的朴素做法。
    # 如果不看监测信号也能拿同样的分，那我们的触发机制就只是装饰品——必须让人看到这个数。
    ("I_periodic4",     ["--train-window", "3", "--guard-mode", "edge",
                         "--retrain-every", "4", "--max-updates", "99"],
                        "周期重训：每 4 窗无脑重训一次，完全不看监测信号"),
    ("J_always",        ["--train-window", "3", "--guard-mode", "edge",
                         "--retrain-every", "1", "--max-updates", "99"],
                        "总是重训：每窗都重训（算力不设限的上界对照）"),
]


def run_one(name, extra, seed, outroot):
    outdir = os.path.join(outroot, name, f"seed{seed}")
    workdir = os.path.join(outroot, name, f"seed{seed}_work")
    cmd = [PY, "-X", "utf8", MAIN, "--data", DATA, "--outdir", outdir,
           "--workdir", workdir, "--seed", str(seed)] + extra
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        print(f"[FAIL] {name} seed{seed}\n{p.stdout[-800:]}\n{p.stderr[-800:]}")
        return None
    with open(os.path.join(outdir, "run_summary.json"), encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--outroot", default="results/suite")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    outroot = os.path.join(ROOT, args.outroot)
    os.makedirs(outroot, exist_ok=True)

    results = {}
    for name, extra, desc in RUNS:
        for seed in seeds:
            print(f"running {name} seed={seed} ...", flush=True)
            s = run_one(name, extra, seed, outroot)
            if s:
                results.setdefault(name, {})["desc"] = desc
                results[name][f"seed{seed}"] = s

    # ---- 汇总表 ----
    rows = []
    for name, _, desc in RUNS:
        if name not in results:
            continue
        ks = [k for k in results[name] if k.startswith("seed")]
        if not ks:
            continue
        def avg(fn):
            vals = [fn(results[name][k]) for k in ks]
            return sum(vals) / len(vals)
        rows.append({
            "name": name, "desc": desc,
            # 把首 seed 的 config 附在行里：SwanLab 的 config.yaml 需要 train_window /
            # guard_mode 这些字段，只从 name 猜不出来（"E_edge_window2" 能猜，别的不行）。
            "config": results[name][ks[0]].get("config", {}),
            "f1": avg(lambda s: s["overall"]["f1"]),
            "auc": avg(lambda s: s["overall"]["auc"]),
            "ap": avg(lambda s: s["overall"]["ap"]),
            "mean": avg(lambda s: s["three_views"]["mean"]),
            "last": avg(lambda s: s["three_views"]["last"]),
            "worst": avg(lambda s: s["three_views"]["worst"]),
            "updates": avg(lambda s: s["n_updates"]),
            "calib": avg(lambda s: s["n_calibrations"]),
            "cpu": avg(lambda s: s["used_cpu_s"]),
            "segs": {seg: avg(lambda s, seg=seg: s["by_segment"].get(seg, 0.0))
                     for seg in ("A_stable", "B_gradual", "C_concept", "D_recover")},
        })

    hdr = ("| 组 | F1 | AUC | AP | 均值窗 | 末窗 | 最差窗 | 更新 | 校准 | CPU(s) | A段 | B段 | C段 | D段 |")
    sep = "|" + "---|" * 14
    lines = [hdr, sep]
    for r in rows:
        s = r["segs"]
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            r["name"], r["f1"], r["auc"], r["ap"], r["mean"], r["last"], r["worst"],
            int(r["updates"]), int(r["calib"]), r["cpu"],
            s["A_stable"], s["B_gradual"], s["C_concept"], s["D_recover"]))
    table = "\n".join(lines)

    # ---- 配对保守增益（相对 A_frozen）----
    gain_lines = []
    if "A_frozen" in results and len(seeds) > 1:
        for name, _, _ in RUNS:
            if name == "A_frozen" or name not in results:
                continue
            ds = []
            for seed in seeds:
                a = results["A_frozen"].get(f"seed{seed}")
                b = results[name].get(f"seed{seed}")
                if a and b:
                    ds.append(b["overall"]["f1"] - a["overall"]["f1"])
            if ds:
                m = sum(ds) / len(ds)
                var = sum((d - m) ** 2 for d in ds) / max(1, len(ds) - 1)
                sd = var ** 0.5
                gain_lines.append(f"- {name}: mean(d)={m:+.4f}, std={sd:.4f}, "
                                  f"conservative={m - sd:+.4f} -> {'WIN' if m - sd > 0 else 'LOSE'}")
    table += "\n\n### 配对保守增益（相对 A_frozen，多 seed）\n" + ("\n".join(gain_lines) if gain_lines else "- 需要 --seeds 0,1,2 才计算")

    out_md = os.path.join(outroot, "compare.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("# 对照组实验\n\n" + table + "\n")
    with open(os.path.join(outroot, "compare.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)

    print("\n" + table)
    print(f"\nwritten: {out_md}")


if __name__ == "__main__":
    main()
