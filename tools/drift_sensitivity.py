"""漂移强度敏感性：自进化策略在不同漂移烈度下，到底赢还是输。

结论如果只在"剧烈漂移"下成立，那它就只是个脆弱结论——真实工业数据大多是温和漂移。
用法：python tools/drift_sensitivity.py
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
MAIN = os.path.join(ROOT, "src", "main.py")
SIM = os.path.join(ROOT, "src", "simulate_drift.py")

LEVELS = [("温和", 0.4), ("基准", 1.0), ("剧烈", 1.5)]
GROUPS = [
    ("冻结基线", ["--frozen"]),
    ("本方案", ["--train-window", "3", "--guard-mode", "edge"]),
]


def main():
    rows = []
    for label, strength in LEVELS:
        csv_path = os.path.join(ROOT, "data", f"drift_{strength}.csv")
        subprocess.run([PY, "-X", "utf8", SIM, "--out", csv_path, "--drift-strength", str(strength)],
                       check=True, capture_output=True, text=True)
        line = {"level": label, "strength": strength}
        for gname, extra in GROUPS:
            tag = f"{label}_{'base' if 'frozen' in extra else 'ours'}"
            outdir = os.path.join(ROOT, "results", "sensitivity", tag)
            workdir = os.path.join(ROOT, "results", "sensitivity", tag + "_work")
            # 每次都从干净状态跑：复用 workdir 会带上上一次的 registry / checkpoint / ledger
            for d in (outdir, workdir):
                if os.path.isdir(d):
                    shutil.rmtree(d)
            cmd = [PY, "-X", "utf8", MAIN, "--data", csv_path, "--outdir", outdir,
                   "--workdir", workdir] + extra
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            if p.returncode != 0:
                print(f"[FAIL] {tag}\n{p.stderr[-500:]}")
                line[gname] = None
                continue
            with open(os.path.join(outdir, "run_summary.json"), encoding="utf-8") as fh:
                s = json.load(fh)
            line[gname] = s
        rows.append(line)

    print(f"\n{'漂移':<8}{'基线F1':>9}{'本方案F1':>10}{'增益(pp)':>10}{'基线AP':>9}{'本方案AP':>10}{'AP增益':>9}")
    for r in rows:
        b, o = r.get("冻结基线"), r.get("本方案")
        if not b or not o:
            continue
        print(f"{r['level']:<8}{b['overall']['f1']:>9.4f}{o['overall']['f1']:>10.4f}"
              f"{(o['overall']['f1'] - b['overall']['f1']) * 100:>+10.2f}"
              f"{b['overall']['ap']:>9.4f}{o['overall']['ap']:>10.4f}"
              f"{(o['overall']['ap'] - b['overall']['ap']) * 100:>+9.2f}")

    out = os.path.join(ROOT, "results", "sensitivity", "summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump([{k: v for k, v in r.items() if k in ("level", "strength")} |
                   {k: (v["overall"] if v else None) for k, v in r.items() if k not in ("level", "strength")}
                   for r in rows], fh, ensure_ascii=False, indent=2)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
