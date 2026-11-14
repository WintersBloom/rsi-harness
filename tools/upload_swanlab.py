"""把 swanlog/ 下的离线 run 一次性回放到 swanlab.cn。

SwanLab 的离线记录早就跑通了（9 个 run 在 `swanlog/`），缺的只是 API Key。
这个脚本把"登录 -> 校验 -> 逐个 sync -> 报告"串成一条命令，Key 到手后不用再想参数。

先看计划（不联网、不需要 Key）：

    python tools/upload_swanlab.py --dry-run

真正上传：

    $env:SWANLAB_API_KEY = "你的key"
    python tools/upload_swanlab.py

## 两个坑（2026-11-06 实测）

1. **必须用 DSH runtime 的 python（3.12）**。`C:\\Python311\\python.exe` 是 3.11，
   加载 `.pylibs` 里的 `pydantic_core` 会报
   `ImportError: cannot import name 'ArgsKwargs' from 'pydantic_core._pydantic_core'`。
   本脚本开头会检查，用错了解释器会直接说清楚。
2. **离线 run 的 `swanlab-metadata.json` 里没有 project 字段**，所以 sync 必须显式
   `-p`，否则会去找"上一次用过的项目"，第一次跑就找不到。

## 为什么输出全用英文

Windows 控制台代码页是 GBK，Python 默认吐 UTF-8 字节，中文提示过了子进程就成
`��Ҫִ��`。`tools/swanlog.py` 的 DESC_EN 就是为同一个坑写的。输出保持 ASCII，
中文只留在注释里。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBS = os.path.join(ROOT, ".pylibs")
SWANLOG = os.path.join(ROOT, "swanlog")

if os.path.isdir(LIBS):
    sys.path.insert(0, LIBS)


def check_env() -> None:
    """早期失败比跑到一半失败好。"""
    if sys.version_info < (3, 12):
        raise SystemExit(
            f"ERROR: running on Python {sys.version.split()[0]}, need 3.12.\n"
            "Use the DSH runtime interpreter instead, e.g.\n"
            r"  %USERPROFILE%\.dsh\dsh-runtimes\dsh-next-primary-runtime\dependencies\python\python.exe tools\upload_swanlab.py"
        )
    try:
        import swanlab  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"ERROR: cannot import swanlab: {exc}\nCheck that .pylibs is complete.")


def swanlab_cli(args: list[str]) -> int:
    """以子进程方式调 `python -m swanlab`，与文档里的命令逐字一致，排错时好对。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = LIBS + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run([sys.executable, "-m", "swanlab", *args], env=env, cwd=ROOT)
    return proc.returncode


def list_runs() -> list[str]:
    if not os.path.isdir(SWANLOG):
        raise SystemExit(f"ERROR: no swanlog/ dir at {SWANLOG}\nRun tools/swanlog.py --mode offline first.")
    runs = sorted(
        os.path.join(SWANLOG, d) for d in os.listdir(SWANLOG)
        if d.startswith("run-") and os.path.isdir(os.path.join(SWANLOG, d))
    )
    if not runs:
        raise SystemExit("ERROR: no run-* dirs under swanlog/")
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="zhipin-engine")
    ap.add_argument("--api-key", default=os.environ.get("SWANLAB_API_KEY", ""))
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不联网")
    args = ap.parse_args()

    runs = list_runs()

    print(f"python : {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    print(f"libs   : {LIBS}")
    print(f"project: {args.project}")
    print(f"runs   : {len(runs)}")

    if args.dry_run:
        print("\n[dry-run] commands to run:")
        print("  python -m swanlab login --api-key <key>")
        print("  python -m swanlab verify")
        for r in runs:
            print(f"  python -m swanlab sync {r} -p {args.project}")
        print(f"\nSWANLAB_API_KEY: {'set' if args.api_key else 'MISSING'}")
        return 0

    check_env()

    if not args.api_key:
        raise SystemExit("ERROR: no API key. Set SWANLAB_API_KEY or pass --api-key.")

    print("\n[1/2] login")
    if swanlab_cli(["login", "--api-key", args.api_key]) != 0:
        raise SystemExit("ERROR: login failed. Check the key and connectivity to swanlab.cn.")

    if swanlab_cli(["verify"]) != 0:
        raise SystemExit("ERROR: verify still fails after login. Check network / account state.")

    print(f"\n[2/2] uploading {len(runs)} runs")
    ok, failed = 0, []
    for r in runs:
        name = os.path.basename(r)
        print(f"  -> {name}", end="", flush=True)
        if swanlab_cli(["sync", r, "-p", args.project]) == 0:
            ok += 1
            print("  ok")
        else:
            failed.append(name)
            print("  FAILED")

    print(f"\ndone: {ok} / {len(runs)} uploaded")
    if failed:
        print("failed:")
        for n in failed:
            print(f"  - {n}")
        return 1

    print(
        "\nmanual steps left (CLI cannot change visibility):\n"
        f"  1. open https://swanlab.cn and go to project {args.project}\n"
        "  2. set visibility to Public -- the rules require a public link\n"
        "  3. paste the project URL into README.md deliverables table\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
