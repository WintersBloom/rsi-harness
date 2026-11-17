"""Local stand-in host for workspace-v2 rounds (no platform involved).

Runs `python app/harness.py train` as a child process, speaks the frame protocol,
persists `save_state` output into work/state so that round N+1 can read it, and
prints what each round produced.

Usage:
    python fake_host.py [A|B|C] [rounds]
"""
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app"
WORK = ROOT / "work"
STATE = WORK / "state"
OFFICIAL_ROOT = Path(
    r"C:\Users\李林峯\Documents\deepseek-harness\default-workspace\data\official"
)
PACKAGES = {
    "A": OFFICIAL_ROOT / "cpu-a-guide-20261110-v1" / "public",
    "B": OFFICIAL_ROOT / "cpu-b-rul-20261110-v1" / "public",
    "C": OFFICIAL_ROOT / "cpu-c-guide-20261110-v1" / "public",
}

PROTOCOL = "contest.harness.workspace.v2"


def start_frame(dataset_key="A", round_index=1, parent_sha="sha256:" + "ab" * 32):
    state_path = str(STATE) if STATE.is_dir() else None
    return {
        "protocol": PROTOCOL,
        "prompt": "【本地假提示】这是一次协议自检，不涉及真实评分。",
        "prompt_version": "industrial-workspace-flexible-v3",
        "dataset_key": dataset_key,
        "round_index": round_index,
        "stage_id": "local",
        "task_contract": {
            "task": "classification" if dataset_key == "A" else "regression",
            "metric": "macro_f1" if dataset_key == "A" else "rmse",
            "metric_direction": "higher" if dataset_key == "A" else "lower",
            "target_unit": "label" if dataset_key == "A" else "unit",
            "public_projection": {
                "train_path": str(PACKAGES[dataset_key] / "train.jsonl"),
                "validation_path": str(PACKAGES[dataset_key] / "validation.jsonl"),
            },
        },
        "model_name": "mock/qwen-local",
        "max_output_tokens": 256,
        "parent_checkpoint_sha256": parent_sha,
        "parent_artifact": {
            "artifact_id": "parent-%03d" % round_index,
            "artifact": {
                "checkpoint_paths": ["checkpoint.npz"],
                "code_paths": [],
                "predict_argv": ["python3", "/app/harness.py", "predict"],
            },
            "files": [
                {"path": "checkpoint.npz", "size_bytes": 1024, "sha256": parent_sha}
            ],
        },
        "artifacts": [],
        "state_path": state_path,
        "state_sha256": "",
        "resource_limits": {},
        "wall_seconds": 600,
    }


def reply_model(frame, text="选择 retrain_10，理由是上一轮短程续训还没试过。"):
    return {
        "type": "model_response",
        "id": frame["id"],
        "call_id": "call-local-0001",
        "response": {
            "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 320, "completion_tokens": 18},
        },
    }


def accept_state(frame):
    """Persist file payloads into WORK/state, return the digest the host reports."""
    files = frame.get("files") or []
    if STATE.exists():
        shutil.rmtree(STATE)
    STATE.mkdir(parents=True)
    digest = hashlib.sha256()
    for item in files:
        rel = item["path"]
        target = STATE / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = base64.b64decode(item["data_base64"])
        target.write_bytes(data)
        digest.update(rel.encode() + b"\0" + data)
    return "sha256:" + digest.hexdigest()


def serve(proc, start, transcript):
    proc.stdin.write(json.dumps(start, ensure_ascii=False) + "\n")
    proc.stdin.flush()

    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("child closed stdout")
        transcript.append(("in", line.strip()[:300]))
        frame = json.loads(line)
        kind = frame.get("type")

        if kind == "round_failed":
            raise RuntimeError("submission failed: %s" % frame.get("error"))
        if kind == "round_complete":
            record = frame["record"]
            proc.stdin.write(json.dumps({"type": "round_ack", "accepted": True}) + "\n")
            proc.stdin.flush()
            proc.stdin.close()
            return record
        if kind == "history_request":
            reply = {"type": "history_response", "id": frame["id"], "records": []}
        elif kind == "model_request":
            reply = reply_model(frame)
        elif kind == "state_request":
            reply = {
                "type": "state_response",
                "id": frame["id"],
                "state_sha256": accept_state(frame),
            }
        elif kind == "candidate_request":
            files = frame.get("files") or []
            reply = {
                "type": "candidate_response",
                "id": frame["id"],
                "artifact_id": "model-local-%04d" % len(files),
                "summary": {"metrics": {}, "files": len(files)},
            }
        else:
            raise RuntimeError("unexpected frame %s" % kind)

        proc.stdin.write(json.dumps(reply, ensure_ascii=False) + "\n")
        proc.stdin.flush()


def run_rounds(dataset_key, rounds):
    env = dict(os.environ)
    env["CONTEST_WORK"] = str(WORK)
    env["PYTHONPATH"] = str(APP)
    env["PYTHONIOENCODING"] = "utf-8"

    parent_sha = "sha256:" + "ab" * 32
    for index in range(1, rounds + 1):
        work_dir = WORK / ("r%02d" % index)
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        env["CONTEST_WORK"] = str(work_dir)

        proc = subprocess.Popen(
            [sys.executable, str(APP / "contest_loop_entry.py"), "train"]
            if (APP / "contest_loop_entry.py").exists()
            else [sys.executable, str(APP / "harness.py"), "train"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(APP), env=env, text=True, encoding="utf-8", bufsize=1,
        )
        transcript = []
        record = None
        error = None
        try:
            record = serve(proc, start_frame(dataset_key, index, parent_sha), transcript)
        except Exception as exc:  # keep stderr for diagnosis
            error = exc
        finally:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            stderr = proc.stderr.read()

        if error is not None:
            print("\n=== round %d FAILED: %s ===" % (index, error))
            print("frames:", [t[1][:120] for t in transcript])
            print("stderr:\n" + stderr.strip()[:4000])
            raise SystemExit(1)

        state_file = STATE / "experience.json"
        exp = None
        if state_file.is_file():
            exp = json.loads(state_file.read_text(encoding="utf-8"))

        print("\n=== round %d (exit=%s) ===" % (index, proc.returncode))
        print("frames:", [t[1][:80] for t in transcript[:4]])
        print("record.keys:", sorted(record.keys()))
        cfg = record.get("configuration") or {}
        print("configuration.directions:", json.dumps(cfg.get("directions"), ensure_ascii=False))
        print("configuration.plateau:", json.dumps(cfg.get("plateau"), ensure_ascii=False))
        print("rounds_since_accept:", cfg.get("rounds_since_accept"))
        print("model_call_ids:", len(record.get("model_call_ids") or []))
        if exp is not None:
            print("state.rounds:", exp.get("rounds"),
                  "state.dead:", exp.get("dead_directions"),
                  "state.directions:", list((exp.get("directions") or {}).keys()))
        if stderr.strip():
            print("stderr:", stderr.strip()[:800])

        parent_sha = "sha256:" + hashlib.sha256(
            json.dumps(record.get("configuration") or {}, sort_keys=True).encode()
        ).hexdigest()


def main():
    dataset_key = sys.argv[1] if len(sys.argv) > 1 else "A"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    WORK.mkdir(parents=True, exist_ok=True)
    for stale in ("candidate", "state"):
        target = WORK / stale
        if target.exists():
            shutil.rmtree(target)
    run_rounds(dataset_key, rounds)


if __name__ == "__main__":
    main()
