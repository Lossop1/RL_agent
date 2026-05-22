import re
import json
import time
import pathlib

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[\\=/>]')

def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub('', text)

TOOL = {
    "name": "training_status",
    "description": (
        "Check remote RL training status. Returns structured data: "
        "latest reward, current step, whether training is running, NaN detection, "
        "and log staleness. Also appends to local training_history.jsonl for trend analysis. "
        "If tensorboard_dir is provided, also parses TensorBoard events for full metrics "
        "(loss, entropy, action_std, grad_norm, kl_divergence, lr_scale, etc.)."
    ),
    "parameters": {
        "log_file": {
            "type": "str", "required": True,
            "desc": "absolute path of remote training log file"
        },
        "tensorboard_dir": {
            "type": "str", "required": False,
            "desc": "absolute path of remote TensorBoard log directory (parent of events.*.tfevents.*)"
        },
        "log_reward_pattern": {
            "type": "str", "required": False,
            "desc": "regex to parse reward from log, default episode_reward_mean pattern"
        },
        "tail_lines": {
            "type": "int", "required": False,
            "desc": "number of log lines to tail, default 60"
        },
    }
}

_HISTORY_FILE = pathlib.Path("training_history.jsonl")


def _parse_reward(log_tail: str, pattern: str) -> float | None:
    """Extract the last matching reward value from log tail."""
    matches = re.findall(pattern, log_tail)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _parse_step(log_tail: str) -> int | None:
    """Heuristic: find the last occurrence of a bare integer that looks like a step."""
    patterns = [
        r"[Ii]teration[:\s]+(\d+)",
        r"[Ss]tep[:\s]+(\d+)",
        r"\[(\d+)/\d+\]",
    ]
    for pat in patterns:
        matches = re.findall(pat, log_tail)
        if matches:
            try:
                return int(matches[-1])
            except ValueError:
                pass
    return None


def _parse_cusrl_iterations(log_tail: str) -> list[dict]:
    """Parse cusrl iteration blocks from log tail.

    Format:
    ┌────── Iteration N / MAX ───────┐
    │ mean episode length     XXX    │
    │ mean episode reward     XXX    │
    │ mean step reward        XXX    │
    │ time consumption     X / Y     │
    └──────────────────────────────────┘
    """
    blocks = re.findall(
        r"Iteration\s+(\d+)\s*/\s*(\d+).*?"
        r"mean episode length\s+([\d.]+).*?"
        r"mean episode reward\s+([\d.\-]+).*?"
        r"mean step reward\s+([\d.\-]+).*?"
        r"time consumption\s+([\d.]+)\s*/\s*([\d.]+)",
        log_tail, re.DOTALL
    )
    result = []
    for b in blocks:
        result.append({
            "iteration": int(b[0]),
            "max_iterations": int(b[1]),
            "mean_episode_length": float(b[2]),
            "mean_episode_reward": float(b[3]),
            "mean_step_reward": float(b[4]),
            "train_time_sec": float(b[5]),
            "inference_time_sec": float(b[6]),
        })
    return result


def _parse_tensorboard_remote(ssh_exec_func, tb_dir: str) -> dict:
    """Parse TensorBoard events file on remote via SSH using Python EventAccumulator."""
    script = f"""
import json, sys, os
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator("{tb_dir}")
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    result = {{"tags": tags, "data": {{}}}}
    for tag in tags:
        events = ea.Scalars(tag)
        if events:
            # Return last 3 values for trend, plus latest
            result["data"][tag] = {{
                "latest": events[-1].value,
                "step": events[-1].step,
                "recent": [e.value for e in events[-5:]]
            }}
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
    # Escape the script for safe SSH passing
    escaped = script.replace('"', '\\"').replace('\n', '\\n')
    cmd = f'python3 -c "{escaped}"'
    ssh_result = ssh_exec_func(cmd, timeout=30)
    stdout = ssh_result.get("stdout", "").strip()
    if stdout:
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return {"error": f"failed to parse tensorboard output: {stdout[:200]}"}
    return {"error": "no output from tensorboard parser"}


def execute(log_file: str, tensorboard_dir: str = None,
            log_reward_pattern: str = r"episode_reward_mean:\s*([0-9.\-]+)",
            tail_lines: int = 60) -> dict:
    from tools.ssh_exec import execute as ssh_exec

    if not str(log_file or "").strip():
        return {"error": "log_file is required"}

    pattern = str(log_reward_pattern or r"episode_reward_mean:\s*([0-9.\-]+)")

    # 1. Check if training process is alive
    pgrep_result = ssh_exec("pgrep -f train.py -a 2>/dev/null", timeout=15)
    is_running = (
        pgrep_result.get("exit_code") == 0
        and bool(pgrep_result.get("stdout", "").strip())
    )

    # 2. Extract reward lines via grep
    _kw_match = re.match(r"^([A-Za-z0-9 _:]+)", pattern)
    keyword = _kw_match.group(1).strip() if _kw_match else "reward"
    grep_cmd = (
        f"grep -iE '({re.escape(keyword)}|[Ll]earning iteration|[Ee]pisode length|Iteration)'"
        f" {log_file} 2>/dev/null | tail -60"
    )
    grep_result = ssh_exec(grep_cmd, timeout=15)
    log_tail = grep_result.get("stdout", "")
    if not log_tail.strip():
        tail_result = ssh_exec(f"tail -n {tail_lines} {log_file} 2>/dev/null", timeout=15)
        log_tail = tail_result.get("stdout", "")

    # 3. Check log staleness
    mtime_result = ssh_exec(f"stat -c %Y {log_file} 2>/dev/null", timeout=10)
    log_mtime_age_sec = None
    mtime_str = mtime_result.get("stdout", "").strip()
    if mtime_str.isdigit():
        log_mtime_age_sec = int(time.time()) - int(mtime_str)

    # 4. Parse log
    log_tail = _strip_ansi(log_tail)
    latest_reward = _parse_reward(log_tail, pattern)
    step = _parse_step(log_tail)
    nan_detected = bool(re.search(r"\bnan\b|\binf\b", log_tail, re.IGNORECASE))
    iterations = _parse_cusrl_iterations(log_tail)

    result = {
        "latest_reward":       latest_reward,
        "step":                step,
        "is_running":          is_running,
        "nan_detected":        nan_detected,
        "log_mtime_age_sec":   log_mtime_age_sec,
        "log_tail":            log_tail[-2000:] if log_tail else "",
        "iterations":          iterations,
        "tensorboard_metrics": None,
        "error":               None,
    }

    # 5. Parse TensorBoard if dir provided
    if tensorboard_dir:
        tb_data = _parse_tensorboard_remote(ssh_exec, tensorboard_dir)
        if "error" not in tb_data:
            result["tensorboard_metrics"] = tb_data
        else:
            result["tensorboard_metrics"] = {"error": tb_data["error"]}

    # 6. Persist to local JSONL
    if latest_reward is not None:
        record = {
            "ts":      time.strftime("%Y-%m-%dT%H:%M:%S"),
            "step":    step,
            "reward":  latest_reward,
            "running": is_running,
        }
        try:
            with open(_HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass

    return result
