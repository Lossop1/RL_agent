"""
tail_log.py -- Display remote log file content in a separate terminal window.

Starts a subprocess in a new cmd window that polls the remote file
at a given interval and prints new content to that window.
The agent reads the same file via ssh_exec for its own analysis.

Supports:
- log_path as a directory: auto-finds the latest train.log inside
- Window reuse: same window_title reuses existing window
"""

import json
import os
import subprocess
import signal
from pathlib import Path

TOOL = {
    "name": "tail_log",
    "description": (
        "Display remote log file content in a separate terminal window. "
        "Starts a subprocess in a new cmd window that polls the remote file "
        "and prints new content to that window, keeping the main window clean. "
        "The agent reads the same file via ssh_exec for its own analysis. "
        "log_path can be a directory (auto-finds latest train.log) or a file path. "
        "window_title is used for reuse: same title reuses the same window."
    ),
    "parameters": {
        "log_path": {
            "type": "str",
            "required": True,
            "desc": "Remote log file or directory path. If directory, monitors latest train.log inside.",
        },
        "action": {
            "type": "str",
            "required": True,
            "desc": "'start' to begin polling in new window, 'stop' to kill the window",
        },
        "interval": {
            "type": "int",
            "required": False,
            "desc": "Polling interval in seconds, default 5. Set based on per-iteration training time.",
        },
        "window_title": {
            "type": "str",
            "required": False,
            "desc": "Window title for reuse. Same title reuses the same window. Default: 'tail_log'.",
        },
    },
}

_procs: dict = {}  # window_title -> subprocess.Popen


def execute(log_path: str, action: str, interval: int = 5, window_title: str = "tail_log") -> dict:
    if action == "start":
        # 同 window_title 已存在 → 先杀掉
        if window_title in _procs:
            proc = _procs[window_title]
            if proc.poll() is None:
                try:
                    if os.name == "nt":
                        subprocess.run(f"taskkill /T /F /PID {proc.pid}", shell=True, capture_output=True)
                    else:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.kill()
            del _procs[window_title]

        # Read SSH config
        ssh_cfg_path = Path(__file__).resolve().parent.parent / "config" / "ssh.json"
        try:
            with open(ssh_cfg_path, "r") as f:
                ssh_cfg = json.load(f)
        except Exception as e:
            return {"error": f"failed to read ssh config: {e}", "stdout": "", "stderr": ""}

        host = ssh_cfg.get("ssh_host", "")
        port = ssh_cfg.get("ssh_port", "22")
        user = ssh_cfg.get("ssh_user", "root")
        pwd = ssh_cfg.get("ssh_pass", "")

        viewer = Path(__file__).resolve().parent / "tail_log_viewer.py"

        proc = subprocess.Popen(
            ["python", str(viewer),
             "--log-path", log_path,
             "--interval", str(interval),
             "--ssh-host", host,
             "--ssh-port", str(port),
             "--ssh-user", user,
             "--ssh-pass", pwd],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        _procs[window_title] = proc
        return {"stdout": f"[tail_log] started monitoring: {log_path} (window={window_title}, interval={interval}s)", "stderr": ""}

    elif action == "stop":
        if window_title not in _procs:
            return {"stdout": f"[tail_log] not monitoring (window={window_title})", "stderr": ""}

        proc = _procs[window_title]
        if proc.poll() is None:
            try:
                if os.name == "nt":
                    subprocess.run(f"taskkill /T /F /PID {proc.pid}", shell=True, capture_output=True)
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.kill()
        del _procs[window_title]
        return {"stdout": f"[tail_log] stopped monitoring (window={window_title})", "stderr": ""}

    else:
        return {"error": f"unknown action: {action}", "stdout": "", "stderr": ""}