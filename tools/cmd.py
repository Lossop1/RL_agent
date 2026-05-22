"""
cmd.py — Persistent Windows cmd.exe session.

bash.py 的 Windows 原生版本：不依赖 WSL，用于本地 Windows 环境操作。
SSH 远程操作请用 ssh_exec 工具。
"""
import subprocess
import threading
import queue
import time
import re

TOOL = {
    "name": "cmd",
    "description": (
        "Execute commands in a persistent local Windows cmd.exe session. "
        "Use for local file exploration (dir, type, findstr) and Windows-native operations. "
        "For remote server commands, use ssh_exec instead."
    ),
    "parameters": {
        "command": {
            "type": "str",
            "required": True,
            "desc": "Windows cmd command to execute (use dir instead of ls, findstr instead of grep)"
        },
        "timeout": {
            "type": "int",
            "required": False,
            "desc": "Timeout in seconds, default 30"
        },
    }
}

BLOCKED_PATTERNS = [
    "rd /s /q C:\\",
    "format C:",
    "del /f /s /q C:\\",
]
MAX_OUTPUT_CHARS = 20_000
SENTINEL = "__CMD_DONE__"

_process = None
_stdout_queue = None
_stderr_queue = None

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[\\=/>]')


def _clean_output(text: str) -> str:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = _ANSI_RE.sub('', text)
    # Strip Windows cmd prompt lines like "C:\path>" or "D:\RL-Dog\...>"
    text = re.sub(r'^[A-Za-z]:\\[^>]*>', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _reader_thread(stream, q):
    try:
        for line in stream:
            q.put(line)
    finally:
        q.put(None)


def start_shell():
    global _process, _stdout_queue, _stderr_queue
    _process = subprocess.Popen(
        ["cmd.exe", "/Q"],   # /Q = quiet mode (suppress command echo globally)
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    _stdout_queue = queue.Queue()
    _stderr_queue = queue.Queue()
    threading.Thread(target=_reader_thread,
                     args=(_process.stdout, _stdout_queue), daemon=True).start()
    threading.Thread(target=_reader_thread,
                     args=(_process.stderr, _stderr_queue), daemon=True).start()

    # Switch to UTF-8 codepage so Chinese chars aren't mangled
    init_cmd = "chcp 65001 > nul\n"
    _process.stdin.buffer.write(init_cmd.encode("utf-8"))
    _process.stdin.buffer.flush()
    time.sleep(0.3)
    while True:
        try:
            _stdout_queue.get_nowait()
        except queue.Empty:
            break


def stop_shell():
    global _process, _stdout_queue, _stderr_queue
    if _process:
        try:
            _process.terminate()
        except OSError:
            pass
        _process = None
        _stdout_queue = None
        _stderr_queue = None


def _drain_queue(q: queue.Queue) -> str:
    parts = []
    while True:
        try:
            line = q.get_nowait()
            if line is None:
                break
            parts.append(line)
        except queue.Empty:
            break
    return "".join(parts)


def execute(command: str, timeout: int = 30) -> dict:
    global _process, _stdout_queue, _stderr_queue
    if not _process:
        try:
            start_shell()
        except Exception as e:
            return {"stdout": "", "stderr": "", "exit_code": -1, "error": str(e)}

    for pattern in BLOCKED_PATTERNS:
        if pattern.lower() in command.lower():
            return {"stdout": "", "stderr": "", "exit_code": -1,
                    "error": "blocked: dangerous command"}

    # ── Non-blocking: `start` commands open a new window, return immediately ──
    if command.strip().startswith("start "):
        try:
            subprocess.Popen(command, shell=True, close_fds=True)
        except Exception as e:
            return {"stdout": "", "stderr": "", "exit_code": -1, "error": str(e)}
        return {"stdout": f"[cmd] launched: {command}", "stderr": "", "exit_code": 0, "error": None}

    # In cmd.exe: %ERRORLEVEL% captures the exit code of the previous command
    # We append the sentinel AFTER the command on a separate line
    full_cmd = f"{command}\necho {SENTINEL}%ERRORLEVEL%\n"
    try:
        _process.stdin.buffer.write(full_cmd.encode("utf-8"))
        _process.stdin.buffer.flush()
    except (BrokenPipeError, OSError, ValueError) as e:
        stop_shell()
        return {"stdout": "", "stderr": "", "exit_code": -1,
                "error": f"cmd crashed: {e}. New shell on next call."}

    stdout_lines = []
    exit_code = -1
    start = time.time()

    while time.time() - start < timeout:
        try:
            line = _stdout_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if line is None:
            stderr_text = _drain_queue(_stderr_queue)
            stop_shell()
            return {"stdout": "".join(stdout_lines), "stderr": stderr_text,
                    "exit_code": -1, "error": "cmd process died unexpectedly."}
        if SENTINEL in line:
            match = re.search(rf"{SENTINEL}(\d+)", line)
            exit_code = int(match.group(1)) if match else -1
            break
        stdout_lines.append(line)
    else:
        stderr_text = _drain_queue(_stderr_queue)
        stdout = _clean_output("".join(stdout_lines))
        stop_shell()
        return {"stdout": stdout, "stderr": stderr_text,
                "exit_code": -1,
                "error": f"timeout({timeout}s), cmd killed, shell reset"}

    raw = "".join(stdout_lines)
    stdout = _clean_output(raw)
    if len(stdout) > MAX_OUTPUT_CHARS:
        stdout = "...[head truncated]\n" + stdout[-MAX_OUTPUT_CHARS:]
    stderr = _clean_output(_drain_queue(_stderr_queue))
    return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code, "error": None}
