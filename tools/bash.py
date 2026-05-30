import os
import subprocess
import threading
import queue
import time
import re

TOOL = {
    "name": "bash",
    "description": "execute shell commands in a persistent local bash session. Use 'ssh user@host' to connect to remote machines.",
    "parameters": {
        "command": {"type": "str", "required": True, "desc": "command to execute"},
        "timeout": {"type": "int", "required": False, "desc": "timeout in seconds, default 30"}
    }
}
BLOCKED_PATTERNS = [
    "rm -rf /",
    "mkfs",
    "dd if=",
    "> /dev/sda",
]
MAX_OUTPUT_CHARS = 20000
SENTINEL = "__CMD_DONE__"
_process = None
_stdout_queue = None
_stderr_queue = None

# ── Output cleaning ──────────────────────────────────────────────────────────
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[\\=/>]')

def _clean_output(text: str) -> str:
    """Strip ANSI codes, collapse \\r progress-bar overwrites, fold blank lines."""
    # 0. Normalize Windows CRLF → LF FIRST, before any \r handling.
    #    Without this, lines ending with \r\n become empty after the split below.
    text = text.replace('\r\n', '\n')
    # 1. Remove ANSI escape sequences (color, cursor movement, etc.)
    text = _ANSI_RE.sub('', text)
    # 2. Handle bare \r progress-bar overwrites: keep only the last write per line
    lines = text.split('\n')
    lines = [l.split('\r')[-1] for l in lines]
    text = '\n'.join(lines)
    # 3. Collapse 3+ consecutive blank lines into one blank line
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _reader_thread(stream, q):
    # Daemon thread: pump lines into queue, put None when stream closes.
    try:
        for line in stream:
            q.put(line)
    finally:
        q.put(None)

def start_shell():
    global _process, _stdout_queue, _stderr_queue
    try:
        _process = subprocess.Popen(
            ["bash"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   # separate stderr so errors are never lost
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )
        _stdout_queue = queue.Queue()
        _stderr_queue = queue.Queue()
        threading.Thread(target=_reader_thread, args=(_process.stdout, _stdout_queue), daemon=True).start()
        threading.Thread(target=_reader_thread, args=(_process.stderr, _stderr_queue), daemon=True).start()
    except OSError as e:
        _process = None
        _stdout_queue = None
        _stderr_queue = None
        raise RuntimeError(f"failed to start shell: {e}")

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
    """Non-blocking drain of a queue; returns all accumulated lines as a string."""
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

def execute(command, timeout=30):
    global _process, _stdout_queue, _stderr_queue
    # Auto-start on first call
    if not _process:
        try:
            start_shell()
        except RuntimeError as e:
            return {"stdout": "", "stderr": "", "exit_code": -1, "error": str(e)}
    # Blocked command check
    for pattern in BLOCKED_PATTERNS:
        if pattern in command:
            return {"stdout": "", "stderr": "", "exit_code": -1, "error": "blocked: dangerous command"}
    # Write to stdin via the binary buffer to prevent Windows text-mode from
    # translating \n → \r\n, which makes WSL bash see commands as "ls\r" etc.
    full_cmd = f"{command}\necho \"{SENTINEL}$?\"\n"
    try:
        _process.stdin.buffer.write(full_cmd.encode('utf-8'))
        _process.stdin.buffer.flush()
    except BrokenPipeError:
        stop_shell()
        return {"stdout": "", "stderr": "", "exit_code": -1,
                "error": "shell crashed (BrokenPipe). New shell on next call. cd/export state lost."}
    except (OSError, ValueError) as e:
        stop_shell()
        return {"stdout": "", "stderr": "", "exit_code": -1,
                "error": f"shell write failed ({type(e).__name__}): {e}. New shell on next call."}
    # Read stdout until sentinel, respecting wall-clock timeout
    stdout_lines = []
    exit_code = -1
    start = time.time()
    while time.time() - start < timeout:
        try:
            line = _stdout_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if line is None:
            # Reader thread EOF: process died mid-command
            stderr_text = _drain_queue(_stderr_queue)
            stop_shell()
            return {"stdout": "".join(stdout_lines), "stderr": stderr_text,
                    "exit_code": -1, "error": "shell died unexpectedly. cd/export state lost."}
        if SENTINEL in line:
            match = re.search(rf"{SENTINEL}(\d+)", line)
            exit_code = int(match.group(1)) if match else -1
            break
        stdout_lines.append(line)
    else:
        # Wall-clock timeout: kill the shell so next call starts clean.
        # Leaving it alive would block all subsequent commands on the stuck process.
        stderr_text = _drain_queue(_stderr_queue)
        raw = "".join(stdout_lines)
        stdout = _clean_output(raw)
        if len(stdout) > MAX_OUTPUT_CHARS:
            stdout = "...[head truncated]\n" + stdout[-MAX_OUTPUT_CHARS:]
        stop_shell()  # ← reset: next execute() call will spawn a fresh shell
        return {"stdout": stdout, "stderr": stderr_text,
                "exit_code": -1, "error": f"timeout({timeout}s), command killed, shell reset"}
    # Clean and truncate stdout (tail-direction: keep most recent output)
    raw_stdout = "".join(stdout_lines)
    stdout = _clean_output(raw_stdout)
    if len(stdout) > MAX_OUTPUT_CHARS:
        stdout = "...[head truncated]\n" + stdout[-MAX_OUTPUT_CHARS:]
    # Drain stderr completely (usually short; always keep in full)
    stderr = _clean_output(_drain_queue(_stderr_queue))
    return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code, "error": None}