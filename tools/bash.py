# import paramiko
# import time
# import re
# #description of tool
# TOOL ={
#     "name":"bash",
#     "description":"excecute shell commands on a remote server, cd and envirment variables persists throughout the entire session ",
#     "parameters":{
#         "command":{"type":"str","required":True,"desc":"command to be excecuted"},
#         "timeout":{"type":"int","required":False, "desc" : "timeout in seconds,default is 30s"}
#     }
# }
# BLOCKED_PATTERNS = [
#     "rm -rf /",
#     "mkfs",
#     "dd if=",
#     "> /dev/sda",
# ]
# MAX_OUTPUT_CHARS = 20000
# SENTINEL = "__CMD_DONE__"

# #global client,persistent connection
# _client = None
# _shell = None


# #connect function
# def connect(host, port, user, password=None, key_path=None):
#     global _client, _shell
#     try:
#         _client = paramiko.SSHClient()
#         _client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#         _client.connect(hostname=host, port=port, username=user,
#                         password=password, key_filename=key_path)
#         _shell = _client.invoke_shell()
#         _shell.settimeout(0.1)
#         time.sleep(0.5)
#         _drain()
#         return {"ok": True, "error": None}
#     except paramiko.AuthenticationException:
#         _client = None
#         _shell = None
#         return {"ok": False, "error": "auth_failed"}
#     except Exception as e:
#         _client = None
#         _shell = None
#         return {"ok": False, "error": str(e)}
# #drain function,drain the data out from buffer by taking it with recv
# def _drain():
#     while True:
#         try:
#             _shell.recv(65536)
#         except Exception:
#             break
# #disconnect function 
# def disconnect():
#     global _client,_shell
#     if _shell:
#         _shell.close()
#         _shell = None
#     if _client:
#         _client.close()
#         _client = None
# #excecute function
# def execute(command,timeout=30):
#     #check connection
#     if not _shell:
#         return {"stdout": "", "stderr": "SSH not connected", "exit_code": -1}
#     #check blocked patterns
#     for pattern in BLOCKED_PATTERNS:
#         if pattern in command:
#             return {"stdout":"","stderr":"Blocked pattern detected","exit_code":-1}
#     #the "$?" is the exit code, invoke mode execute the command by "send", which is persistent,but can't get the exit code.
#     _shell.send(f"{command}\necho \"{SENTINEL}$?\"\n")
#     output = ""
#     start = time.time()
#     while time.time() - start < timeout:
#         try:
#             #yes, the data took by recv() would be removed from buffer
#             chunk = _shell.recv(65536).decode("utf-8",errors="replace")
#             output += chunk
#             if SENTINEL in chunk:
#                 break
#         except Exception:
#             time.sleep(0.1)
#     else:
#         return {
#         "stdout": output[:MAX_OUTPUT_CHARS] + "\n...[TIMEOUT]...",
#         "stderr": f"超时({timeout}s)，命令可能仍在后台运行",
#         "exit_code": -1
#     }

#     #analyse exit code
#     match = re.search(rf"{SENTINEL}(\d+)",output)
#     #() is the cpture group signala
#     exit_code = int(match.group(1)) if match else -1

#     lines = output.split("\n")
#     clean = [l for l in lines if SENTINEL not in l and l.strip() != command.strip()]
#     stdout = "\n".join(clean).strip()
#     if len(stdout) > MAX_OUTPUT_CHARS:
#         stdout = stdout[:MAX_OUTPUT_CHARS] + "\n...[truncated]..."
#     return {"stdout": stdout, "stderr": "", "exit_code": exit_code}


#     # #execute command，execute() just do the check, the real execute is done by _client
#     # try:
#     #     stdin,stdout,stderr=_client.exec_command(command,timeout=timeout)
#     #     exit_code = stdout.channel.recv_exit_status()
#     #     out = stdout.read().decode("utf-8",errors="replace")
#     #     err = stderr.read().decode("utf-8",errors="replace")
#     # except Exception as e:
#     #     return {"stdout":"","stderr":str(e),"exit_code":-1}
#     # #truncate output
#     # if len(out) > MAX_OUTPUT_CHARS:
#     #     out = out[:MAX_OUTPUT_CHARS] + "...[truncated]..."
#     # if len(err) > MAX_OUTPUT_CHARS:
#     #     err = err[:MAX_OUTPUT_CHARS] + "...[truncated]..."
#     # return {"stdout":out,"stderr":err,"exit_code":exit_code}

#the above is the version that can only execute commands via SSH, the
#following is the local bash, with SSH implemented through bash.
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