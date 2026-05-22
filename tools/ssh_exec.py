"""
ssh_exec.py — Remote SSH execution via paramiko.

维护一个持久化 SSH 连接（自动重连），凭据从运行时配置读取。
Agent 应用此工具执行所有远程服务器命令，而不是在 bash/cmd 里手写 ssh 命令。
"""
import threading
import time
import re
import runtime_state

TOOL = {
    "name": "ssh_exec",
    "description": (
        "Execute a command on the remote GPU server via SSH. "
            "Connection and authentication are handled automatically using loaded runtime config. "
        "Use this for ALL remote server operations (training, log check, file read/edit on server)."
    ),
    "parameters": {
        "command": {
            "type": "str",
            "required": True,
            "desc": "Shell command to run on the remote server (runs in work_dir unless you specify cd)"
        },
        "timeout": {
            "type": "int",
            "required": False,
            "desc": "Timeout in seconds, default 60 (use longer for commands that take time to start)"
        },
        "silent": {
            "type": "bool",
            "required": False,
            "desc": "If true, suppress stdout/stderr from terminal output (agent still sees it). Default false."
        },
    }
}

MAX_OUTPUT_CHARS = 20_000
_client = None
_lock = threading.Lock()


def _auto_adjust_timeout(user_command: str, timeout: int) -> int:
    """Avoid false timeout for explicit wait commands like 'sleep 90'."""
    if timeout <= 0:
        return timeout

    m = re.match(r"^\s*sleep\s+(\d+)\s*$", str(user_command or ""))
    if not m:
        return timeout

    wait_sec = int(m.group(1))
    # Keep a small buffer for transport/command overhead.
    return max(timeout, wait_sec + 10)


def _connect():
    """Create and return a connected paramiko SSHClient using runtime config."""
    import paramiko
    c = runtime_state.get_config()
    host = c.get("ssh_host", "")
    port = int(c.get("ssh_port", 22))
    user = c.get("ssh_user", "root")
    pwd  = c.get("ssh_pass", "")
    if not host:
        raise RuntimeError("ssh_host not set in loaded runtime config")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if pwd:
        client.connect(hostname=host, port=port, username=user,
                       password=pwd, timeout=10, banner_timeout=15)
    else:
        # Key-based auth (use default key from ~/.ssh/)
        client.connect(hostname=host, port=port, username=user, timeout=10)
    return client


def _get_client():
    """Return cached client, reconnecting if the transport died."""
    global _client
    with _lock:
        try:
            if _client and _client.get_transport() and _client.get_transport().is_active():
                return _client
        except Exception:
            pass
        _client = _connect()
        return _client


def disconnect():
    """Close the persistent SSH connection (called at agent shutdown)."""
    global _client
    with _lock:
        if _client:
            try:
                _client.close()
            except Exception:
                pass
            _client = None


def execute(command: str, timeout: int = 60, silent: bool = False) -> dict:
    """
    Run a command on the remote server.
    silent=true suppresses stdout/stderr from terminal output (agent still sees it).
    Returns: {stdout, stderr, exit_code, error}
    """
    raw_command = command
    timeout = _auto_adjust_timeout(raw_command, timeout)

    # Prepend shell_init (conda activate, env vars, etc.) from loaded runtime config
    try:
        c = runtime_state.get_config()
        shell_init = c.get("shell_init", "")
        work_dir   = c.get("work_dir", "")
        if shell_init and work_dir:
            command = f"{shell_init} && cd {work_dir} && {command}"
        elif shell_init:
            command = f"{shell_init} && {command}"
        elif work_dir:
            command = f"cd {work_dir} && {command}"
    except Exception:
        pass

    # Try once, then retry once on connection failure
    for attempt in range(2):
        try:
            client = _get_client()
            transport = client.get_transport()
            chan = transport.open_session()
            chan.settimeout(timeout)
            chan.exec_command(command)

            stdout_buf = b""
            stderr_buf = b""
            deadline = time.time() + timeout

            while True:
                if time.time() > deadline:
                    chan.close()
                    return {
                        "stdout": stdout_buf.decode("utf-8", errors="replace")[:MAX_OUTPUT_CHARS],
                        "stderr": stderr_buf.decode("utf-8", errors="replace"),
                        "exit_code": -1,
                        "error": f"timeout ({timeout}s) — command may still be running on server"
                    }
                if chan.recv_ready():
                    stdout_buf += chan.recv(65536)
                if chan.recv_stderr_ready():
                    stderr_buf += chan.recv_stderr(65536)
                if chan.exit_status_ready():
                    break
                time.sleep(0.05)

            # Drain anything left
            while chan.recv_ready():
                stdout_buf += chan.recv(65536)
            while chan.recv_stderr_ready():
                stderr_buf += chan.recv_stderr(65536)

            exit_code = chan.recv_exit_status()
            chan.close()

            stdout = stdout_buf.decode("utf-8", errors="replace")
            stderr = stderr_buf.decode("utf-8", errors="replace")
            if len(stdout) > MAX_OUTPUT_CHARS:
                stdout = "...[head truncated]\n" + stdout[-MAX_OUTPUT_CHARS:]

            return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code, "error": None}

        except Exception as e:
            disconnect()  # Force reconnect on next attempt
            if attempt == 1:
                return {"stdout": "", "stderr": str(e), "exit_code": -1,
                        "error": f"SSH failed after retry: {e}"}
            # First failure: loop back and retry once
            time.sleep(1)
