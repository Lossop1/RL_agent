"""
scp_copy.py — Copy files from remote server to local machine via paramiko SFTP.

Reuses the same SSH credentials as ssh_exec from runtime_state.
"""
import os
import runtime_state

TOOL = {
    "name": "scp_copy",
    "description": (
        "Copy a file from the remote GPU server to the local machine via SFTP. "
        "Uses the same SSH connection credentials as ssh_exec. "
        "The remote_path is relative to the remote work_dir unless an absolute path is given."
    ),
    "parameters": {
        "remote_path": {
            "type": "str",
            "required": True,
            "desc": "Remote file path to copy (absolute, or relative to remote work_dir)"
        },
        "local_path": {
            "type": "str",
            "required": True,
            "desc": "Local destination path (including filename)"
        },
    }
}

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB


def _connect():
    import paramiko
    c = runtime_state.get_config()
    host = c.get("ssh_host", "")
    port = int(c.get("ssh_port", 22))
    user = c.get("ssh_user", "root")
    pwd = c.get("ssh_pass", "")
    if not host:
        raise RuntimeError("ssh_host not set in loaded runtime config")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if pwd:
        client.connect(hostname=host, port=port, username=user,
                       password=pwd, timeout=10, banner_timeout=15)
    else:
        client.connect(hostname=host, port=port, username=user, timeout=10)
    return client


def execute(remote_path: str, local_path: str) -> dict:
    try:
        c = runtime_state.get_config()
        work_dir = c.get("work_dir", "")
        if not os.path.isabs(remote_path) and work_dir:
            remote_path = os.path.join(work_dir, remote_path).replace("\\", "/")

        client = _connect()
        sftp = client.open_sftp()

        try:
            stat = sftp.stat(remote_path)
            if stat.st_size > MAX_FILE_SIZE:
                sftp.close()
                client.close()
                return {
                    "error": f"File too large: {stat.st_size / 1024 / 1024:.1f}MB > 500MB limit"
                }
        except FileNotFoundError:
            sftp.close()
            client.close()
            return {"error": f"Remote file not found: {remote_path}"}

        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        sftp.get(remote_path, local_path)
        sftp.close()
        client.close()

        local_size = os.path.getsize(local_path)
        return {
            "stdout": f"Copied {remote_path} -> {local_path} ({local_size / 1024:.1f} KB)",
            "stderr": "",
            "local_path": local_path,
            "size_bytes": local_size,
        }

    except Exception as e:
        return {"error": f"scp_copy failed: {type(e).__name__}: {e}"}
