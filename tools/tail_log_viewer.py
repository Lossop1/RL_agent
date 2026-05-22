"""
tail_log_viewer.py -- Standalone process that polls a remote log file
and prints new content to its own terminal window.

Launched by tail_log.py as a subprocess in a new cmd window.

Supports:
- log_path as a directory: auto-finds the latest train.log inside
- Auto-switch to new train.log when a newer one appears (for multi-round training)
"""

import argparse
import json
import os
import time
import paramiko



def find_latest_log(client, log_path):
    """If log_path is a directory, find the latest train.log inside.
    Checks the directory directly first, then searches subdirectories.
    Returns the resolved file path, or None if not found."""
    stdin, stdout, stderr = client.exec_command(f"test -d {log_path} && echo DIR || echo FILE")
    result = stdout.read().decode().strip()
    if result == "DIR":
        # First, check if train.log exists directly in this directory
        direct_log = f"{log_path.rstrip('/')}/train.log"
        stdin, stdout, stderr = client.exec_command(f"test -f {direct_log} && echo EXISTS || echo NOT_FOUND")
        if stdout.read().decode().strip() == "EXISTS":
            return direct_log

        # Otherwise, find latest subdirectory with train.log
        cmd = (
            f"ls -1t {log_path} 2>/dev/null | "
            f"while read d; do "
            f"  f={log_path}/$d/train.log; "
            f"  test -f \"$f\" && echo \"$f\" && break; "
            f"done"
        )
        stdin, stdout, stderr = client.exec_command(cmd)
        latest = stdout.read().decode().strip()
        if latest:
            return latest
        return None
    else:
        # It's a file, check if it exists
        stdin, stdout, stderr = client.exec_command(f"test -f {log_path} && echo EXISTS || echo NOT_FOUND")
        if stdout.read().decode().strip() == "EXISTS":
            return log_path
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--ssh-pass", default="")
    parser.add_argument("--window-title", default="tail_log")
    args = parser.parse_args()

    # 设窗口标题
    import ctypes
    ctypes.windll.kernel32.SetConsoleTitleW(args.window_title)

    last_size = 0
    current_file = None
    print(f"[tail_log] monitoring: {args.log_path} (interval={args.interval}s)")
    print(f"[tail_log] SSH: {args.ssh_user}@{args.ssh_host}:{args.ssh_port}")
    print("-" * 60)

    while True:
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            if args.ssh_pass:
                client.connect(
                    hostname=args.ssh_host, port=args.ssh_port,
                    username=args.ssh_user, password=args.ssh_pass,
                    timeout=10,
                )
            else:
                client.connect(
                    hostname=args.ssh_host, port=args.ssh_port,
                    username=args.ssh_user, timeout=10,
                )

            # Resolve latest log file (supports directory mode)
            resolved = find_latest_log(client, args.log_path)
            if resolved is None:
                # No log file yet, wait and retry
                client.close()
                time.sleep(args.interval)
                continue

            if resolved != current_file:
                if current_file is not None:
                    print(f"\n[tail_log] switched to: {resolved}")
                else:
                    print(f"[tail_log] tracking: {resolved}")
                current_file = resolved
                last_size = 0

            # Get current file size
            _, stdout, _ = client.exec_command(f"wc -c {current_file} 2>/dev/null")
            size_str = stdout.read().decode().strip().split()[0] if stdout else "0"
            current_size = int(size_str) if size_str.isdigit() else 0

            if current_size > last_size:
                cmd = (
                    f"dd if={current_file} bs=1 skip={last_size} "
                    f"count={current_size - last_size} 2>/dev/null"
                )
                _, stdout, _ = client.exec_command(cmd)
                new_data = stdout.read().decode("utf-8", errors="replace")
                if new_data:
                    print(new_data, end="", flush=True)
                last_size = current_size

            client.close()
            time.sleep(args.interval)

        except Exception as e:
            print(f"[tail_log] error: {e}")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()