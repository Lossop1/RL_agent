"""
tuner.py — 奖励配置修改执行器（纯工具，无 LLM 参与）

MCP Tool: tuner
"""
import json
import os
import re
import paramiko
from pathlib import Path

TOOL = {
    "name": "tuner",
    "description": "修改远程奖励配置文件。只允许操作 env_cfg.py。原子操作——任何一步失败自动回滚。",
    "parameters": {
        "remote_path": {"type": "str", "required": True, "desc": "远程奖励配置文件绝对路径"},
        "modifications": {"type": "object", "required": True, "desc": "修改列表 [{field, old_value, new_value}]"},
        "timestamp": {"type": "str", "required": True, "desc": "快照时间戳，用于备份目录命名"},
        "round": {"type": "int", "required": False, "desc": "当前轮数"},
    }
}

# ============================================================
# 权限边界
# ============================================================

ALLOWED_PATH_PREFIX = "/source/robot_lab/robot_lab/tasks/"
ALLOWED_FIELD_PREFIX = "self.rewards."
ALLOWED_FIELD_SUFFIX = ".weight"
SNAPSHOT_DIR = "data/snapshots"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# SSH 工具
# ============================================================

def _load_ssh_config():
    path = _PROJECT_ROOT / "config" / "ssh.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def _ssh_exec(cmd, timeout=15):
    """执行远程命令，返回 stdout"""
    cfg = _load_ssh_config()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=cfg.get("ssh_host", ""),
        port=int(cfg.get("ssh_port", 22)),
        username=cfg.get("ssh_user", "root"),
        password=cfg.get("ssh_pass", ""),
        timeout=10
    )
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    client.close()
    return out


# ============================================================
# 权限检查
# ============================================================

def _check_permission(remote_path, modifications, context_stack):
    """权限检查。不通过抛 PermissionError。"""
    # 调用者必须是 rectify
    caller = context_stack[-1] if context_stack and len(context_stack) >= 2 else None
    if caller != "rectify":
        raise PermissionError(f"tuner 只能由 rectify 调用，当前调用者: {caller}")

    # 路径必须在允许范围内
    if ALLOWED_PATH_PREFIX not in remote_path:
        raise PermissionError(f"不允许的路径: {remote_path}")

    # 每个 modification 必须符合格式
    for mod in modifications:
        field = mod.get("field", "")
        if not field.startswith(ALLOWED_FIELD_PREFIX) or not field.endswith(ALLOWED_FIELD_SUFFIX):
            raise PermissionError(f"不允许的修改: {field}，只允许修改 {ALLOWED_FIELD_PREFIX}*{ALLOWED_FIELD_SUFFIX}")
        for key in ("field", "old_value", "new_value"):
            if key not in mod:
                raise PermissionError(f"modification 缺少必需字段: {key}")


# ============================================================
# 主入口
# ============================================================

def execute(remote_path, modifications, timestamp, round=1, _control=None):
    """
    原子修改远程奖励配置。
    步骤：
      1. 权限检查
      2. 杀训练进程
      3. 备份远程文件到本地快照
      4. 逐条 sed 修改
      5. 逐条验证
      任何步骤失败 → 用备份回滚 → 返回 error
    """
    context_stack = _control.get("context_stack", []) if _control else []

    # ── 0. 权限检查 ──
    try:
        _check_permission(remote_path, modifications, context_stack)
    except PermissionError as e:
        return {"error": str(e)}

    # ── 1. 杀训练进程 ──
    try:
        _ssh_exec("pkill -f 'train.py' 2>/dev/null; echo done", timeout=10)
    except Exception as e:
        return {"error": f"终止训练进程失败: {e}"}

    # ── 2. 备份远程文件到本地 ──
    try:
        content = _ssh_exec(f"cat {remote_path}", timeout=10)
    except Exception as e:
        return {"error": f"读取远程文件失败: {e}"}

    snapshot_path = os.path.join(SNAPSHOT_DIR, timestamp, "env_cfg.py")
    try:
        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        with open(snapshot_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        return {"error": f"写入本地快照失败: {e}"}

    # ── 3. 逐条修改 ──
    changed = []
    try:
        for mod in modifications:
            # 转义特殊字符（负号、小数点）
            field = mod["field"].replace(".", "\\.")
            old = str(mod["old_value"]).replace("-", "\\-").replace(".", "\\.")
            new = str(mod["new_value"]).replace("-", "\\-").replace(".", "\\.")
            sed_cmd = f"sed -i 's/{field} = {old}/{field} = {new}/' {remote_path}"
            _ssh_exec(sed_cmd, timeout=10)
            changed.append(mod)
    except Exception as e:
        # 回滚
        _ssh_exec(f"cat > {remote_path} < {snapshot_path}", timeout=10)
        return {"error": f"修改失败，已回滚。失败于: {mod.get('field', 'unknown')}, 错误: {e}"}

    # ── 4. 验证 ──
    failed = []
    for mod in modifications:
        try:
            result = _ssh_exec(f"grep '{mod['field']}' {remote_path}", timeout=10)
            if str(mod["new_value"]) not in result:
                failed.append(mod["field"])
        except Exception:
            failed.append(mod["field"])

    if failed:
        # 回滚
        _ssh_exec(f"cat > {remote_path} < {snapshot_path}", timeout=10)
        return {"error": f"验证失败: {failed}，已回滚"}

    # ── 5. 返回结果 ──
    return {
        "status": "completed",
        "summary": f"修改完成: 备份到 {snapshot_path}, 修改了 {len(modifications)} 个权重",
        "snapshot_path": snapshot_path,
        "modified_fields": [mod["field"] for mod in modifications],
    }