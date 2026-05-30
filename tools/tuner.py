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

    # ── 0. Rebuttal 闸门 ──
    if _control and _control.get("_pending_rebuttal"):
        return {"error": "诊断后必须先 switch_context 到 rebuttal 验证。tuner 被拒绝执行。"}

    # ── 1. 权限检查 ──
    try:
        _check_permission(remote_path, modifications, context_stack)
    except PermissionError as e:
        return {"error": str(e)}

    # ── 1. 杀训练进程 ──
    try:
        _ssh_exec("pkill -f 'train.py' 2>/dev/null; echo done", timeout=10)
    except Exception as e:
        return {"error": f"终止训练进程失败: {e}"}

    # ── 2. 备份（本地 + 远程） ──
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

    # 远程备份（用于回滚，避免本地路径在远程不可用）
    remote_bak = f"{remote_path}.bak"
    _ssh_exec(f"cp {remote_path} {remote_bak}", timeout=10)

    # ── 3. 逐条修改 ──
    changed = []
    try:
        for mod in modifications:
            field = mod["field"].replace(".", "\\.")
            # 处理科学计数法格式差异：-2.975e-07 vs -2.975e-7
            old_raw = str(mod["old_value"])
            new_raw = str(mod["new_value"])
            # 生成两种格式：e-07 和 e-7
            old_variants = [old_raw]
            if "e-" in old_raw:
                # 去掉指数补零
                old_variants.append(re.sub(r"e-0(\d+)$", r"e-\1", old_raw))
                # 补零到3位指数
                old_variants.append(re.sub(r"e-(\d+)$", lambda m: f"e-{int(m.group(1)):03d}", old_raw))
            new_variants = [new_raw]
            if "e-" in new_raw:
                new_variants.append(re.sub(r"e-0(\d+)$", r"e-\1", new_raw))
                new_variants.append(re.sub(r"e-(\d+)$", lambda m: f"e-{int(m.group(1)):03d}", new_raw))
            # 去重
            old_variants = list(dict.fromkeys(old_variants))
            new_variants = list(dict.fromkeys(new_variants))
            # 尝试每种 old 格式匹配
            matched = False
            for ov in old_variants:
                escaped_old = ov.replace("-", "\\-").replace(".", "\\.")
                for nv in new_variants:
                    escaped_new = nv.replace("-", "\\-").replace(".", "\\.")
                    sed_cmd = f"sed -i 's/{field} = {escaped_old}/{field} = {escaped_new}/' {remote_path}"
                    _ssh_exec(sed_cmd, timeout=10)
                    # 验证是否真的改了
                    check = _ssh_exec(f"grep '{mod['field']}' {remote_path}", timeout=10)
                    if nv in check:
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                raise ValueError(f"无法匹配字段 {mod['field']} 的旧值 {old_raw}")
            changed.append(mod)
    except Exception as e:
        # 回滚（用远程备份）
        _ssh_exec(f"cp {remote_bak} {remote_path}", timeout=10)
        _ssh_exec(f"rm -f {remote_bak}", timeout=10)
        return {"error": f"修改失败，已回滚。失败于: {mod.get('field', 'unknown')}, 错误: {e}"}

    # ── 4. 验证 ──
    failed = []
    for mod in modifications:
        try:
            result = _ssh_exec(f"grep '{mod['field']}' {remote_path}", timeout=10)
            new_raw = str(mod["new_value"])
            if new_raw not in result:
                # 也检查补零格式
                alt_new = re.sub(r"e-0(\d+)$", r"e-\1", new_raw) if "e-" in new_raw else None
                if alt_new and alt_new not in result:
                    failed.append(mod["field"])
                elif not alt_new:
                    failed.append(mod["field"])
        except Exception:
            failed.append(mod["field"])

    if failed:
        # 回滚（用远程备份）
        _ssh_exec(f"cp {remote_bak} {remote_path}", timeout=10)
        _ssh_exec(f"rm -f {remote_bak}", timeout=10)
        return {"error": f"验证失败: {failed}，已回滚"}

    # 清理远程备份
    _ssh_exec(f"rm -f {remote_bak}", timeout=10)

    # ── 5. 返回结果 ──
    return {
        "status": "completed",
        "summary": f"修改完成: 备份到 {snapshot_path}, 修改了 {len(modifications)} 个权重",
        "snapshot_path": snapshot_path,
        "modified_fields": [mod["field"] for mod in modifications],
    }