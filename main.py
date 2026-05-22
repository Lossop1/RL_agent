import json
import os
import time
import sys
import threading
import tempfile
import shutil
from queue import Queue, Empty
from typing import Optional, Dict, Any, List
from openai import OpenAI, BadRequestError, RateLimitError, APIConnectionError
from tools import REGISTRY, get_openai_tools, dispatch
import runtime_state
from config_loader import (
    load_llm_config,
    load_meta_skill_prompt,
    load_style_prompt,
    load_preference_prompts,
    load_ssh_config,
)
from tools.bash import stop_shell
from tools.cmd import stop_shell as cmd_stop_shell
from tools.ssh_exec import disconnect as ssh_disconnect
from context import micro_compact, auto_compact_if_needed, emergency_snip, get_context_stats, get_context_size
from stuck import StuckDetector

# ═══════════════════════════════════════════════════════════
# 主题加载
# ═══════════════════════════════════════════════════════════

def _load_theme():
    path = "config/theme.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

_theme = _load_theme()
_colors = _theme.get("colors", {})
_symbols = _theme.get("symbols", {})

_reset = _colors.get("reset", "")

def _make_tag_style(tag_cfg: dict, default_prefix: str) -> dict:
    """从 tag 配置构建 prefix 和 style 字符串."""
    color = _colors.get(tag_cfg.get("color", ""), "")
    italic = _colors.get("italic", "") if tag_cfg.get("italic") else ""
    bold = _colors.get("bold", "") if tag_cfg.get("bold") else ""
    prefix = tag_cfg.get("prefix", default_prefix)
    return {
        "prefix": f"{italic}{bold}{color}{prefix}{_reset}",
        "style": f"{italic}{bold}{color}",
    }

_tags_cfg = _theme.get("tags", {})
TAGS = {
    tag_name: _make_tag_style(tag_cfg, f"[{tag_name}]")
    for tag_name, tag_cfg in _tags_cfg.items()
}

# 向后兼容：思考内容流式输出
THINK_TAG = TAGS.get("think", {"prefix": "[思考]", "style": ""})
THINK_PREFIX = THINK_TAG["prefix"]
THINK_STYLE = THINK_TAG["style"]

# 工具标签
TOOL_TAG = TAGS.get("tool", {"prefix": "", "style": _colors.get("gray", "")})

HR_CHAR = _symbols.get("hr_char", "-")
HR_LEN = _symbols.get("hr_length", 54)
PROMPT = _symbols.get("prompt", "  > ")
PROMPT_CONFIRM = _symbols.get("prompt_confirm", "  确认? [y/N]: ")
PROMPT_DELETE = _symbols.get("prompt_delete", "  删除哪个? ")

# 上下文显示名称
CTX_DISPLAY = {
    "meta": "meta",
    "rectify": "rectify",
    "training": "training",
    "env_setup": "env_setup",
}


def _format_tool_call(name: str, args: dict) -> str:
    """格式化工具调用的单行显示."""

    gray = _colors.get("gray", "")

    # switch_context / pop_context
    if name == "switch_context":
        target = args.get("context_name", "?")
        info = args.get("info", "")
        try:
            if isinstance(info, str) and info:
                info_obj = json.loads(info)
                task_type = info_obj.get("task_type", "")
                if task_type:
                    return f"{gray}[switch] → {target}/{task_type}{_reset}"
        except (json.JSONDecodeError, TypeError):
            pass
        return f"{gray}[switch] → {target}{_reset}"

    if name == "pop_context":
        result_str = args.get("result", "")
        try:
            if result_str:
                result_obj = json.loads(result_str)
                status = result_obj.get("status", "")
                summary = result_obj.get("summary", "")[:60]
                return f"{gray}[pop] {status}: {summary}{_reset}"
        except (json.JSONDecodeError, TypeError):
            pass
        return f"{gray}[pop]{_reset}"

    # ssh_exec — 只显示命令前 100 字符
    if name == "ssh_exec":
        cmd = args.get("command", "")
        short = cmd[:100] + ("..." if len(cmd) > 100 else "")
        return f"{gray}[ssh] {short}{_reset}"

    # 其他工具 — 只显示关键参数
    key_params = {
        "parse_training_log": "log_path",
        "deep_dig": "snapshot_path",
        "tuner": "remote_path",
        "tail_log": "action",
        "file_read": "path",
        "file_write": "path",
        "file_edit": "path",
        "grep": "pattern",
        "experience_query": "query",
        "training_status": "log_path",
    }
    key = key_params.get(name)
    if key and key in args:
        val = str(args[key])[:80]
        return f"{gray}[{name}] {key}={val}{_reset}"

    # 兜底
    all_args = json.dumps(args, ensure_ascii=False)
    short_args = all_args[:80] + ("..." if len(all_args) > 80 else "")
    return f"{gray}[{name}] {short_args}{_reset}"


# ═══════════════════════════════════════════════════════════
# 会话持久化
# ═══════════════════════════════════════════════════════════

SESSIONS_DIR = "sessions"
CONTEXTS_DIR = os.path.join(SESSIONS_DIR, "contexts")


def _session_path(session_id: str) -> str:
    return os.path.join(CONTEXTS_DIR, f"{session_id}.json")


def _save_contexts(control: dict) -> bool:
    sid = control.get("_session_id")
    if not sid:
        return False
    if control.get("current_turn", 0) <= 1:
        return False
    os.makedirs(CONTEXTS_DIR, exist_ok=True)
    payload = {
        "session_id": sid,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "contexts": control.get("contexts", {}),
        "context_stack": control.get("context_stack", []),
        "current_turn": control.get("current_turn", 0),
        "session_log_path": control.get("session_log_path", ""),
    }
    try:
        fd, tmp = tempfile.mkstemp(dir=CONTEXTS_DIR, suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        shutil.move(tmp, _session_path(sid))
        return True
    except Exception:
        return False


def _load_contexts(session_id: str) -> Optional[dict]:
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _delete_contexts(session_id: str) -> bool:
    try:
        os.remove(_session_path(session_id))
        return True
    except OSError:
        return False


def _list_sessions() -> List[dict]:
    if not os.path.isdir(CONTEXTS_DIR):
        return []
    results = []
    for fname in os.listdir(CONTEXTS_DIR):
        if not fname.endswith(".json") or fname.startswith("tmp"):
            continue
        path = os.path.join(CONTEXTS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        results.append({
            "session_id": payload.get("session_id", fname.replace(".json", "")),
            "saved_at": payload.get("saved_at", "unknown"),
            "turn": payload.get("current_turn", 0),
        })
    results.sort(key=lambda r: r.get("saved_at", ""), reverse=True)
    return results


# ═══════════════════════════════════════════════════════════
# 日志
# ═══════════════════════════════════════════════════════════

def _create_agent_trace_log_path() -> str:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(SESSIONS_DIR, f"agent_debug_{ts}.log")


def _append_agent_trace_log(log_path: str, text: str) -> None:
    if not log_path:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {text}\n")


# ═══════════════════════════════════════════════════════════
# UI 工具
# ═══════════════════════════════════════════════════════════

def _hr():
    print(f"  {HR_CHAR * HR_LEN}")


def _banner(text: str):
    print(f"\n  {text}")
    _hr()


def _session_summary(s: dict) -> str:
    """从会话上下文中提取任务摘要（第一条 user 消息前 30 字）"""
    sid = s["session_id"]
    path = _session_path(sid)
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return ""
    ctxs = payload.get("contexts", {})
    for msgs in ctxs.values():
        for m in msgs:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                text = m["content"].strip()
                if len(text) > 30:
                    return text[:30] + "..."
                return text
    return ""


def _show_resume_ui(sessions: List[dict]) -> Optional[tuple]:
    """返回 (session_id, task_text) 或 None（无会话可恢复）"""
    if not sessions:
        return None

    _banner("恢复上次会话")
    for i, s in enumerate(sessions):
        summary = _session_summary(s)
        line = f"  {i+1}. {s['session_id']}"
        line += f"\n       {s['saved_at']}  \xb7  turn {s['turn']}"
        if summary:
            line += f"  \xb7  {summary}"
        print(line)
    _hr()

    while True:
        choice = input(PROMPT).strip()
        if not choice:
            continue
        # d<数字> → 删除
        if len(choice) > 1 and choice[0].lower() == "d" and choice[1:].isdigit():
            n = int(choice[1:])
            if 1 <= n <= len(sessions):
                _delete_contexts(sessions[n - 1]["session_id"])
                print(f"  已删除 {sessions[n-1]['session_id']}")
                sessions.pop(n - 1)
                if not sessions:
                    print("  所有会话已删除.")
                    return None
                # 刷新显示
                print()
                for j, s in enumerate(sessions):
                    summary = _session_summary(s)
                    line = f"  {j+1}. {s['session_id']}"
                    line += f"\n       {s['saved_at']}  \xb7  turn {s['turn']}"
                    if summary:
                        line += f"  \xb7  {summary}"
                    print(line)
                _hr()
                continue
            print(f"  无效编号: {n}")
            continue
        # 数字 → 恢复
        if choice.isdigit():
            n = int(choice)
            if 1 <= n <= len(sessions):
                return (sessions[n - 1]["session_id"], None)
            print(f"  无效编号: {n}")
            continue
        # 其他 → 新建会话，内容作为任务
        return (None, choice)


# ═══════════════════════════════════════════════════════════
# 控制状态
# ═══════════════════════════════════════════════════════════

def _new_control_state() -> Dict[str, Any]:
    return {
        "running": threading.Event(),
        "stop_requested": threading.Event(),
        "pending_user_inputs": Queue(),
        "current_turn": 0,
        "max_turns": 0,
        "task": "",
        "session_log_path": "",
        "_pending_confirm": None,
        "_confirm_result": False,
        "_session_id": None,
        "contexts": {},
        "context_stack": [],
    }


# ═══════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════

def _drain_pending_user_inputs(control: Dict[str, Any]) -> List[str]:
    pending = []
    q = control["pending_user_inputs"]
    while True:
        try:
            pending.append(q.get_nowait())
        except Empty:
            break
    return pending


def _build_system_prompt() -> str:
    meta = load_meta_skill_prompt()
    if not meta:
        meta = "你是一个自主强化学习训练助手。"
    style = load_style_prompt() or ""
    prefs = load_preference_prompts() or ""
    return "\n\n".join(filter(None, [meta, style, prefs]))


def _resolve_api_key(llm: dict) -> str:
    api_key = llm.get("api_key")
    if api_key:
        return api_key
    legacy = llm.get("api_key_env_var")
    if legacy:
        return os.environ.get(legacy, legacy)
    raise KeyError("llm.api_key missing")


def _needs_confirmation(name: str, args: dict) -> bool:
    path = args.get("path", "")
    if name == "file_write":
        return not path.startswith(("sessions/", "prompts/"))
    if name == "file_edit":
        return not path.startswith("sessions/")
    if name == "ssh_exec":
        cmd = args.get("command", "")
        if any(rp in cmd for rp in ["--version", "--help", "which ", "ls ", "cat ", "head ", "tail ", "grep ", "find "]):
            return False
        return any(d in cmd for d in ["sed -i", "rm ", "mv ", "cp ", "chmod", "chown", "apt install", "apt-get install", "pip install", "> ", ">>"])
    return False


def _prompt_for_confirmation(name: str, args: dict, control: dict = None) -> bool:
    print(f"\n  [确认] {name}")
    if name == "ssh_exec":
        cmd = args.get("command", "")
        print(f"  [确认] 命令: {cmd[:200]}")
    elif name == "file_write":
        print(f"  [确认] 写入: {args.get('path', '')}")
    elif name == "file_edit":
        print(f"  [确认] 编辑: {args.get('path', '')}")
    else:
        print(f"  [确认] {json.dumps(args, ensure_ascii=False)}")
    if control is None:
        print("  输入 y 确认 / n 拒绝")
        while True:
            try:
                resp = input("  [确认] 执行? (y/N): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
            if resp in ("y", "yes"):
                return True
            if resp in ("", "n", "no"):
                return False

    control["_pending_confirm"] = {"name": name, "args": args}
    while control["_pending_confirm"] is not None:
        time.sleep(0.1)
        if control.get("stop_requested", threading.Event()).is_set():
            return False
    return control.get("_confirm_result", False)


def _print_interactive_help() -> None:
    print("  /exit       退出")
    print("  /help       显示帮助")
    print("  直接输入     发送指令给 agent")
    print()


# ═══════════════════════════════════════════════════════════
# 交互循环
# ═══════════════════════════════════════════════════════════

def _interactive_loop() -> None:
    print()
    _banner("RL-Dog Training Agent")

    control = _new_control_state()

    sessions = _list_sessions()
    result = _show_resume_ui(sessions)

    if result is None:
        pass
    else:
        resume_sid, task_text = result
        if resume_sid:
            saved = _load_contexts(resume_sid)
            if saved:
                control["_session_id"] = saved["session_id"]
                control["_resume_saved_at"] = saved.get("saved_at", "unknown")
                control["contexts"] = saved.get("contexts", {})
                control["context_stack"] = saved.get("context_stack", ["meta"])
                control["session_log_path"] = saved.get("session_log_path", "")
                print(f"\n  已恢复 {resume_sid}.\n")
                worker = threading.Thread(target=_run_agent_worker, args=("__RESUME__", control), daemon=True)
                worker.start()
                _interactive_input_loop(control, worker)
                return
            else:
                print(f"\n  加载 {resume_sid} 失败. 开始新会话.\n")
        elif task_text:
            sid = f"S{time.strftime('%Y-%m-%d_%H-%M-%S')}"
            control["_session_id"] = sid
            control["session_log_path"] = os.path.join(SESSIONS_DIR, f"trace_{time.strftime('%Y%m%d_%H%M%S')}.log")
            os.makedirs(SESSIONS_DIR, exist_ok=True)
            print(f"\n  会话: {sid}")
            print(f"  日志: {control['session_log_path']}")
            _hr()
            print()
            worker = threading.Thread(target=_run_agent_worker, args=(task_text, control), daemon=True)
            worker.start()
            _interactive_input_loop(control, worker)
            return

    sid = f"S{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    control["_session_id"] = sid
    control["session_log_path"] = os.path.join(SESSIONS_DIR, f"trace_{time.strftime('%Y%m%d_%H%M%S')}.log")
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    print(f"\n  会话: {sid}")
    print(f"  日志: {control['session_log_path']}")
    _hr()
    print()
    _print_interactive_help()
    _interactive_input_loop(control, None)


def _interactive_input_loop(control: dict, worker: Optional[threading.Thread]) -> None:
    while True:
        try:
            user_input = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            _save_contexts(control)
            print("\n  再见.")
            break

        if not user_input:
            continue

        if user_input == "/help":
            _print_interactive_help()
            continue

        if user_input == "/exit":
            if control["running"].is_set():
                control["stop_requested"].set()
                if worker is not None:
                    worker.join(timeout=30)
            _save_contexts(control)
            print("  再见.")
            break

        if user_input.startswith("/"):
            print(f"  未知命令: {user_input}")
            continue

        if control["running"].is_set():
            pending = control.get("_pending_confirm")
            if pending is not None:
                resp = user_input.strip().lower()
                if resp in ("y", "yes"):
                    control["_confirm_result"] = True
                    control["_pending_confirm"] = None
                else:
                    control["_confirm_result"] = False
                    control["_pending_confirm"] = None
                continue
            control["pending_user_inputs"].put(user_input)
            continue

        worker = threading.Thread(target=_run_agent_worker, args=(user_input, control), daemon=True)
        worker.start()


def _run_agent_worker(task_description: str, control: Dict[str, Any]) -> None:
    control["running"].set()
    control["stop_requested"].clear()
    control["task"] = task_description
    try:
        run_agent(task_description, control=control)
    except Exception as e:
        print(f"\n  错误: {e}")
    finally:
        control["running"].clear()


# ═══════════════════════════════════════════════════════════
# Agent 循环
# ═══════════════════════════════════════════════════════════

def run_agent(task_description: str, control: Optional[Dict[str, Any]] = None) -> None:
    is_resume = (task_description == "__RESUME__")

    ssh_cfg, ssh_cfg_path = load_ssh_config()
    runtime_state.set_config(ssh_cfg, ssh_cfg_path)
    llm = load_llm_config()
    api_key = _resolve_api_key(llm)
    client = OpenAI(api_key=api_key, base_url=llm["base_url"])
    model = llm["model"]
    max_turns = llm.get("max_turns", 300)

    if control is not None:
        debug_log_path = control.get("session_log_path") or ""
        control["current_turn"] = control.get("current_turn", 0)
        control["max_turns"] = max_turns
    else:
        debug_log_path = _create_agent_trace_log_path()

    if debug_log_path:
        _append_agent_trace_log(debug_log_path, f"TASK_START: {task_description}")

    if not is_resume:
        system_prompt = _build_system_prompt()
        contexts = {"meta": [{"role": "system", "content": system_prompt}]}
        if task_description:
            contexts["meta"].append({"role": "user", "content": task_description})
        if control is not None:
            control["contexts"] = contexts
            control["context_stack"] = ["meta"]
    else:
        contexts = control.get("contexts", {}) if control else {}
        stack = control.get("context_stack", ["meta"]) if control else ["meta"]
        active_ctx = stack[-1]
        saved_at = control.get("_resume_saved_at", "unknown") if control else "unknown"

        for ctx_name, msgs in contexts.items():
            if msgs and msgs[-1].get("tool_calls"):
                for tc in msgs[-1]["tool_calls"]:
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps({"error": "sleep interrupted by session restart"}, ensure_ascii=False),
                    })

        if active_ctx in contexts:
            contexts[active_ctx].append({
                "role": "user",
                "content": (
                    f"[系统] 会话已恢复。\n"
                    f"- 上次保存时间：{saved_at}\n"
                    f"- 当前活跃上下文：{active_ctx}\n"
                    f"- 上下文栈：{' > '.join(stack)}\n"
                    f"- 请检查最近的对话历史，确认当前状态后继续执行未完成的步骤。"
                )
            })
        if control is not None:
            control["context_stack"] = stack

    active_ctx = control["context_stack"][-1] if control and control["context_stack"] else "meta"
    messages = contexts[active_ctx]

    CONTEXT_TOOLS = {
        "meta": ["task_complete", "switch_context", "pop_context"],
        "rectify": [
            "file_read", "parse_training_log", "deep_dig", "tuner",
            "think", "switch_context", "pop_context",
            "ssh_exec", "cmd", "bash", "file_write",
            "grep", "glob_util", "experience_query", "experience_write",
            "training_status", "file_edit",
        ],
        "training": [
            "file_read", "file_write", "ssh_exec", "cmd", "tail_log",
            "think", "switch_context", "pop_context",
        ],
        "env_setup": [
            "file_read", "file_write", "ssh_exec", "cmd",
            "switch_context", "pop_context",
        ],
    }
    all_tools = get_openai_tools()
    allowed_names = CONTEXT_TOOLS.get(active_ctx, list(REGISTRY.keys()))
    all_tools = [t for t in all_tools if t["function"]["name"] in allowed_names]
    stuck = StuckDetector()
    turn = control.get("current_turn", 0) if control else 0

    try:
        while turn < max_turns:
            turn += 1
            if control is not None:
                control["current_turn"] = turn

            if control is not None:
                for extra in _drain_pending_user_inputs(control):
                    messages.append({"role": "user", "content": extra})

            ctx_size = get_context_size(messages)
            ctx_display = CTX_DISPLAY.get(active_ctx, active_ctx)
            print(f"\n  Turn {turn}/{max_turns}  \xb7  {ctx_display}  \xb7  {ctx_size['total_chars']}/{ctx_size['max_chars']} chars")

            _append_agent_trace_log(debug_log_path, f"TURN_START: {turn}/{max_turns} ctx={active_ctx}")

            messages = micro_compact(messages)
            messages = auto_compact_if_needed(messages, client, model)

            is_stuck, scenario = stuck.check()
            if is_stuck:
                messages.append(stuck.intervention_message(scenario))
                stuck.reset()

            stream = None
            for retry in range(4):
                try:
                    request = {
                        "model": model,
                        "messages": messages,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "tools": all_tools,
                        "tool_choice": "auto",
                    }
                    stream = client.chat.completions.create(**request)
                    break
                except BadRequestError as e:
                    if any(kw in str(e).lower() for kw in ("context", "too long", "token", "length")):
                        messages = emergency_snip(messages)
                        continue
                    error_tag = TAGS.get("error", {}).get("prefix", "[ERROR]")
                    print(f"\n{error_tag} API 错误: {e}")
                    _append_agent_trace_log(debug_log_path, f"API_ERROR: {e}")
                    return
                except RateLimitError:
                    time.sleep(10)
                    continue
                except APIConnectionError as e:
                    _append_agent_trace_log(debug_log_path, f"CONNECTION_ERROR: {e}")
                    time.sleep(5)
                    continue
                except Exception as e:
                    _append_agent_trace_log(debug_log_path, f"API_ERROR: {e}")
                    return

            if stream is None:
                return

            content_parts: List[str] = []
            reasoning_parts: List[str] = []
            tool_call_deltas: Dict[int, dict] = {}

            for chunk in stream:
                if chunk.usage:
                    pass
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    if not reasoning_parts:
                        print(f"\n{THINK_PREFIX}")
                    reasoning_parts.append(delta.reasoning_content)
                    print(f"{THINK_STYLE}{delta.reasoning_content}{_reset}", end="", flush=True)
                if delta.content:
                    if reasoning_parts and not content_parts:
                        print()
                    content_parts.append(delta.content)
                    print(delta.content, end="", flush=True)
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_deltas:
                            tool_call_deltas[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                        if tc_delta.id:
                            tool_call_deltas[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_call_deltas[idx]["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_call_deltas[idx]["function"]["arguments"] += tc_delta.function.arguments

            if content_parts:
                print()

            full_content = "".join(content_parts) or None
            full_reasoning = "".join(reasoning_parts) or None

            msg_dict: dict = {"role": "assistant", "content": full_content}
            if tool_call_deltas:
                reconstructed = []
                for idx in sorted(tool_call_deltas.keys()):
                    d = tool_call_deltas[idx]
                    reconstructed.append({
                        "id": d["id"],
                        "type": "function",
                        "function": {"name": d["function"]["name"], "arguments": d["function"]["arguments"]},
                    })
                msg_dict["tool_calls"] = reconstructed
            if full_reasoning:
                msg_dict["reasoning_content"] = full_reasoning
            messages.append(msg_dict)

            if not tool_call_deltas:
                _append_agent_trace_log(debug_log_path, f"NO_TOOL_CALLS ctx={active_ctx}")
                if control is not None and len(control["context_stack"]) > 1:
                    control["context_stack"].pop()
                    active_ctx = control["context_stack"][-1]
                    messages = contexts[active_ctx]
                continue

            for tc in msg_dict.get("tool_calls", []):
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}

                _append_agent_trace_log(debug_log_path, f"TOOL_CALL: {name}")

                if _needs_confirmation(name, args):
                    if not _prompt_for_confirmation(name, args, control):
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps({"error": "rejected by user"}, ensure_ascii=False),
                        })
                        continue

                # 格式化工具调用显示
                tool_line = _format_tool_call(name, args)
                print(f"\n  {tool_line}")

                ctx_before = active_ctx
                result = dispatch(name, args, _control=control)

                if name == "switch_context":
                    if control is not None:
                        active_ctx = control["context_stack"][-1]
                        messages = contexts[active_ctx]
                        allowed_names = CONTEXT_TOOLS.get(active_ctx, list(REGISTRY.keys()))
                        all_tools = [t for t in get_openai_tools() if t["function"]["name"] in allowed_names]
                    _append_agent_trace_log(debug_log_path, f"CTX_SWITCH: to {args.get('context_name', '')}")
                if name == "pop_context":
                    child_ctx = active_ctx
                    if control is not None and len(control["context_stack"]) >= 1:
                        active_ctx = control["context_stack"][-1]
                        messages = contexts[active_ctx]
                        # 更新工具列表
                        allowed_names = CONTEXT_TOOLS.get(active_ctx, list(REGISTRY.keys()))
                        all_tools = [t for t in get_openai_tools() if t["function"]["name"] in allowed_names]
                    _append_agent_trace_log(debug_log_path, f"CTX_POP: back to {active_ctx}")

                if name == "task_complete":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    _save_contexts(control)
                    return

                _append_agent_trace_log(debug_log_path, f"TOOL_RESULT: {name}")
                stuck.record(name, args, result)

                if name == "switch_context":
                    contexts[ctx_before].append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    continue

                if name == "pop_context":
                    contexts[child_ctx].append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    contexts[active_ctx].append({
                        "role": "assistant",
                        "content": f"[来自 {child_ctx}] 任务完成: {result.get('result', '')}",
                    })
                    continue

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })

            if control is not None:
                _save_contexts(control)

        done_tag = TAGS.get("done", {}).get("prefix", "[DONE]")
        print(f"\n{done_tag} 达到最大轮数 ({max_turns}).")
        _append_agent_trace_log(debug_log_path, f"MAX_TURNS: {max_turns}")

    finally:
        stop_shell()
        cmd_stop_shell()
        ssh_disconnect()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:]).strip()
        control = _new_control_state()
        control["_session_id"] = f"S{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        _run_agent_worker(task, control)
    else:
        _interactive_loop()