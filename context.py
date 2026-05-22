# ── Settings ─────────────────────────────────────────────────────────────────
KEEP_RECENT_TOOLS  = 80       # MicroCompact: how many tool messages to keep verbatim
MAX_CONTEXT_CHARS  = 600_000   # AutoCompact trigger (chars of messages[1:])
MAX_COMPACT_RETRIES = 3       # circuit breaker: skip compact after N consecutive failures
COMPACT_MARKER     = "[AUTOCOMPACT_SUMMARY]"

_compact_fail_count = 0       # circuit breaker state

# ── Monitoring counters (per-context) ────────────────────────────────────────
_micro_compact_count: dict[str, int] = {}
_auto_compact_count: dict[str, int] = {}
_emergency_snip_count: dict[str, int] = {}

def _ctx_key(messages: list) -> str:
    """Derive a context key from the system prompt first line."""
    if messages and messages[0].get("role") == "system":
        first_line = messages[0].get("content", "").split("\n")[0][:60]
        return first_line
    return "default"

def get_context_stats(ctx_name: str = "default") -> dict:
    """Return context compression statistics for a given context."""
    return {
        "micro_compact_count": _micro_compact_count.get(ctx_name, 0),
        "auto_compact_count": _auto_compact_count.get(ctx_name, 0),
        "emergency_snip_count": _emergency_snip_count.get(ctx_name, 0),
    }

def get_context_size(messages: list) -> dict:
    """Return current context size info."""
    total_chars = sum(len(str(m.get("content", ""))) for m in messages[1:])
    total_msgs = len(messages)
    tool_msgs = sum(1 for m in messages if m.get("role") == "tool")
    return {
        "total_chars": total_chars,
        "max_chars": MAX_CONTEXT_CHARS,
        "total_messages": total_msgs,
        "tool_messages": tool_msgs,
    }

SUMMARY_PROMPT = SUMMARY_PROMPT = """你是一个上下文压缩助手。请对以下 Agent 对话历史生成结构化摘要。

必须保留（逐条列出，使用精确数值，不得省略）：
1. 原始任务目标
2. 已完成的操作列表（含具体命令、文件路径、参数值）
3. 当前最优 reward 值及对应的参数配置
4. 已尝试过的参数组合（避免重复）
5. 遇到的错误及已试过的修复方案
6. 环境状态（SSH 连接情况、训练进程 PID、日志路径）
7. 下一步计划

以下内容必须保留原文，不得摘要化：
- 所有以 [BEHAVIOR] 开头的段落
- 所有以 [GAP] 开头的段落
- 所有以 [ROOT_CAUSE] 开头的段落
- 所有以 [ACTION] 开头的段落
- 惩罚结构的精确数值（各惩罚项名称、占比%、正负比）
- 偏相关矩阵的关键值（partial_r 和 p 值）

格式：结构化中文列表，不超过 2000 字。不要写叙述性段落。"""


# ── Tier 1: MicroCompact ─────────────────────────────────────────────────────
def micro_compact(messages: list) -> list:
    """
    Zero-LLM-cost: replace content of old tool messages with a stub.
    In-place modification so messages and contexts[ctx] stay the same list.
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) > KEEP_RECENT_TOOLS:
        key = _ctx_key(messages)
        _micro_compact_count[key] = _micro_compact_count.get(key, 0) + 1
        for i in tool_indices[:-KEEP_RECENT_TOOLS]:
            messages[i]["content"] = "[old result cleared]"
    return messages


# ── Tier 2: AutoCompact ──────────────────────────────────────────────────────
def auto_compact_if_needed(messages: list, client, model: str) -> list:
    """
    LLM-based summarisation. Circuit breaker skips after MAX_COMPACT_RETRIES
    consecutive failures to avoid burning API budget in a broken state.
    In-place modification so messages and contexts[ctx] stay the same list.
    """
    global _compact_fail_count

    total_chars = sum(len(str(m.get("content", ""))) for m in messages[1:])
    if total_chars < MAX_CONTEXT_CHARS:
        return messages

    if _compact_fail_count >= MAX_COMPACT_RETRIES:
        print(f"[context] AutoCompact circuit open ({_compact_fail_count} failures), skipping")
        return messages

    print(f"[context] chars={total_chars}, triggering AutoCompact")
    history = "\n".join(
        f"[{m.get('role','?')}] {m.get('content','')}" for m in messages[1:]
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": SUMMARY_PROMPT + "\n\n" + history}],
        )
        summary = resp.choices[0].message.content
        _compact_fail_count = 0
        key = _ctx_key(messages)
        _auto_compact_count[key] = _auto_compact_count.get(key, 0) + 1
        print("[context] AutoCompact done")
        # In-place: keep system, replace rest with summary
        del messages[1:]
        messages.append({
            "role": "user",
            "content": f"{COMPACT_MARKER}\n{summary}\n\n请继续执行未完成的步骤。",
        })
        return messages
    except Exception as e:
        _compact_fail_count += 1
        print(f"[context] AutoCompact failed ({_compact_fail_count}/{MAX_COMPACT_RETRIES}): {e}")
        return messages


# ── Tier 3: EmergencySnip ────────────────────────────────────────────────────
def emergency_snip(messages: list) -> list:
    """
    Hard truncation when the context is already too large to call the model.
    Keeps: system + last COMPACT_MARKER summary (if exists) + last 4 messages.
    In-place modification so messages and contexts[ctx] stay the same list.
    """
    key = _ctx_key(messages)
    _emergency_snip_count[key] = _emergency_snip_count.get(key, 0) + 1
    print("[context] EmergencySnip triggered")
    system = messages[0]
    # Find the most recent autocompact summary
    summary_msg = None
    for m in reversed(messages[1:]):
        content = m.get("content", "")
        if isinstance(content, str) and content.startswith(COMPACT_MARKER):
            summary_msg = m
            break
    tail = messages[-4:]
    del messages[:]
    messages.append(system)
    if summary_msg and summary_msg not in tail:
        messages.append(summary_msg)
    messages.extend(tail)
    return messages
