TOOL = {
    "name": "think",
    "description": "Record your reasoning before taking a high-stakes action. No side effects. Use before editing reward weights or launching training.",
    "parameters": {
        "thought": {"type": "str", "required": True, "desc": "Your reasoning, analysis, or plan"}
    }
}

def execute(thought: str, _control: dict = None) -> dict:
    """No-op: records the thought in message history for audit and stuck detection."""
    result = {"recorded": True, "error": None}

    if _control:
        stack = _control.get("context_stack", [])
        if stack and stack[-1] in ("rebuttal", "data_parse"):
            result["context_reminder"] = (
                f"你当前在 [{stack[-1]}] 上下文中。"
                "你必须调用 pop_context 才能返回。"
                '口头说"任务完成"不会让你离开当前上下文。'
            )

    return result
