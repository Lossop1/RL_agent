TOOL = {
    "name": "think",
    "description": "Record your reasoning before taking a high-stakes action. No side effects. Use before editing reward weights or launching training.",
    "parameters": {
        "thought": {"type": "str", "required": True, "desc": "Your reasoning, analysis, or plan"}
    }
}

def execute(thought: str) -> dict:
    """No-op: records the thought in message history for audit and stuck detection."""
    return {"recorded": True, "error": None}
