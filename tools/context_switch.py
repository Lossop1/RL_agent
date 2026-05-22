"""Context switching tools for multi-agent support.

These tools manage the context stack: switch_context pushes a new context
onto the stack, pop_context pops back to the previous context.
"""

import json
import os

# 合法的子 agent 上下文名称
VALID_CONTEXTS = {"meta", "rectify", "training", "env_setup"}

TOOL = {
    "name": "switch_context",
    "description": (
        "Switch to a different skill context. The current context is preserved "
        "and execution continues in the new context. Use this when a skill "
        "needs to delegate to another skill. "
        "The new context starts fresh with only the info passed in."
    ),
    "parameters": {
        "context_name": {
            "type": "str", "required": True,
            "desc": "Target context name, e.g. 'training', 'env_setup'"
        },
        "info": {
            "type": "str", "required": False,
            "desc": "JSON object with information to pass to the new context"
        },
    }
}


def execute(context_name: str, info: str = "{}", _control: dict = None) -> dict:
    """Switch to a new context, pushing the current one onto the stack."""
    if _control is None:
        return {"error": "switch_context requires _control (internal error)"}

    if not context_name:
        return {"error": "context_name is required"}

    if context_name not in VALID_CONTEXTS:
        return {"error": f"unknown context: {context_name}. 可用: {', '.join(sorted(VALID_CONTEXTS))}"}

    contexts = _control.get("contexts", {})
    context_stack = _control.get("context_stack", [])

    # Always push to stack (even if context already exists)
    context_stack.append(context_name)

    if context_name not in contexts:
        # Load skill file as system prompt
        skill_path = f"prompts/skills/{context_name}.md"
        skill_content = ""
        try:
            with open(skill_path, "r", encoding="utf-8") as f:
                skill_content = f.read()
        except FileNotFoundError:
            pass

        # Load rules for this context
        rules_parts = []
        general_rules_path = "prompts/rules/general.md"
        try:
            with open(general_rules_path, "r", encoding="utf-8") as f:
                rules_parts.append(f.read())
        except FileNotFoundError:
            pass

        context_rules_path = f"prompts/rules/{context_name}.md"
        try:
            with open(context_rules_path, "r", encoding="utf-8") as f:
                rules_parts.append(f.read())
        except FileNotFoundError:
            pass

        rules_text = "\n\n".join(rules_parts)
        system_text = skill_content or f"你是 {context_name} 助手。"
        if rules_text:
            system_text += "\n\n" + rules_text
        contexts[context_name] = [
            {"role": "system", "content": system_text},
        ]

    # Append info as new user message (even if context already exists)
    if info:
        if isinstance(info, str):
            try:
                info = json.loads(info)
            except json.JSONDecodeError:
                pass
        contexts[context_name].append(
            {"role": "user", "content": f"[来自 meta] 任务参数：{json.dumps(info, ensure_ascii=False)}"}
        )

    return {"stdout": f"switched to ctx={context_name}", "stderr": ""}


TOOL_POP = {
    "name": "pop_context",
    "description": (
        "Return to the previous context. Use this when the current skill "
        "has completed its task and wants to go back to the caller."
    ),
    "parameters": {
        "result": {
            "type": "str", "required": False,
            "desc": "JSON string with result data to pass back to the caller"
        },
    },
}


def execute_pop(result: str = "", _control: dict = None) -> dict:
    """Pop back to the previous context."""
    if _control is None:
        return {"error": "pop_context requires _control (internal error)"}

    context_stack = _control.get("context_stack", [])
    contexts = _control.get("contexts", {})

    if len(context_stack) <= 1:
        return {"error": "already at root context, cannot pop"}

    context_stack.pop()
    active_ctx = context_stack[-1]

    # Pass the child skill's result back to the caller
    return {"stdout": f"popped back to ctx={active_ctx}", "stderr": "", "result": result}