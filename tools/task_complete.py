"""Task completion signal tool.

When the agent (meta) determines a task is finished, it calls this tool.
main.py intercepts this tool call and returns to the interactive loop.
"""

TOOL = {
    "name": "task_complete",
    "description": (
        "声明当前任务已完成。调用此工具后，系统会等待用户输入下一个任务。"
        "当子 skill 返回的结果表明任务已完成时，调用此工具。"
    ),
    "parameters": {
        "summary": {
            "type": "str",
            "required": True,
            "desc": "任务完成摘要，例如'训练 go2 完成: 100/100 轮, reward=250.0'",
        },
    },
}


def execute(summary: str = "") -> dict:
    """Signal task completion. The actual blocking happens in main.py."""
    return {"status": "ok", "summary": summary}
