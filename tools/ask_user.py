"""
ask_user.py — Ask the user a yes/no or choice question and wait for input.

Blocks until the user responds. Returns the user's answer as a string.
"""
import json

TOOL = {
    "name": "ask_user",
    "description": (
        "Ask the user a question and wait for their response. "
        "Use this when you need user confirmation or a decision before proceeding. "
        "The question should be a clear yes/no or multiple choice."
    ),
    "parameters": {
        "question": {
            "type": "str",
            "required": True,
            "desc": "The question to ask the user"
        },
        "options": {
            "type": "str",
            "required": False,
            "desc": "Optional comma-separated list of valid options (e.g. 'yes,no' or 'continue,restart,abort')"
        },
    }
}


def execute(question: str, options: str = "") -> dict:
    print(f"\n  [ASK] {question}")
    if options:
        print(f"  选项: {options}")
    while True:
        try:
            resp = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            return {"response": "", "error": "input interrupted"}
        if not resp:
            continue
        if options:
            valid = [o.strip().lower() for o in options.split(",")]
            if resp.lower() in valid:
                return {"response": resp}
            print(f"  无效输入，请输入以下选项之一: {options}")
            continue
        return {"response": resp}
