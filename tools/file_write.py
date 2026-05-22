import os

TOOL = {
    "name": "file_write",
    "description": "Write content to a local file, creating parent directories if needed.",
    "parameters": {
        "path":    {"type": "str", "required": True, "desc": "absolute path to write"},
        "content": {"type": "str", "required": True, "desc": "content to write"},
    }
}

def execute(path: str, content: str) -> dict:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"written": path, "bytes": len(content.encode()), "error": None}
    except Exception as e:
        return {"error": str(e)}