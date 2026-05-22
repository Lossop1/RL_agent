import glob as _glob
import os

TOOL = {
    "name": "glob",
    "description": "Find local files matching a pattern. Returns up to 200 results sorted by modification time (newest first).",
    "parameters": {
        "pattern": {"type": "str", "required": True,  "desc": "glob pattern, e.g. '/workspace/**/*.py' or '*.json'"},
        "root":    {"type": "str", "required": False, "desc": "base directory to search from (optional, defaults to cwd)"},
    }
}

MAX_RESULTS = 200

def execute(pattern: str, root: str = None) -> dict:
    try:
        original_cwd = os.getcwd()
        if root:
            os.chdir(root)
        matches = _glob.glob(pattern, recursive=True)
        if root:
            os.chdir(original_cwd)
        # Sort newest-first (most relevant for checkpoint/log discovery)
        matches.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
        truncated = len(matches) > MAX_RESULTS
        return {
            "matches":   matches[:MAX_RESULTS],
            "count":     len(matches),
            "truncated": truncated,
            "error":     None,
        }
    except Exception as e:
        return {"matches": [], "count": 0, "error": str(e)}
