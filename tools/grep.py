import subprocess

TOOL = {
    "name": "grep",
    "description": "Search for a pattern in local files using grep. Returns matching lines with line numbers. Output capped at 200 lines.",
    "parameters": {
        "pattern": {"type": "str", "required": True,  "desc": "search pattern (regex supported)"},
        "path":    {"type": "str", "required": True,  "desc": "file or directory to search"},
        "include": {"type": "str", "required": False, "desc": "file filter glob, e.g. '*.py'"},
        "ignore_case": {"type": "bool", "required": False, "desc": "case-insensitive search, default false"},
    }
}

MAX_LINES = 200

def execute(pattern: str, path: str, include: str = None, ignore_case: bool = False) -> dict:
    opts = ["-rn"]
    if ignore_case:
        opts.append("-i")
    if include:
        opts += ["--include", include]
    # Run via bash -c so WSL grep is used on Windows (no native grep in Win PATH)
    parts = ["grep"] + opts + [pattern, path]
    cmd = " ".join(f'"{p}"' if " " in p else p for p in parts)
    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True, text=True, timeout=15, encoding="utf-8", errors="replace"
        )
        lines = result.stdout.splitlines()
        truncated = len(lines) > MAX_LINES
        output = "\n".join(lines[:MAX_LINES])
        if truncated:
            output += f"\n...[{len(lines) - MAX_LINES} more lines truncated, narrow your search]..."
        return {
            "output":    output,
            "matches":   min(len(lines), MAX_LINES),
            "truncated": truncated,
            "error":     result.stderr.strip() if result.returncode > 1 else None
        }
    except FileNotFoundError:
        return {"output": "", "matches": 0, "error": "grep not found on this system"}
    except subprocess.TimeoutExpired:
        return {"output": "", "matches": 0, "error": "grep timed out (15s)"}
    except Exception as e:
        return {"output": "", "matches": 0, "error": str(e)}
