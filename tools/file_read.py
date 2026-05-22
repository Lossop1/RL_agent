from pathlib import Path

TOOL = {
    "name": "file_read",
    "description": "Read local file content (absolute path or workspace-relative path).",
    "parameters": {
        "path": {"type": "str", "required": True, "desc": "absolute path or workspace-relative path"},
        "start_line": {"type": "int", "required": False, "desc": "start line (1-indexed), default 1"},
        "end_line": {"type": "int", "required": False, "desc": "end line, default all"},
    }
}

MAX_READ_CHARS = 30000
_BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_path(path: str) -> Path:
    """Resolve flexible input path to a concrete local file path."""
    raw = str(path or "").strip().strip('"').strip("'")
    if not raw:
        return Path(raw)

    p = Path(raw)
    if p.is_absolute():
        return p

    # Prefer current process cwd, then project root.
    from_cwd = Path.cwd() / p
    if from_cwd.exists():
        return from_cwd
    return _BASE_DIR / p


def execute(path, start_line=None, end_line=None):
    resolved = _resolve_path(path)

    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return {"content": "", "resolved_path": str(resolved), "error": str(e)}

    start = max((start_line or 1) - 1, 0)
    end = end_line if end_line is not None else len(lines)
    end = min(max(end, 0), len(lines))
    lines = lines[start:end]

    offset = start + 1
    numbered = [f"{offset + i}: {line}" for i, line in enumerate(lines)]
    content = "".join(numbered)

    if len(content) > MAX_READ_CHARS:
        content = content[:MAX_READ_CHARS] + "\n...[TRUNCATED]...\n"

    return {"content": content, "resolved_path": str(resolved), "error": None}
