TOOL = {
    "name": "file_edit",
    "description": "edit local file by replacing old_string with new_string",
    "parameters": {
        "path": {"type": "str", "required": True, "desc": "absolute path"},
        "old_string": {"type": "str", "required": True, "desc": "string to find"},
        "new_string": {"type": "str", "required": True, "desc": "replacement string"},
        "replace_all": {"type": "bool", "required": False, "desc": "replace all matches, default False"}
    },
}

def execute(path, old_string, new_string, replace_all=False):
    if old_string == new_string:
        return {"error": "old_string and new_string are identical"}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        count = content.count(old_string)
        if count == 0:
            return {"error": f"old_string not found in {path}"}
        if count > 1 and not replace_all:
            return {"error": f"{count} matches found, use replace_all=True or provide more context"}
        new_content = content.replace(old_string, new_string)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return {"error": None, "message": f"replaced {count} match(es) in {path}"}
    except Exception as e:
        return {"error": str(e)}
