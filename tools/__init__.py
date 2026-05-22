import pkgutil
import importlib
import os

REGISTRY = {}
_pkg_dir = os.path.dirname(__file__)

# Auto-discover tools; sort by name for prompt cache stability.
_tool_names = sorted(
    module_name
    for _, module_name, _ in pkgutil.iter_modules([_pkg_dir])
    if not module_name.startswith("_")
)
for _module_name in _tool_names:
    _module = importlib.import_module(f"tools.{_module_name}")
    _tool_def = getattr(_module, "TOOL", None)
    _execute_fn = getattr(_module, "execute", None)
    if _tool_def and _execute_fn:
        REGISTRY[_tool_def["name"]] = {"definition": _tool_def, "execute": _execute_fn}
    # Support secondary tool definitions (e.g. TOOL_POP in context_switch.py)
    for attr_name in dir(_module):
        if attr_name.startswith("TOOL_") and attr_name != "TOOL":
            extra_def = getattr(_module, attr_name)
            extra_fn_name = f"execute_{attr_name[5:].lower()}"
            extra_fn = getattr(_module, extra_fn_name, None)
            if isinstance(extra_def, dict) and extra_def.get("name") and extra_fn:
                REGISTRY[extra_def["name"]] = {"definition": extra_def, "execute": extra_fn}

# JSON Schema type mapping (internal shorthand → OpenAI-compatible)
_TYPE_MAP = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "object": "object"}

def get_openai_tools() -> list:
    """Return all registered tools in OpenAI function-calling format."""
    tools = []
    for name, entry in REGISTRY.items():
        defn = entry["definition"]
        properties = {}
        required = []
        for p_name, p_info in defn.get("parameters", {}).items():
            raw_type = p_info.get("type", "str")
            properties[p_name] = {
                "type": _TYPE_MAP.get(raw_type, "string"),
                "description": p_info.get("desc", ""),
            }
            if p_info.get("required"):
                required.append(p_name)
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": defn.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return tools

def dispatch(tool_name: str, arguments: dict, _control: dict = None) -> dict:
    """Execute a tool by name. Always returns a JSON-serializable dict."""
    if tool_name not in REGISTRY:
        return {"error": f"unknown tool: {tool_name}"}
    try:
        fn = REGISTRY[tool_name]["execute"]
        # Only pass _control to tools that accept it (e.g. context_switch)
        if _control is not None:
            import inspect
            sig = inspect.signature(fn)
            if "_control" in sig.parameters:
                return fn(**arguments, _control=_control)
        return fn(**arguments)
    except TypeError as e:
        return {"error": f"{tool_name} called with wrong arguments: {e}"}
    except Exception as e:
        return {"error": f"{tool_name} failed: {type(e).__name__}: {e}"}
