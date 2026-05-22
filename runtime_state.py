"""Runtime config state shared by main loop and tools."""

_CURRENT_CONFIG = {}
_CURRENT_CONFIG_PATH = ""


def set_config(cfg: dict, cfg_path: str = "") -> None:
    global _CURRENT_CONFIG, _CURRENT_CONFIG_PATH
    _CURRENT_CONFIG = dict(cfg or {})
    _CURRENT_CONFIG_PATH = str(cfg_path or "")


def get_config() -> dict:
    if not _CURRENT_CONFIG:
        raise RuntimeError("config state not loaded")
    return _CURRENT_CONFIG


def get_config_path() -> str:
    return _CURRENT_CONFIG_PATH


# Backward-compatible aliases for existing tool code paths.
def set_project(cfg: dict, cfg_path: str = "") -> None:
    set_config(cfg, cfg_path)


def get_project() -> dict:
    return get_config()


def get_project_path() -> str:
    return get_config_path()
