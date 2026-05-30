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

