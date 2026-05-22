import json
from pathlib import Path

_CONFIG_DIR = Path("config")
_META_SKILL_PATH = Path("prompts/skills/meta.md")
_STYLE_PATH = Path("prompts/skills/style.md")
_PREFERENCES_DIR = Path("prompts/preferences")


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config file must be a JSON object: {path}")
    return data


def _read_optional_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def load_llm_config() -> dict:
    path = _CONFIG_DIR / "llm.json"
    return _read_json(path)


def load_ssh_config() -> tuple[dict, str]:
    path = _CONFIG_DIR / "ssh.json"
    cfg = _read_json(path)
    return cfg, str(path).replace("\\", "/")


def load_meta_skill_prompt() -> str:
    return _read_optional_text(_META_SKILL_PATH)


def load_style_prompt() -> str:
    return _read_optional_text(_STYLE_PATH)


def load_preference_prompts() -> str:
    remote_system = _read_optional_text(_PREFERENCES_DIR / "remote_system.md")
    training = _read_optional_text(_PREFERENCES_DIR / "training.md")

    blocks = []
    if remote_system:
        blocks.append("# 远程机与系统环境偏好\n\n" + remote_system)
    if training:
        blocks.append("# 训练参数偏好\n\n" + training)
    return "\n\n".join(blocks)
