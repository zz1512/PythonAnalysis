from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config json must be an object at top-level")
    return payload


def pick_value(*, cli_value: Any, config_value: Any, preset_value: Any) -> Any:
    """Apply the priority: CLI > config > preset.

    We treat "None" as "not provided".
    """

    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return preset_value


def pick_path(
    *,
    cli_value: Path | None,
    config_payload: dict[str, Any],
    config_key: str,
    preset_value: Path,
) -> Path:
    raw = pick_value(
        cli_value=cli_value,
        config_value=config_payload.get(config_key),
        preset_value=preset_value,
    )
    return raw if isinstance(raw, Path) else Path(raw)

