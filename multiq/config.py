"""Configuration and reproducibility helpers."""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects silent duplicate-key overrides."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(
                f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise KeyError(f"Environment variable {name!r} is required by the config")
        return os.environ[name]

    return _ENV_PATTERN.sub(replace, value)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar and sequence values."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_parents(
    path: Path, ancestors: tuple[Path, ...] = ()
) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in ancestors:
        cycle = " -> ".join(str(item) for item in (*ancestors, resolved))
        raise ValueError(f"Config inheritance cycle: {cycle}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing config: {resolved}")
    with resolved.open() as handle:
        config = yaml.load(handle, Loader=_UniqueKeySafeLoader)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {resolved}")
    parent = config.pop("extends", None)
    if parent is None:
        return config
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError(f"Config extends must be a non-empty path: {resolved}")
    parent_path = Path(parent).expanduser()
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    inherited = _load_yaml_with_parents(parent_path, (*ancestors, resolved))
    return _deep_merge(inherited, config)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load inherited YAML, then expand environment variables deterministically."""
    config_path = Path(path).resolve()
    for parent in (config_path.parent, *config_path.parents):
        candidate = parent / ".env"
        if candidate.is_file():
            _load_dotenv(candidate)
            break
    config = _load_yaml_with_parents(config_path)
    config = _expand(config)
    config["_config_path"] = str(config_path)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
