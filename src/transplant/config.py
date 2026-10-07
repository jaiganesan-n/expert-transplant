"""YAML configs with includes, deep merge and command-line overrides.

A config file may list other files under ``include:`` (paths relative to the
including file). Later files and ``--set key.sub=value`` overrides win.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(dict):
    """A dict with attribute access for nested keys (``cfg.e2e.lr``)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:  # pragma: no cover - message only
            raise AttributeError(f"config has no key '{key}'") from exc
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict:
        return _plain(self)


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict) and not isinstance(obj, Config):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, Config):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_file(path: Path, seen: set[Path]) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"include cycle at {path}")
    seen = seen | {path}
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    merged: dict = {}
    for inc in data.pop("include", []) or []:
        merged = deep_merge(merged, _load_file(path.parent / inc, seen))
    return deep_merge(merged, data)


def parse_override(text: str) -> tuple[list[str], Any]:
    if "=" not in text:
        raise ValueError(f"override '{text}' must look like key.sub=value")
    key, raw = text.split("=", 1)
    return key.strip().split("."), yaml.safe_load(raw)


def set_path(cfg: dict, keys: list[str], value: Any) -> None:
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_config(paths: Iterable[str | Path], overrides: Iterable[str] = ()) -> Config:
    merged: dict = {}
    for p in paths:
        merged = deep_merge(merged, _load_file(Path(p), set()))
    for text in overrides:
        keys, value = parse_override(text)
        set_path(merged, keys, value)
    return _wrap(merged)


def config_hash(cfg: dict) -> str:
    """Stable short hash of a config, logged with every run."""
    blob = json.dumps(_plain(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(_plain(cfg), fh, sort_keys=False)
