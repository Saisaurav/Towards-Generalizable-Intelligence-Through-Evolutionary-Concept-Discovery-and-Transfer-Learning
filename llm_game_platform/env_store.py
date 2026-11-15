"""Persistence for generated environments: source + metadata, reloadable and hash-checked."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

_ID = re.compile(r"^[0-9a-f]{32}$")
_REPO_ROOT = Path(__file__).resolve().parents[1]


class EnvStoreError(RuntimeError):
    pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EnvStore:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root or os.getenv("GENERATED_ENVS_DIR") or _REPO_ROOT / "generated_envs")

    def path(self, env_id: str) -> Path:
        if not _ID.fullmatch(env_id):
            raise EnvStoreError("invalid environment id")
        return self.root / env_id

    def save(self, env_id: str, *, prompt: str, source: str, spec: dict[str, Any],
             report: dict[str, Any], attempts: int, model: str | None = None) -> Path:
        directory = self.path(env_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "env.py").write_text(source, encoding="utf-8")
        meta = {
            "id": env_id, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompt": prompt, "model": model, "attempts": attempts,
            "source_sha256": sha256_text(source), "spec": spec, "validation": report,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return directory

    def load(self, env_id: str) -> tuple[str, dict[str, Any]]:
        directory = self.path(env_id)
        try:
            source = (directory / "env.py").read_text(encoding="utf-8")
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EnvStoreError(f"could not load environment {env_id}: {error}") from error
        if meta.get("source_sha256") != sha256_text(source):
            raise EnvStoreError(f"environment {env_id} source does not match its recorded hash")
        return source, meta

    def source_path(self, env_id: str) -> Path:
        return self.path(env_id) / "env.py"

    def list(self) -> list[dict[str, Any]]:
        items = []
        if self.root.is_dir():
            for child in sorted(self.root.iterdir()):
                if _ID.fullmatch(child.name) and (child / "meta.json").is_file():
                    try:
                        meta = json.loads((child / "meta.json").read_text(encoding="utf-8"))
                        items.append({"id": meta["id"], "title": meta["spec"]["title"], "created": meta["created"]})
                    except (OSError, KeyError, json.JSONDecodeError):
                        continue
        return items
