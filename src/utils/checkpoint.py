from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from src.config import CHECKPOINTS_DIR

class Checkpoint:
    def __init__(self, name: str):
        self._path: Path = CHECKPOINTS_DIR / f"{name}.json"
        self._done: set[str] = set()
        self._load()

    @property
    def has_progress(self) -> bool:
        return bool(self._done)

    def is_done(self, key: str) -> bool:
        return key in self._done

    def mark_done(self, key: str) -> None:
        self._done.add(key)
        self._persist()

    def pending(self, all_keys: list[str]) -> list[str]:
        return [k for k in all_keys if k not in self._done]

    def clear(self) -> None:
        self._done.clear()
        if self._path.exists():
            self._path.unlink()

    @property
    def done_count(self) -> int:
        return len(self._done)

    def _load(self) -> None:
        if self._path.exists():
            data = json.loads(self._path.read_text())
            self._done = set(data.get('completed', []))
    def _persist(self) -> None:
        payload = {
            "completed": sorted(self._done),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        self._path.write_text(json.dumps(payload, indent=2))


def append_csv(df: pd.DataFrame, path: Path):
    if df.empty:
        return
    df.to_csv(path, mode="a", header=not path.exists(), index=False)

def truncate_csv(path: Path) -> None:
    if path.exists():
        path.unlink()
