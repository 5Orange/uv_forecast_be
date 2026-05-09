from __future__ import annotations

class AllKeyExhaustedError(Exception):
    """every api key in the pool has hit its quota"""

class KeyPool:
    def __init__(self, keys: list[str], service: str = "API"):
        if not keys:
            raise ValueError("No keys provided")
        self._keys = list(keys)
        self._exhausted: set[str] = set()
        self._index = 0
        self._service = service

    @property
    def available(self) -> bool:
        return len(self._exhausted) < len(self._keys)
    @property
    def remaining(self) -> int:
        return len(self._keys) - len(self._exhausted)

    def next_key(self) -> str:
        if not self.available:
            raise AllKeyExhaustedError(
                f"{self._service}: all {len(self._keys)} keys exhausted"
            )
        for _ in range(len(self._keys)):
            key = self._keys[self._index % len(self._keys)]
            self._index += 1
            if key not in self._exhausted:
                return key
        raise AllKeyExhaustedError(
            f"{self._service}: all {len(self._keys)} keys exhausted"
        )

    def mark_exhausted(self, key: str) -> None:
        self._exhausted.add(key)
        remaining = self.remaining
        masked = f"...{key[-6:]}" if len(key) > 6 else "***"
        print(f"{self._service}: key {masked} exhausted")