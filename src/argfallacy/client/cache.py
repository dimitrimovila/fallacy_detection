"""The call cache: a question already asked is never asked again.

SQLite in ``runs/cache.sqlite``, keyed by everything that decides the answer.
Two consequences worth stating, because they are the reason the cache is shaped
this way rather than being a folder of files:

* a new prompt version changes the key, so it costs new calls for that prompt
  and leaves every other cached answer alone;
* nothing is ever evicted.  A run interrupted after nine hours resumes from the
  tenth, and the only way to lose work is to delete the file by hand.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from ..paths import REPO_ROOT
from .call import RawResponse, Request, dumps

RUNS_DIR = REPO_ROOT / "runs"
CACHE_FILE = RUNS_DIR / "cache.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    cache_key    TEXT PRIMARY KEY,
    model_id     TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    sample_index INTEGER NOT NULL,
    params       TEXT NOT NULL,
    response     TEXT NOT NULL,
    stored_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS responses_by_model ON responses (model_id, prompt_version);
"""


def cache_key(request: Request) -> str:
    """sha256 of everything that decides the answer.

    The rendered prompt goes in whole rather than an item id: two items with the
    same text are the same question, and a template edit that changes one word
    has to invalidate the entry even if the version string was not bumped.
    Revision, tier and reasoning are in it too: the same model id at another
    commit, or with thinking switched on, is a different model.
    """
    material = dumps({
        "model_id": request.model_id,
        "prompt_version": request.prompt_version,
        "prompt": request.prompt,
        "params": request.generation_params(),
        "sample_index": request.sample_index,
        "revision": request.revision,
        "tier": request.tier,
        "reasoning": request.reasoning or {},
    })
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class ResponseCache:
    """Read-through store for :class:`RawResponse`.  Failures are never stored.

    Calls go out concurrently, so the connection is shared across threads and
    every use of it is serialised by a lock.  SQLite handles the contention far
    better than the network calls it is guarding, and one writer keeps the file
    consistent when a run is killed mid-flight.
    """

    def __init__(self, path: str | Path = CACHE_FILE) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self.connection.executescript(SCHEMA)
            self.connection.commit()

    def __enter__(self) -> ResponseCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def get(self, key: str) -> RawResponse | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT response FROM responses WHERE cache_key = ?", (key,)
            ).fetchone()
        return RawResponse.from_dict(json.loads(row[0])) if row else None

    def put(self, key: str, request: Request, response: RawResponse) -> bool:
        """Store a successful answer.  Returns whether it was stored.

        An error is deliberately not cached: the call has to be retried on the
        next run, which is what makes resume work after a model went down.
        """
        if response.error:
            return False
        with self._lock:
            self.connection.execute(
                "INSERT OR REPLACE INTO responses "
                "(cache_key, model_id, prompt_version, sample_index, params, response) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    key,
                    request.model_id,
                    request.prompt_version,
                    request.sample_index,
                    dumps(request.generation_params()),
                    json.dumps(response.to_dict(), ensure_ascii=False),
                ),
            )
            self.connection.commit()
        return True

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def count(self) -> int:
        with self._lock:
            return int(
                self.connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
            )
