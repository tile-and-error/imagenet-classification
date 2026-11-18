import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TextIO, cast

from lwfm.results.events import RunEvent, RunStatus, utc_now
from lwfm.results.manifest import environment


class SinkError(RuntimeError):
    """A sink failed to store an event. Raised instead of letting a run look
    successful when its results were not saved.
    """


class OutputSink(ABC):
    """A destination for events, such as local files or the console."""

    @abstractmethod
    def write(self, event: RunEvent) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


def _json_default(obj: object) -> object:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON-safe")


def to_json(event: RunEvent) -> str:
    try:
        return json.dumps(asdict(event), default=_json_default)
    except TypeError as error:
        raise TypeError(f"event {event.name!r}: {error}") from None


def write_json_atomic(path: Path, data: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=_json_default) + "\n")
    os.replace(tmp, path)  # readers never see a half-written file


class LocalFileSink(OutputSink):
    """Writes a run directory: the local, canonical record of a run.

    On creation: `manifest.json` (the given manifest plus `environment()`) and
    `status.json` = running. Each event is appended to `events.jsonl` and flushed
    at once, so a crashed run keeps its data. Status events rewrite `status.json`,
    with the traceback after a failure. On close: `summary.json`, the last value
    of every scalar. Refuses a directory that already holds a run.
    """

    def __init__(self, run_dir: str | Path, manifest: dict | None = None):
        self.run_dir = Path(run_dir)
        events_path = self.run_dir / "events.jsonl"
        if events_path.exists():
            raise FileExistsError(f"{self.run_dir} already holds a run")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            self.run_dir / "manifest.json",
            {**(manifest or {}), "environment": environment()},
        )
        self._traceback: str | None = None
        self._write_status("running")
        self._last_scalars: dict[str, object] = {}
        self._events = events_path.open("a", encoding="utf-8")

    def write(self, event: RunEvent) -> None:
        line = to_json(event)  # fails before anything is written
        self._events.write(line + "\n")
        self._events.flush()
        if event.kind == "scalar":
            self._last_scalars[event.name] = event.value
        elif event.kind == "message" and event.name == "traceback":
            self._traceback = str(event.value)
        elif event.kind == "status":
            self._write_status(cast(RunStatus, event.value))  # checked by RunEvent

    def close(self) -> None:
        if self._events.closed:
            return
        write_json_atomic(self.run_dir / "summary.json", self._last_scalars)
        self._events.close()

    def _write_status(self, status: RunStatus) -> None:
        record: dict[str, object] = {"status": status, "updated_utc": utc_now()}
        if self._traceback is not None:
            record["traceback"] = self._traceback
        write_json_atomic(self.run_dir / "status.json", record)


class ConsoleSink(OutputSink):
    """Prints one line per event, e.g. `[fit step 10] train/loss/total = 0.4321`.

    Writes to a stream (stdout by default) rather than through `logging`: it is
    the console destination, so logging configuration shouldn't hide it.
    """

    def __init__(self, stream: TextIO = sys.stdout):
        self.stream = stream

    def write(self, event: RunEvent) -> None:
        where = " ".join(
            part
            for part in (
                event.stage,
                None if event.step is None else f"step {event.step}",
            )
            if part
        )
        value = f"{event.value:.4g}" if isinstance(event.value, float) else event.value
        prefix = f"[{where}] " if where else ""
        self.stream.write(f"{prefix}{event.name} = {value}\n")

    def close(self) -> None:
        self.stream.flush()
