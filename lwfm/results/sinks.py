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


def _json_default(unencodable_value: object) -> object:
    """This converts a non-standard object into a JSON representation."""
    if is_dataclass(unencodable_value) and not isinstance(unencodable_value, type):
        return asdict(unencodable_value)
    if isinstance(unencodable_value, Path):
        return str(unencodable_value)
    raise TypeError(f"{type(unencodable_value).__name__} is not JSON-safe")


def to_json(event: RunEvent) -> str:
    """Convert a RunEvent to JSON representation."""
    try:
        return json.dumps(asdict(event), default=_json_default)
    except TypeError as error:
        raise TypeError(f"event {event.name!r}: {error}") from None


def write_json_atomic(target_path: Path, json_data: object) -> None:
    """Write JSON to a file safely. It uses a temp directory and then OS.replace
    to make sure readers never see a half written file.
    """
    temp_path = target_path.with_name(target_path.name + ".tmp")
    temp_path.write_text(json.dumps(json_data, indent=2, default=_json_default) + "\n")
    os.replace(temp_path, target_path)  # readers never see a half-written file


class LocalFileSink(OutputSink):
    """Writes a run directory: the local, canonical record of a run.

    On creation: `manifest.json` (the given manifest plus `environment()`) and
    `status.json` = running. Each event is appended to `events.jsonl` and flushed
    at once, so a crashed run keeps its data. Status events rewrite `status.json`,
    with the traceback after a failure. On close: `summary.json`, the last value
    of every scalar. Refuses a directory that already holds a run.
    """

    def __init__(self, run_dir: str | Path, manifest_fields: dict | None = None):
        self.run_dir = Path(run_dir)
        events_path = self.run_dir / "events.jsonl"
        if events_path.exists():
            raise FileExistsError(f"{self.run_dir} already holds a run")
        # Create the `run_dir` directory, or do nothing otherwise.
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            self.run_dir / "manifest.json",
            {**(manifest_fields or {}), "environment": environment()},
        )
        self._failure_traceback: str | None = None
        self._write_status("running")
        self._last_scalar_values: dict[str, object] = {}
        self._events_file = events_path.open("a", encoding="utf-8")

    def write(self, event: RunEvent) -> None:
        """Write a RunEvent to the sink"""
        json_line = to_json(event)  # fails before anything is written
        self._events_file.write(json_line + "\n")
        self._events_file.flush()
        if event.kind == "scalar":
            self._last_scalar_values[event.name] = event.value
        elif event.kind == "message" and event.name == "traceback":
            self._failure_traceback = str(event.value)
        elif event.kind == "status":
            self._write_status(cast(RunStatus, event.value))  # checked by RunEvent

    def close(self) -> None:
        if self._events_file.closed:
            return
        write_json_atomic(self.run_dir / "summary.json", self._last_scalar_values)
        self._events_file.close()

    def _write_status(self, status: RunStatus) -> None:
        status_record: dict[str, object] = {"status": status, "updated_utc": utc_now()}
        if self._failure_traceback is not None:
            status_record["traceback"] = self._failure_traceback
        write_json_atomic(self.run_dir / "status.json", status_record)


class ConsoleSink(OutputSink):
    """Prints one line per event, e.g. `[fit step 10] train/loss/total = 0.4321`.

    Writes to a stream (stdout by default) rather than through `logging`: it is
    the console destination, so logging configuration shouldn't hide it.
    """

    def __init__(self, output_stream: TextIO = sys.stdout):
        self.output_stream = output_stream

    def write(self, event: RunEvent) -> None:
        stage_and_step = " ".join(
            part
            for part in (
                event.stage,
                None if event.step is None else f"step {event.step}",
            )
            if part
        )
        shown_value = (
            f"{event.value:.4g}" if isinstance(event.value, float) else event.value
        )
        prefix = f"[{stage_and_step}] " if stage_and_step else ""
        self.output_stream.write(f"{prefix}{event.name} = {shown_value}\n")

    def close(self) -> None:
        self.output_stream.flush()
