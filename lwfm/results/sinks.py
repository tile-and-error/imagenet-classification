import csv
import io
import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Literal, TextIO, cast, get_args

import yaml

from lwfm.results.events import RunEvent, RunStatus, utc_now
from lwfm.results.manifest import environment

# The format of every file in a run directory
type FileFormat = Literal["json", "yaml", "csv"]
FILE_FORMATS: tuple[str, ...] = get_args(FileFormat.__value__)
# The events file is appended to, so JSON uses JSON Lines: one object per line
EVENTS_FILE_NAMES = {"json": "events.jsonl", "yaml": "events.yaml", "csv": "events.csv"}
# The columns of events.csv, in RunEvent's field order
EVENT_FIELDS = tuple(field.name for field in fields(RunEvent))


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


def _json_default(unencodable_value):
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


def write_text_atomic(target_path: Path, text: str) -> None:
    """Writes `text` to a temporary file, then renames it over `target_path`."""
    temp_path = target_path.with_name(target_path.name + ".tmp")
    temp_path.write_text(text, encoding="utf-8", newline="")
    os.replace(temp_path, target_path)  # readers never see a half-written file


def csv_row(cells) -> str:
    """One CSV line, quoted as needed (e.g. a traceback spanning several lines)."""
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerow(cells)
    return buffer.getvalue()


def csv_cell(value):
    # Numbers and strings as they are; nested values (a table's rows, a dataclass)
    # as JSON text; None as an empty cell
    if value is None:
        return ""
    if isinstance(value, dict | list):
        return json.dumps(value)
    return value


def format_event(event: RunEvent, file_format: FileFormat) -> str:
    """The text appended to the events file for one event."""
    json_line = to_json(event)  # also checks that the value is serialisable
    if file_format == "json":
        return json_line + "\n"
    record = json.loads(json_line)  # plain types: dataclasses and paths converted
    if file_format == "yaml":
        # One list item, so the whole file is a YAML list of events
        return yaml.safe_dump([record], sort_keys=False)
    return csv_row(csv_cell(record[field]) for field in EVENT_FIELDS)


def flatten(record, prefix: str = ""):
    """Nested dicts as dotted keys: {"environment": {"torch": "2.9"}} becomes
    {"environment.torch": "2.9"}."""
    flat: dict[str, object] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            flat |= flatten(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def format_record(record, file_format: FileFormat, key_header: str = "key") -> str:
    """The text of a whole-file record (manifest, status, summary). CSV has no
    nesting, so it gets one `key,value` row per flattened entry."""
    json_text = json.dumps(record, indent=2, default=_json_default)
    if file_format == "json":
        return json_text + "\n"
    plain_record = json.loads(json_text)  # dataclasses and paths converted
    if file_format == "yaml":
        return yaml.safe_dump(plain_record, sort_keys=False)
    rows = (csv_row([k, csv_cell(v)]) for k, v in flatten(plain_record).items())
    return csv_row([key_header, "value"]) + "".join(rows)


class LocalFileSink(OutputSink):
    """Writes a run directory: the local, canonical record of a run.

    On creation: the manifest (the given manifest plus `environment()`) and the
    status, `running`. Each event is appended to the events file and flushed at
    once, so a crashed run keeps its data. Status events rewrite the status file,
    with the traceback after a failure. On close: the summary, the last value of
    every scalar. Refuses a directory that already holds a run.

    `file_format` sets the format of every file: `"json"` writes `manifest.json`,
    `status.json`, `events.jsonl` (one object per line) and `summary.json`;
    `"yaml"` writes the same names as `.yaml`, with the events as a YAML list;
    `"csv"` writes them as `.csv`: one row per event (columns as in `RunEvent`),
    and `key,value` rows for the others, nested keys dotted (`environment.torch`)
    and the summary's header `name,value`.
    """

    def __init__(
        self,
        run_dir: str | Path,
        manifest_fields=None,
        file_format: FileFormat = "json",
    ):
        if file_format not in FILE_FORMATS:
            raise ValueError(f"file_format must be one of {FILE_FORMATS}")
        self.run_dir = Path(run_dir)
        self.file_format: FileFormat = file_format
        # A run of any format claims the directory
        if any((self.run_dir / name).exists() for name in EVENTS_FILE_NAMES.values()):
            raise FileExistsError(f"{self.run_dir} already holds a run")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Exclusive-create mode: if another sink claimed the directory since the
        # check above, fail rather than write into its run. newline="" keeps line
        # endings as written, which CSV needs for quoted multi-line cells.
        events_path = self.run_dir / EVENTS_FILE_NAMES[file_format]
        try:
            self._events_file = events_path.open("x", encoding="utf-8", newline="")
        except FileExistsError:
            raise FileExistsError(f"{self.run_dir} already holds a run") from None
        if file_format == "csv":
            self._append(csv_row(EVENT_FIELDS))
        self._write_record(
            "manifest", {**(manifest_fields or {}), "environment": environment()}
        )
        self._failure_traceback: str | None = None
        self._write_status("running")
        self._last_scalar_values: dict[str, object] = {}

    def write(self, event: RunEvent) -> None:
        """Write a RunEvent to the sink"""
        self._append(format_event(event, self.file_format))  # fails before writing
        if event.kind == "scalar":
            self._last_scalar_values[event.name] = event.value
        elif event.kind == "message" and event.name == "traceback":
            self._failure_traceback = str(event.value)
        elif event.kind == "status":
            self._write_status(cast(RunStatus, event.value))  # checked by RunEvent

    def close(self) -> None:
        if self._events_file.closed:
            return
        self._write_record("summary", self._last_scalar_values, key_header="name")
        self._events_file.close()

    def _append(self, text: str) -> None:
        self._events_file.write(text)
        self._events_file.flush()

    def _write_status(self, status: RunStatus) -> None:
        status_record: dict[str, object] = {"status": status, "updated_utc": utc_now()}
        if self._failure_traceback is not None:
            status_record["traceback"] = self._failure_traceback
        self._write_record("status", status_record)

    def _write_record(self, stem: str, record, key_header: str = "key") -> None:
        # Written to a temporary file, then renamed: never seen half-written
        write_text_atomic(
            self.run_dir / f"{stem}.{self.file_format}",
            format_record(record, self.file_format, key_header),
        )


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
        self.output_stream.flush()

    def close(self) -> None:
        self.output_stream.flush()
