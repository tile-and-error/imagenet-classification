import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml

from lwfm.results.events import RunEvent
from lwfm.results.sinks import ConsoleSink, FileFormat, LocalFileSink, OutputSink

FORMATS = ["json", "yaml", "csv"]


def read_json(path: Path):
    return json.loads(path.read_text())


def read_events(run_dir: Path):
    return [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]


def test_manifest_and_running_status_are_written_on_creation(tmp_path: Path):
    LocalFileSink(tmp_path / "run", manifest_fields={"seed": 42})
    manifest = read_json(tmp_path / "run" / "manifest.json")
    assert manifest["seed"] == 42
    assert manifest["environment"].keys() == {
        "python",
        "platform",
        "torch",
        "torchvision",
        "cuda_device",
    }
    assert read_json(tmp_path / "run" / "status.json")["status"] == "running"


def test_events_are_on_disk_before_close(tmp_path: Path):
    sink = LocalFileSink(tmp_path)
    sink.write(
        RunEvent("scalar", "train/loss/ce", torch.tensor(2.5), stage="fit", step=1)
    )
    (event,) = read_events(tmp_path)  # read while the sink is still open
    assert event["name"] == "train/loss/ce"
    assert event["value"] == 2.5
    assert (event["stage"], event["step"]) == ("fit", 1)


def test_dataclass_event_value_round_trips(tmp_path: Path):
    @dataclass(frozen=True)
    class Reference:
        kind: str
        path: str

    reference = Reference("torch_checkpoint", "artifacts/fit_last.pt")
    sink = LocalFileSink(tmp_path)
    sink.write(RunEvent("artifact", "fit/checkpoint", reference))
    assert Reference(**read_events(tmp_path)[0]["value"]) == reference


def test_summary_holds_last_value_of_each_scalar(tmp_path: Path):
    sink = LocalFileSink(tmp_path)
    for step, loss in enumerate([3.0, 2.0, 1.0]):
        sink.write(RunEvent("scalar", "train/loss/total", loss, step=step))
    sink.write(RunEvent("scalar", "eval/val/top1", 0.5))
    sink.write(RunEvent("message", "note", "not in the summary"))
    sink.close()
    summary = read_json(tmp_path / "summary.json")
    assert summary == {"train/loss/total": 1.0, "eval/val/top1": 0.5}


def test_status_event_updates_status_file_with_traceback(tmp_path: Path):
    sink = LocalFileSink(tmp_path)
    sink.write(RunEvent("message", "traceback", "Traceback ... boom"))
    sink.write(RunEvent("status", "status", "failed"))
    status = read_json(tmp_path / "status.json")
    assert status["status"] == "failed"
    assert "boom" in status["traceback"]


def test_existing_run_directory_is_refused(tmp_path: Path):
    LocalFileSink(tmp_path)
    with pytest.raises(FileExistsError, match="already holds a run"):
        LocalFileSink(tmp_path)


def test_no_temporary_files_are_left(tmp_path: Path):
    sink = LocalFileSink(tmp_path)
    sink.write(RunEvent("status", "status", "completed"))
    sink.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "events.jsonl",
        "manifest.json",
        "status.json",
        "summary.json",
    ]


def test_non_json_value_fails_without_writing(tmp_path: Path):
    sink = LocalFileSink(tmp_path)
    with pytest.raises(TypeError, match="'weird'.*object is not JSON-safe"):
        sink.write(RunEvent("table", "weird", [object()]))
    assert read_events(tmp_path) == []


def test_close_is_safe_to_call_twice(tmp_path: Path):
    sink = LocalFileSink(tmp_path)
    sink.close()
    sink.close()


def read_events_as(run_dir: Path, file_format: FileFormat):
    """The events file parsed back into dicts. CSV cells come back as strings."""
    if file_format == "json":
        return read_events(run_dir)
    if file_format == "yaml":
        return yaml.safe_load((run_dir / "events.yaml").read_text()) or []
    with (run_dir / "events.csv").open(newline="") as events_file:
        return list(csv.DictReader(events_file))


def read_summary_as(run_dir: Path, file_format: FileFormat):
    if file_format == "json":
        return read_json(run_dir / "summary.json")
    if file_format == "yaml":
        return yaml.safe_load((run_dir / "summary.yaml").read_text())
    with (run_dir / "summary.csv").open(newline="") as summary_file:
        return {
            row["name"]: float(row["value"]) for row in csv.DictReader(summary_file)
        }


@pytest.mark.parametrize("file_format", FORMATS)
def test_events_round_trip_in_every_format(tmp_path: Path, file_format: FileFormat):
    sink = LocalFileSink(tmp_path, file_format=file_format)
    sink.write(
        RunEvent("scalar", "train/loss/ce", torch.tensor(2.5), stage="fit", step=1)
    )
    sink.write(RunEvent("artifact", "checkpoint/best", tmp_path / "best.pt"))
    sink.write(RunEvent("message", "traceback", "Traceback:\n  line 1, in fit"))
    scalar, artifact, message = read_events_as(tmp_path, file_format)  # still open

    as_read = str if file_format == "csv" else (lambda value: value)
    assert (scalar["name"], scalar["value"]) == ("train/loss/ce", as_read(2.5))
    assert (scalar["stage"], scalar["step"]) == ("fit", as_read(1))
    assert artifact["value"] == str(tmp_path / "best.pt")
    assert message["value"] == "Traceback:\n  line 1, in fit"  # multi-line survives


@pytest.mark.parametrize("file_format", FORMATS)
def test_summary_in_every_format(tmp_path: Path, file_format: FileFormat):
    sink = LocalFileSink(tmp_path, file_format=file_format)
    for step, loss in enumerate([3.0, 1.0]):
        sink.write(RunEvent("scalar", "train/loss/total", loss, step=step))
    sink.write(RunEvent("scalar", "eval/val/top1", 0.5))
    sink.close()
    summary = read_summary_as(tmp_path, file_format)
    assert summary == {"train/loss/total": 1.0, "eval/val/top1": 0.5}


@pytest.mark.parametrize(
    ("file_format", "names"),
    [
        ("json", ["events.jsonl", "manifest.json", "status.json", "summary.json"]),
        ("yaml", ["events.yaml", "manifest.yaml", "status.yaml", "summary.yaml"]),
        ("csv", ["events.csv", "manifest.csv", "status.csv", "summary.csv"]),
    ],
)
def test_every_file_follows_the_format(
    tmp_path: Path, file_format: FileFormat, names: list[str]
):
    LocalFileSink(tmp_path, file_format=file_format).close()
    assert sorted(p.name for p in tmp_path.iterdir()) == names


def read_record_as(path: Path, file_format: FileFormat):
    """A manifest or status file as a dict. CSV comes back flat, with dotted keys."""
    if file_format == "yaml":
        return yaml.safe_load(path.read_text())
    with path.open(newline="") as record_file:
        return {row["key"]: row["value"] for row in csv.DictReader(record_file)}


@pytest.mark.parametrize("file_format", ["yaml", "csv"])
def test_manifest_and_status_in_every_format(tmp_path: Path, file_format: FileFormat):
    sink = LocalFileSink(
        tmp_path, manifest_fields={"seed": 42}, file_format=file_format
    )
    sink.write(RunEvent("message", "traceback", "Traceback:\n  boom"))
    sink.write(RunEvent("status", "status", "failed"))

    manifest = read_record_as(tmp_path / f"manifest.{file_format}", file_format)
    status = read_record_as(tmp_path / f"status.{file_format}", file_format)
    if file_format == "yaml":
        assert manifest["seed"] == 42
        assert "torch" in manifest["environment"]
    else:  # flat: nested keys dotted, values as text
        assert manifest["seed"] == "42"
        assert "environment.torch" in manifest
    assert status["status"] == "failed"
    assert status["traceback"] == "Traceback:\n  boom"  # multi-line survives CSV too


def test_a_run_of_another_format_is_refused(tmp_path: Path):
    LocalFileSink(tmp_path, file_format="csv")
    with pytest.raises(FileExistsError, match="already holds a run"):
        LocalFileSink(tmp_path, file_format="yaml")


def test_unknown_format_fails_before_creating_anything(tmp_path: Path):
    with pytest.raises(ValueError, match="file_format"):
        LocalFileSink(tmp_path / "run", file_format="xml")  # type: ignore[arg-type]
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("file_format", FORMATS)
def test_bad_value_fails_without_writing_in_every_format(
    tmp_path: Path, file_format: FileFormat
):
    sink = LocalFileSink(tmp_path, file_format=file_format)
    with pytest.raises(TypeError, match="not JSON-safe"):
        sink.write(RunEvent("table", "weird", [object()]))
    assert read_events_as(tmp_path, file_format) == []


@pytest.mark.parametrize(
    ("event", "line"),
    [
        (
            RunEvent("scalar", "train/loss/total", 0.43211, stage="fit", step=10),
            "[fit step 10] train/loss/total = 0.4321",
        ),
        (
            RunEvent("scalar", "eval/val/top1", 0.5, stage="eval_val"),
            "[eval_val] eval/val/top1 = 0.5",
        ),
        (RunEvent("status", "status", "completed"), "status = completed"),
    ],
    ids=["stage_and_step", "stage_only", "neither"],
)
def test_console_line_format(event: RunEvent, line: str):
    stream = io.StringIO()
    ConsoleSink(stream).write(event)
    assert stream.getvalue() == line + "\n"


def test_output_sink_is_abstract():
    with pytest.raises(TypeError):
        OutputSink()  # type: ignore[abstract]
