import io
import json
from dataclasses import dataclass

import pytest
import torch

from lwfm.results.events import RunEvent
from lwfm.results.sinks import ConsoleSink, LocalFileSink, OutputSink


def read_json(path):
    return json.loads(path.read_text())


def read_events(run_dir):
    return [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]


def test_manifest_and_running_status_are_written_on_creation(tmp_path):
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


def test_events_are_on_disk_before_close(tmp_path):
    sink = LocalFileSink(tmp_path)
    sink.write(
        RunEvent("scalar", "train/loss/ce", torch.tensor(2.5), stage="fit", step=1)
    )
    (event,) = read_events(tmp_path)  # read while the sink is still open
    assert event["name"] == "train/loss/ce"
    assert event["value"] == 2.5
    assert (event["stage"], event["step"]) == ("fit", 1)


def test_dataclass_event_value_round_trips(tmp_path):
    @dataclass(frozen=True)
    class Reference:
        kind: str
        path: str

    reference = Reference("torch_checkpoint", "artifacts/fit_last.pt")
    sink = LocalFileSink(tmp_path)
    sink.write(RunEvent("artifact", "fit/checkpoint", reference))
    assert Reference(**read_events(tmp_path)[0]["value"]) == reference


def test_summary_holds_last_value_of_each_scalar(tmp_path):
    sink = LocalFileSink(tmp_path)
    for step, loss in enumerate([3.0, 2.0, 1.0]):
        sink.write(RunEvent("scalar", "train/loss/total", loss, step=step))
    sink.write(RunEvent("scalar", "eval/val/top1", 0.5))
    sink.write(RunEvent("message", "note", "not in the summary"))
    sink.close()
    summary = read_json(tmp_path / "summary.json")
    assert summary == {"train/loss/total": 1.0, "eval/val/top1": 0.5}


def test_status_event_updates_status_file_with_traceback(tmp_path):
    sink = LocalFileSink(tmp_path)
    sink.write(RunEvent("message", "traceback", "Traceback ... boom"))
    sink.write(RunEvent("status", "status", "failed"))
    status = read_json(tmp_path / "status.json")
    assert status["status"] == "failed"
    assert "boom" in status["traceback"]


def test_existing_run_directory_is_refused(tmp_path):
    LocalFileSink(tmp_path)
    with pytest.raises(FileExistsError, match="already holds a run"):
        LocalFileSink(tmp_path)


def test_no_temporary_files_are_left(tmp_path):
    sink = LocalFileSink(tmp_path)
    sink.write(RunEvent("status", "status", "completed"))
    sink.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "events.jsonl",
        "manifest.json",
        "status.json",
        "summary.json",
    ]


def test_non_json_value_fails_without_writing(tmp_path):
    sink = LocalFileSink(tmp_path)
    with pytest.raises(TypeError, match="'weird'.*object is not JSON-safe"):
        sink.write(RunEvent("table", "weird", [object()]))
    assert read_events(tmp_path) == []


def test_close_is_safe_to_call_twice(tmp_path):
    sink = LocalFileSink(tmp_path)
    sink.close()
    sink.close()


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
def test_console_line_format(event, line):
    stream = io.StringIO()
    ConsoleSink(stream).write(event)
    assert stream.getvalue() == line + "\n"


def test_output_sink_is_abstract():
    with pytest.raises(TypeError):
        OutputSink()  # type: ignore[abstract]
