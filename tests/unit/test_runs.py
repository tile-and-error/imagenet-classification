import pytest

from lwfm.results.events import RunEvent
from lwfm.results.router import OutputRouter
from lwfm.results.runs import completed_runs, run_status, skip_if_completed
from lwfm.results.sinks import LocalFileSink


def record(run_dir, top1, fail=False):
    """A run folder as a real run leaves it: completed, or failed."""
    try:
        with OutputRouter(LocalFileSink(run_dir, {"student": "resnet18"})) as sink:
            sink.write(RunEvent("scalar", "eval/val/top1", top1))
            if fail:
                raise RuntimeError("crash")
    except RuntimeError:
        pass


def test_completed_runs_reads_manifest_and_final_values(tmp_path):
    record(tmp_path / "a/s0", 0.5)
    record(tmp_path / "a/s1", 0.7)
    record(tmp_path / "b/s0", 0.9, fail=True)
    rows = completed_runs(tmp_path)
    assert [(r["run"], r["student"], r["eval/val/top1"]) for r in rows] == [
        ("a/s0", "resnet18", 0.5),
        ("a/s1", "resnet18", 0.7),
    ]
    assert [r["run"] for r in completed_runs(tmp_path, "a/s1")] == ["a/s1"]


def test_skip_if_completed(tmp_path):
    record(tmp_path / "done", 0.5)
    record(tmp_path / "crashed", 0.5, fail=True)
    assert skip_if_completed(tmp_path / "done") is True
    assert skip_if_completed(tmp_path / "missing") is False
    assert skip_if_completed(tmp_path / "crashed") is False
    assert not (tmp_path / "crashed").exists()  # deleted, to start again


def test_a_run_still_marked_running_is_not_touched(tmp_path):
    LocalFileSink(tmp_path / "live")  # opened and never closed: "running"
    assert run_status(tmp_path / "live") == "running"
    with pytest.raises(FileExistsError, match="marked running"):
        skip_if_completed(tmp_path / "live")
    assert (tmp_path / "live").exists()
