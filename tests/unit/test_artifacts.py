import json
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from lwfm.results.artifacts import save_artifact
from lwfm.results.router import OutputRouter
from lwfm.results.sinks import LocalFileSink

from conftest import is_png


def test_the_file_is_saved_then_recorded_with_a_relative_path(tmp_path, sink):
    path = save_artifact(
        tmp_path,
        "notes/a.txt",
        lambda target: target.write_text("hello"),
        sink,
        name="note/a",
        step=3,
    )
    assert path == tmp_path / "notes/a.txt" and path.read_text() == "hello"
    [event] = sink.events
    assert (event.kind, event.name, event.value, event.step) == (
        "artifact",
        "note/a",
        Path("notes/a.txt"),
        3,
    )
    assert sorted(p.name for p in path.parent.iterdir()) == ["a.txt"]  # no temp file


def test_a_figure_keeps_its_format(tmp_path):
    # savefig picks the format from the suffix, so the temporary file keeps it
    figure = Figure()
    figure.subplots().plot([0, 1], [0, 1])
    path = save_artifact(tmp_path, "plots/line.png", figure.savefig, None, name="p")
    assert is_png(path)


def test_a_failed_write_keeps_the_old_file_and_records_nothing(tmp_path, sink):
    save_artifact(
        tmp_path, "a.txt", lambda target: target.write_text("old"), None, name="a"
    )

    def fail(target):
        target.write_text("half")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        save_artifact(tmp_path, "a.txt", fail, sink, name="a")
    assert (tmp_path / "a.txt").read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.txt"]
    assert sink.events == []


def test_the_event_reaches_the_run_record(tmp_path):
    with OutputRouter(LocalFileSink(tmp_path)) as sink:
        figure = Figure()
        figure.subplots().bar(["cat", "dog"], [3, 5])
        save_artifact(
            tmp_path, "plots/counts.png", figure.savefig, sink, name="plot/counts"
        )
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").open()]
    assert {
        "kind": "artifact",
        "name": "plot/counts",
        "value": "plots/counts.png",
    }.items() <= next(event for event in events if event["kind"] == "artifact").items()
    assert (tmp_path / "plots/counts.png").exists()
