import json
from pathlib import Path
from typing import NoReturn

import pytest

from lwfm.results.events import RunEvent
from lwfm.results.router import OutputRouter
from lwfm.results.sinks import LocalFileSink, OutputSink, SinkError

from conftest import ListSink


class BrokenSink(OutputSink):
    def write(self, event: RunEvent) -> NoReturn:
        raise OSError("disk full")

    def close(self) -> None:
        pass


def statuses(sink: ListSink) -> list[str]:
    return [
        e.value for e in sink.events if e.kind == "status" and isinstance(e.value, str)
    ]


def test_router_is_an_output_sink_and_broadcasts_in_order() -> None:
    log = []
    first, second = ListSink(log), ListSink(log)
    router = OutputRouter(first, second)
    assert isinstance(router, OutputSink)
    router.write(RunEvent("scalar", "a", 1.0))
    router.write(RunEvent("scalar", "b", 2.0))
    assert log == [(first, "a"), (second, "a"), (first, "b"), (second, "b")]
    router.close()
    assert first.closed and second.closed


def test_normal_exit_records_completed_and_closes(sink: ListSink) -> None:
    with OutputRouter(sink) as router:
        router.write(RunEvent("scalar", "train/loss/total", 1.0))
    assert statuses(sink) == ["completed"]
    assert sink.closed


def test_exception_records_traceback_and_failed_then_propagates(sink: ListSink) -> None:
    with pytest.raises(RuntimeError, match="boom"), OutputRouter(sink):
        raise RuntimeError("boom")
    (message,) = [e for e in sink.events if e.kind == "message"]
    assert message.name == "traceback"
    assert isinstance(message.value, str)
    assert "RuntimeError: boom" in message.value
    assert statuses(sink) == ["failed"]
    assert sink.closed


def test_keyboard_interrupt_records_interrupted(sink: ListSink) -> None:
    with pytest.raises(KeyboardInterrupt), OutputRouter(sink):
        raise KeyboardInterrupt
    assert statuses(sink) == ["interrupted"]
    assert sink.closed


def test_failing_sink_raises_sink_error_naming_sink_and_event() -> None:
    router = OutputRouter(BrokenSink())
    with pytest.raises(SinkError, match="BrokenSink.*'train/loss/total'.*disk full"):
        router.write(RunEvent("scalar", "train/loss/total", 1.0))


def test_run_directory_after_a_failed_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError), OutputRouter(LocalFileSink(tmp_path)) as router:
        router.write(RunEvent("scalar", "train/loss/total", 1.0, step=0))
        raise ValueError("loss is nan")
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "failed"
    assert "loss is nan" in status["traceback"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary == {"train/loss/total": 1.0}
