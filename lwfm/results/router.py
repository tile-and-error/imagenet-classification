import traceback
from collections.abc import Iterable
from types import TracebackType

from lwfm.results.events import EventSink, RunEvent, RunStatus
from lwfm.results.sinks import OutputSink, SinkError


class OutputRouter(EventSink):
    """Sends every event to every sink, in order, and manages the run's status.

    Use it as a context manager around a run: a normal exit records `completed`,
    `KeyboardInterrupt` records `interrupted`, and any other exception records the
    traceback and `failed` before it propagates. Every case then closes the
    sinks. A sink that fails raises `SinkError` naming the sink and the event.
    """

    def __init__(self, sinks: Iterable[OutputSink]):
        self.sinks = list(sinks)

    def emit(self, event: RunEvent) -> None:
        for sink in self.sinks:
            try:
                sink.write(event)
            except Exception as error:
                raise SinkError(
                    f"{type(sink).__name__} failed to write event {event.name!r}: "
                    f"{error}"
                ) from error

    def close(self) -> None:
        failed: list[str] = []
        for sink in self.sinks:  # close every sink, even after one fails
            try:
                sink.close()
            except Exception as error:
                failed.append(f"{type(sink).__name__}: {error}")
        if failed:
            raise SinkError(f"failed to close sinks: {'; '.join(failed)}")

    def __enter__(self) -> "OutputRouter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        status: RunStatus
        try:
            if exc_type is None:
                status = "completed"
            elif issubclass(exc_type, KeyboardInterrupt):
                status = "interrupted"
            else:
                status = "failed"
                text = "".join(traceback.format_exception(exc_type, exc, tb))
                self.emit(RunEvent("message", "traceback", text))
            self.emit(RunEvent("status", "status", status))
        finally:
            self.close()
