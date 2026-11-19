import traceback
from types import TracebackType

from lwfm.results.events import RunEvent, RunStatus
from lwfm.results.sinks import OutputSink, SinkError


class OutputRouter(OutputSink):
    """Sends every event to every sink, in order, and manages the run's status.

    Use it as a context manager around a run: a normal exit records `completed`,
    `KeyboardInterrupt` records `interrupted`, and any other exception records the
    traceback and `failed` before it propagates. Every case then closes the
    sinks. A sink that fails raises `SinkError` naming the sink and the event.
    """

    def __init__(self, *output_sinks: OutputSink):
        self.output_sinks = list(output_sinks)

    def write(self, event: RunEvent) -> None:
        """Write an event to all output sinks"""
        for sink in self.output_sinks:
            try:
                sink.write(event)
            except Exception as error:
                raise SinkError(
                    f"{type(sink).__name__} failed to write event {event.name!r}: "
                    f"{error}"
                ) from error

    def close(self) -> None:
        """Close all sinks when no longer needed"""
        close_failures: list[str] = []
        for sink in self.output_sinks:  # close every sink, even after one fails
            try:
                sink.close()
            except Exception as error:
                close_failures.append(f"{type(sink).__name__}: {error}")
        if close_failures:
            raise SinkError(f"failed to close sinks: {'; '.join(close_failures)}")

    def __enter__(self) -> "OutputRouter":
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        exception_traceback: TracebackType | None,
    ) -> None:
        """
        This method is called when used with a `with` statement.
        It executes when Python leaves the `with` block.
        
        This records the final status in each of the sinks.
        """
        final_status: RunStatus
        try:
            if exception_type is None:
                final_status = "completed"
            elif issubclass(exception_type, KeyboardInterrupt):
                final_status = "interrupted"
            else:
                final_status = "failed"
                traceback_text = "".join(
                    traceback.format_exception(
                        exception_type, exception, exception_traceback
                    )
                )
                self.write(RunEvent("message", "traceback", traceback_text))
            self.write(RunEvent("status", "status", final_status))
        finally:
            self.close()
