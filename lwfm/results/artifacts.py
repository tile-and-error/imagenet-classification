import os
from collections.abc import Callable
from pathlib import Path

from lwfm.results.events import RunEvent
from lwfm.results.sinks import OutputSink


def save_artifact(
    run_dir: str | Path,
    relative_path: str | Path,
    write: Callable[[Path], object],
    output_sink: OutputSink | None,
    *,
    name: str,
    step: int | None = None,
) -> Path:
    """Saves a file into the run directory and records it as an `artifact` event.

    `write(path)` writes the file, e.g. `figure.savefig` or
    `lambda path: torch.save(weights, path)`. The event holds `relative_path`, so
    the run directory can be moved without breaking its records. Returns the full
    path of the saved file.
    """
    path = Path(run_dir) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same suffix as the target, since savefig picks the format from it
    temporary = path.with_stem(path.stem + ".tmp")
    try:
        write(temporary)
        os.replace(temporary, path)  # atomic: a killed job leaves the old file intact
    finally:
        temporary.unlink(missing_ok=True)
    # File first, then the event, so the record never points at a missing file
    if output_sink is not None:
        output_sink.write(RunEvent("artifact", name, Path(relative_path), step=step))
    return path
