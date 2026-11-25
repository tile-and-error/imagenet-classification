"""Reads back the run folders LocalFileSink writes (with file_format="json")."""

import json
import shutil
from pathlib import Path

from lwfm.results.events import RunStatus


def run_status(run_dir: str | Path) -> RunStatus | None:
    """The status a run folder records, or None if it holds no run."""
    status_file = Path(run_dir) / "status.json"
    if not status_file.exists():
        return None
    return json.loads(status_file.read_text())["status"]


def completed_runs(root: str | Path, pattern: str = "**") -> list[dict]:
    """One dict per completed run under root/<pattern>: its manifest, its final
    values (summary) and its path relative to root, as "run"."""
    root = Path(root)
    rows = []
    for status_file in sorted(root.glob(f"{pattern}/status.json")):
        run_dir = status_file.parent
        if run_status(run_dir) != "completed":
            continue
        rows.append(
            json.loads((run_dir / "manifest.json").read_text())
            | json.loads((run_dir / "summary.json").read_text())
            | {"run": run_dir.relative_to(root).as_posix()}
        )
    return rows


def skip_if_completed(run_dir: str | Path) -> bool:
    """True if run_dir holds a completed run, so the caller can skip it."""
    status = run_status(run_dir)
    if status is None:  # no run yet
        return False
    if status == "completed":
        return True
    if status in ("interrupted", "failed"):  # can't resume partway: start again
        shutil.rmtree(run_dir)
        return False
    # "running": the folder may belong to a job that is still going
    raise FileExistsError(
        f"{run_dir} is marked {status}; if no job is using it, delete it"
    )
