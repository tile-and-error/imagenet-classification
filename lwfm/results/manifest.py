import platform
import subprocess
from pathlib import Path

import torch
import torchvision

PACKAGE_DIR = Path(__file__).resolve().parents[1]


def environment(repo: Path = PACKAGE_DIR) -> dict[str, object]:
    """Versions, machine and git state to record with every run.

    Git is read from the code's own directory, not the working directory, so the
    revision is the code that ran. Outside a git checkout the git fields are
    `None` rather than an error.
    """
    sha = _git(repo, "rev-parse", "HEAD")
    status = None if sha is None else _git(repo, "status", "--porcelain")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "git_sha": sha,
        "git_dirty": None if status is None else bool(status),
    }


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()
