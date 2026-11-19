import platform
import subprocess
from pathlib import Path

import torch
import torchvision

# Get the absolute path to the project directory
PACKAGE_DIR = Path(__file__).resolve().parents[1]


def environment(repo_dir: Path = PACKAGE_DIR) -> dict[str, object]:
    """Versions, machine and git state to record with every run.

    Git is read from the code's own directory, not the working directory, so the
    revision is the code that ran. Outside a git checkout the git fields are
    `None` rather than an error.

    The --porcelain argument outputs the git status in a program readable format
    """
    git_sha = _git(repo_dir, "rev-parse", "HEAD")
    git_changes = None if git_sha is None else _git(repo_dir, "status", "--porcelain")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "git_sha": git_sha,
        "git_dirty": None if git_changes is None else bool(git_changes),
    }


def _git(repo_dir: Path, *git_args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *git_args],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()
