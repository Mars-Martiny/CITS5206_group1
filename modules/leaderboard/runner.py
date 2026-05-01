"""Get the runner's identity and resolve it."""

import socket
import subprocess
from pathlib import Path


def resolve_runner(leaderboard_dir: Path) -> str:
    """Get the runner's identity and resolve it.

    Priority chain:
    1. leaderboard/owner.conf (cached value)
    2. git config user.name  (or user.email as fallback)
    3. socket.gethostname()

    On the first successful git resolution the result is cached to owner.conf
    so subsequent runs skip the subprocess call.

    Args:
        leaderboard_dir: directory for leaderboard

    Returns:
        String for the runner's identity
    """
    conf_path = leaderboard_dir / "owner.conf"

    if conf_path.exists():
        cached = conf_path.read_text(encoding="utf-8").strip()
        if cached:
            return cached

    name = _git_user()
    if name:
        leaderboard_dir.mkdir(parents=True, exist_ok=True)
        conf_path.write_text(name, encoding="utf-8")
        return name

    return socket.gethostname()


def _git_user() -> str:
    """Get the git user utilizing subprocess to call git config.

    Returns:
        String for either user.name or user.email 
    """
    for field in ("user.name", "user.email"):
        try:
            result = subprocess.run(
                ["git", "config", field],
                capture_output=True,
                text=True,
                timeout=5,
            )
            value = result.stdout.strip()
            if value:
                return value
        except Exception:
            pass
    return ""
