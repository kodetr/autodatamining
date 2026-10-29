"""Launcher for the AutoDataMining Interactive Web Dashboard."""

import os
from pathlib import Path
import subprocess
import sys


def get_app_path() -> Path:
    """Return the absolute path to the app.py dashboard file."""
    # Look for app.py in root or alongside package
    current = Path(__file__).resolve()
    # Check parent workspace roots
    for candidate in [
        current.parent.parent.parent / "app.py",
        current.parent / "app.py",
        Path.cwd() / "app.py",
    ]:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("app.py dashboard file not found.")


def launch_ui(host: str = "127.0.0.1", port: int = 8501, browser: bool = True):
    """Launch the interactive Streamlit Web Dashboard.

    Parameters
    ----------
    host : str, default="127.0.0.1"
        Host IP address to bind server to.
    port : int, default=8501
        Port number for the web server.
    browser : bool, default=True
        Whether to open the browser automatically.
    """
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Antarmuka web membutuhkan pustaka `streamlit`. "
            "Silakan pasang terlebih dahulu dengan perintah: pip install \"autodatamining[ui]\""
        ) from exc

    app_path = get_app_path()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        f"--server.address={host}",
        f"--server.port={port}",
    ]
    if not browser:
        cmd.append("--server.headless=true")

    print(f"🚀 Membuka AutoDataMining Web Studio di http://{host}:{port} ...")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    launch_ui()
