"""Tests for AutoDataMining Interactive Web Dashboard UI."""

from pathlib import Path
import py_compile
from unittest.mock import patch

import pytest

from autodatamining.ui import get_app_path, launch_ui


def test_app_file_compilation():
    """Verify that app.py has valid Python syntax and compiles cleanly."""
    app_path = get_app_path()
    assert app_path.is_file()
    assert app_path.name == "app.py"
    # Test bytecode compilation
    compiled = py_compile.compile(str(app_path), doraise=True)
    assert compiled is not None


def test_get_app_path_not_found(monkeypatch, tmp_path):
    """Verify FileNotFoundError when app.py is missing from search paths."""
    monkeypatch.chdir(tmp_path)
    with patch("autodatamining.ui.Path.cwd", return_value=tmp_path):
        # Even if cwd is empty, workspace parent might still find it, so mock resolve
        with patch("autodatamining.ui.Path.is_file", return_value=False):
            with pytest.raises(FileNotFoundError):
                get_app_path()


def test_launch_ui_command_construction():
    """Verify that launch_ui constructs the correct subprocess command."""
    with patch("subprocess.run") as mock_run:
        launch_ui(host="127.0.0.1", port=9999, browser=False)
        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert "-m" in cmd
        assert "streamlit" in cmd
        assert "run" in cmd
        assert "--server.address=127.0.0.1" in cmd
        assert "--server.port=9999" in cmd
        assert "--server.headless=true" in cmd
