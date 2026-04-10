"""Executable entrypoint for launching the Streamlit UI."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    app_path = repo_root / "src" / "app.py"

    # Delegate to Streamlit's CLI so PyInstaller can expose a single app binary.
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
    ]

    from streamlit.web import cli as stcli

    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
