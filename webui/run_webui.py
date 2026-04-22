#!/usr/bin/env python3
"""
Launcher for the KABOCS web UI backend.

From the project root run::

    python webui/run_webui.py
    python webui/run_webui.py --host 0.0.0.0 --port 8080 --reload

This ensures that the ``kabo`` package from the current repository is
importable without installation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_root_on_syspath() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the KABOCS web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")
    args = parser.parse_args()

    _ensure_root_on_syspath()

    import uvicorn

    uvicorn.run(
        "webui.backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
