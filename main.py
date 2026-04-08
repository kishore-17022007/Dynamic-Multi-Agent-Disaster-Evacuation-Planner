from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launcher for Dynamic Multi-Agent Disaster Evacuation Planner"
    )
    parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit server")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for Streamlit server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Streamlit in headless mode",
    )
    args = parser.parse_args()

    app_path = pathlib.Path(__file__).with_name("app.py")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
        "--server.headless",
        "true" if args.headless else "false",
    ]

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
