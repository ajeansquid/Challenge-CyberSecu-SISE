#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Challenge Toolkit v2 - Application Launcher

Usage:
    python run.py              # Run the Streamlit app
    python run.py --port 8502  # Run on specific port
    python run.py --help       # Show help
"""

import sys
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="CyberSec ML Toolkit - Application Launcher"
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8501,
        help='Port to run the application on (default: 8501)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host to bind to (default: localhost)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )

    args = parser.parse_args()

    # Get the directory containing this script
    app_dir = Path(__file__).parent.absolute()
    main_file = app_dir / "main.py"

    if not main_file.exists():
        print(f"Error: main.py not found at {main_file}")
        sys.exit(1)

    # Build streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(main_file),
        f"--server.port={args.port}",
        f"--server.address={args.host}",
    ]

    if args.debug:
        cmd.append("--logger.level=debug")

    print(f"Starting CyberSec ML Toolkit on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print("-" * 50)

    try:
        subprocess.run(cmd, cwd=str(app_dir))
    except KeyboardInterrupt:
        print("\nApplication stopped.")


if __name__ == "__main__":
    main()
