#!/usr/bin/env python3
"""Run the CodeSage FastAPI backend server."""

import argparse
import sys
from pathlib import Path

# Add 'src' to python search path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the CodeSage FastAPI server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    args = parser.parse_args()

    print(f"Starting CodeSage backend API on http://{args.host}:{args.port}...")
    uvicorn.run("codesage.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
