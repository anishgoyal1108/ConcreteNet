#!/usr/bin/env python3
"""Launcher for ConcreteNet desktop application."""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import main

if __name__ == "__main__":
    sys.exit(main())
