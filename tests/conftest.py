"""Pytest configuration and fixtures."""

import os
import sys

# Add parent directory to path so tests can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
