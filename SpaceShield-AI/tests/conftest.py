"""
conftest.py
============
Shared pytest configuration and fixtures for SpaceShield AI test suite.
"""

import sys
import os

# Ensure project root is on path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
