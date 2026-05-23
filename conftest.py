"""Make the `iemcs` package importable in tests without installation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
