"""Make the repository root importable, so tests can reach the examples.

The example modules double as fixtures for behavioral tests; they live outside
the installed ``fdg`` package, so pytest's default path insertion (which only
adds the test directory) cannot see them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
