"""Make the top-level packages importable during `pytest` regardless of install method.

Redundant with `pip install -e .`, but keeps `pytest` working from a plain
`pip install -r requirements.txt` environment too.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
