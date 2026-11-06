"""Redo only the component findings of runs: uv run python scripts/dev/refresh_components.py runs/<name> ..."""
import json
import shutil
import sys
import time
from pathlib import Path

import torch  # noqa: F401

from chipscan.stages.flow import refresh_components

if __name__ == "__main__":
    for r in sys.argv[1:]:
        t = time.time()
        bak = Path(r) / "findings.before_components2.json"
        if not bak.exists():
            shutil.copy(Path(r) / "findings.json", bak)
        o = refresh_components(r)
        print(r, json.dumps(o["summary"].get("component")), json.dumps(o["components"]), f"{time.time() - t:.0f}s", flush=True)
