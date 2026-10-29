"""Redo findings (components, extraction, risk) from a run's stored maps: uv run python scripts/dev/reprocess.py runs/<name> ..."""
import json
import sys
import time

import torch  # noqa: F401

from chipscan.stages.flow import reprocess

if __name__ == "__main__":
    for r in sys.argv[1:]:
        t = time.time()
        o = reprocess(r)
        print(r, json.dumps(o["summary"]), f"{time.time() - t:.0f}s", flush=True)
