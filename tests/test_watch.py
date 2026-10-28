import time

import numpy as np
from PIL import Image

from chipscan.web.server import Watcher


class FakeJobs:
    def __init__(self):
        self.specs = []

    def submit(self, spec, resume=False):
        self.specs.append(spec)


def test_watcher_queues_new_images_once_settled(tmp_path):
    folder = tmp_path / "exports"
    folder.mkdir()
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(folder / "old.png")  # there before watching
    jobs = FakeJobs()
    w = Watcher(tmp_path / "runs", jobs, poll_s=0.1, settle_s=0.2)
    w.add(str(folder), pixel_size_um=0.814)
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(folder / "new_scan.png")
    deadline = time.time() + 5
    while not jobs.specs and time.time() < deadline:
        time.sleep(0.1)
    assert [s.name for s in jobs.specs] == ["new_scan"]
    assert jobs.specs[0].pixel_size_um == 0.814
    time.sleep(0.5)
    assert len(jobs.specs) == 1  # not queued twice, old file ignored
    assert w.list()[0]["queued"][0]["name"] == "new_scan"
