import sqlite3

from chipscan.web.server import ReviewDB


def test_merge_from_keeps_every_review(tmp_path):
    (tmp_path / "scanA").mkdir()
    a = ReviewDB(tmp_path / "scanA" / "reviews.sqlite")
    a.set("scanA", 10.0, 20.0, "real", None, None, "me")
    b_dir = tmp_path / "scanB"
    b_dir.mkdir()
    b = ReviewDB(b_dir / "reviews.sqlite")
    b.set("scanB", 5.0, 5.0, "false_alarm", None, None, "me")
    main = ReviewDB(tmp_path / "reviews.sqlite")
    assert main.merge_from(a.path) == 1 and main.merge_from(b.path) == 1
    assert main.merge_from(a.path) == 0  # merging twice adds nothing
    assert [p["status"] for p in main.positions("scanA")] == ["real"]
    assert [p["status"] for p in main.positions("scanB")] == ["false_alarm"]
    with sqlite3.connect(str(a.path)) as c:  # the old file is left as it was
        assert c.execute("SELECT count(*) FROM review_pos").fetchone()[0] == 1
