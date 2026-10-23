import base64
import io
import json
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import import_test_photos as imp  # noqa: E402


def data_url(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "gray").save(buf, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def export(tmp_path):
    photos = [
        {"id": "aaa.1", "ts": 1, "pageUrl": "https://www.facebook.com/m/1", "host": "www.facebook.com",
         "photo": data_url(800, 600), "boxes": [[0, 0, 50, 99], [10, 10, 60, 160], [0, 0, 80, 250]]},
        {"id": "bbb.2", "ts": 2, "pageUrl": None, "host": None,
         "photo": data_url(400, 400), "boxes": [[0, 0, 90, 199], [0, 0, 90, 300]]},
    ]
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"exportedAt": "x", "extVersion": "2.1.0", "photos": photos}))
    return path


def test_import_outputs_and_buckets(export, tmp_path, capsys):
    out = tmp_path / "out"
    assert imp.main([str(export), "--out", str(out)]) == 0
    assert Image.open(out / "aaa.1.jpg").size == (800, 600)
    index = json.loads((out / "boxes.json").read_text())
    assert index["aaa.1"]["box_heights"] == [99, 150, 250]
    assert index["aaa.1"]["width"] == 800 and index["bbb.2"]["host"] is None
    assert (out / "sheets" / "sheet-1.jpg").exists()
    summary = imp.summarize(index)
    assert summary["photos"] == 2 and summary["boxes"] == 5
    assert summary["height_buckets"] == {"<100": 1, "100-149": 0, "150-199": 2, "200-299": 1, ">=300": 1}
    assert summary["hosts"] == {"www.facebook.com": 1, "(none)": 1}
    assert "photos: 2" in capsys.readouterr().out


def test_refuses_overwrite_without_force(export, tmp_path):
    out = tmp_path / "out"
    assert imp.main([str(export), "--out", str(out)]) == 0
    assert imp.main([str(export), "--out", str(out)]) == 1
    assert imp.main([str(export), "--out", str(out), "--force"]) == 0
