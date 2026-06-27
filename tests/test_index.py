"""Full-coverage unit tests for index.py (the FastMCP tool layer).

FastMCP's @mcp.tool() returns the original function, so the tools are called
directly. Engines are replaced with fakes; OCR backends are never invoked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engines as eng  # noqa: E402
import index as ix  # noqa: E402


class FakeEngine:
    def __init__(self, name="rapidocr", avail=(True, "ok"), result=None):
        self.name = name
        self._avail = avail
        self._result = result if result is not None else eng.OcrResult(engine=name, text="hello")

    def available(self):
        return self._avail

    def ocr_image(self, path, lang="en"):
        return self._result


@pytest.fixture
def fake_engines(monkeypatch):
    engines = {
        "rapidocr": FakeEngine("rapidocr", result=eng.OcrResult(engine="rapidocr", text="hello", mean_confidence=0.9)),
        "tesseract": FakeEngine("tesseract", avail=(False, "missing")),
    }
    monkeypatch.setattr(ix, "ENGINES", engines)
    return engines


# ----------------------------------------------------------- _check_path ------
def test_check_path_no_allowlist(monkeypatch):
    monkeypatch.setattr(ix, "_ALLOWED", [])
    assert ix._check_path("/anything") is None


def test_check_path_allowed(monkeypatch, tmp_path):
    monkeypatch.setattr(ix, "_ALLOWED", [tmp_path.resolve()])
    f = tmp_path / "x.png"
    f.write_text("x")
    assert ix._check_path(str(f)) is None


def test_check_path_blocked(monkeypatch, tmp_path):
    monkeypatch.setattr(ix, "_ALLOWED", [tmp_path.resolve()])
    assert "outside" in ix._check_path("C:/elsewhere/x.png")


def test_check_path_invalid(monkeypatch):
    monkeypatch.setattr(ix, "_ALLOWED", [Path("C:/a").resolve()])
    assert ix._check_path("bad\x00path").startswith("invalid path")


# --------------------------------------------------------- _resolve_engine ----
def test_resolve_engine_default(fake_engines):
    assert ix._resolve_engine(None) == "rapidocr"
    assert ix._resolve_engine("auto") == "rapidocr"


def test_resolve_engine_known(fake_engines):
    assert ix._resolve_engine("tesseract") == "tesseract"


def test_resolve_engine_unknown(fake_engines):
    with pytest.raises(ValueError, match="unknown engine"):
        ix._resolve_engine("nope")


# ------------------------------------------------------------ list_engines ----
def test_list_engines(fake_engines):
    out = json.loads(ix.list_engines())
    assert out["default"] == "rapidocr"
    assert out["engines"]["rapidocr"]["available"] is True
    assert out["engines"]["tesseract"]["available"] is False


# -------------------------------------------------------------- ocr_image -----
def test_ocr_image_blocked(monkeypatch, tmp_path):
    monkeypatch.setattr(ix, "_ALLOWED", [tmp_path.resolve()])
    out = json.loads(ix.ocr_image("C:/other/x.png"))
    assert "error" in out


def test_ocr_image_bad_engine(fake_engines, monkeypatch):
    monkeypatch.setattr(ix, "_ALLOWED", [])
    out = json.loads(ix.ocr_image("x.png", engine="nope"))
    assert "unknown engine" in out["error"]


def test_ocr_image_success_no_preprocess(fake_engines, monkeypatch):
    monkeypatch.setattr(ix, "_ALLOWED", [])
    out = json.loads(ix.ocr_image("x.png"))
    assert out["text"] == "hello"
    assert "preprocessing" not in out


def test_ocr_image_with_preprocess(fake_engines, monkeypatch):
    monkeypatch.setattr(ix, "_ALLOWED", [])
    monkeypatch.setattr(ix._eval, "preprocess", lambda p: ("proc.png", ["grayscale"]))
    out = json.loads(ix.ocr_image("x.png", preprocess=True))
    assert out["preprocessing"] == ["grayscale"]


# --------------------------------------------------------------- ocr_pdf ------
def test_ocr_pdf_blocked(monkeypatch):
    monkeypatch.setattr(ix, "_ALLOWED", [Path("C:/allowed").resolve()])
    out = json.loads(ix.ocr_pdf("C:/other/x.pdf"))
    assert "error" in out


def test_ocr_pdf_bad_engine(fake_engines, monkeypatch):
    monkeypatch.setattr(ix, "_ALLOWED", [])
    out = json.loads(ix.ocr_pdf("x.pdf", engine="nope"))
    assert "unknown engine" in out["error"]


def test_ocr_pdf_not_a_file(fake_engines, monkeypatch):
    monkeypatch.setattr(ix, "_ALLOWED", [])
    out = json.loads(ix.ocr_pdf("missing.pdf"))
    assert out["error"].startswith("file not found")


def test_ocr_pdf_rasterize_fails(fake_engines, monkeypatch, tmp_path):
    pdf = tmp_path / "d.pdf"
    pdf.write_text("x")
    monkeypatch.setattr(ix, "_ALLOWED", [])
    monkeypatch.setattr(ix._eval, "pdf_to_images", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = json.loads(ix.ocr_pdf(str(pdf)))
    assert out["error"].startswith("pdf rasterization failed")


def test_ocr_pdf_success_and_cleanup(fake_engines, monkeypatch, tmp_path):
    pdf = tmp_path / "d.pdf"
    pdf.write_text("x")
    real_img = tmp_path / "p1.png"
    real_img.write_text("img")
    monkeypatch.setattr(ix, "_ALLOWED", [])
    # one removable file + one ghost -> exercises both finally branches
    monkeypatch.setattr(ix._eval, "pdf_to_images", lambda *a, **k: ([str(real_img), "ghost.png"], ["note"]))
    out = json.loads(ix.ocr_pdf(str(pdf)))
    assert out["page_count"] == 2
    assert out["pages"][0]["page"] == 1
    assert not real_img.exists()  # cleaned up


# ------------------------------------------------------------- batch_ocr ------
def test_batch_ocr_bad_engine(fake_engines):
    out = json.loads(ix.batch_ocr("*.png", engine="nope"))
    assert "unknown engine" in out["error"]


def test_batch_ocr_bad_json(fake_engines):
    out = json.loads(ix.batch_ocr("[not json"))
    assert out["error"].startswith("bad JSON path list")


def test_batch_ocr_no_match(fake_engines):
    out = json.loads(ix.batch_ocr("C:/nonexistent_dir_zzz/*.png"))
    assert out["error"].startswith("no files matched")


def test_batch_ocr_json_list_success(fake_engines, monkeypatch):
    monkeypatch.setattr(ix, "_ALLOWED", [])
    out = json.loads(ix.batch_ocr('["a.png", "b.png"]'))
    assert out["count"] == 2
    assert out["results"][0]["text"] == "hello"


def test_batch_ocr_glob_success(fake_engines, monkeypatch, tmp_path):
    (tmp_path / "one.png").write_text("x")
    monkeypatch.setattr(ix, "_ALLOWED", [])
    out = json.loads(ix.batch_ocr(str(tmp_path / "*.png")))
    assert out["count"] == 1


def test_batch_ocr_allowlist_blocks(fake_engines, monkeypatch, tmp_path):
    monkeypatch.setattr(ix, "_ALLOWED", [tmp_path.resolve()])
    out = json.loads(ix.batch_ocr('["C:/evil/a.png"]'))
    assert "outside" in out["error"]
    assert out["examples"]


def test_batch_ocr_allowlist_passes(fake_engines, monkeypatch, tmp_path):
    f = tmp_path / "ok.png"
    f.write_text("x")
    monkeypatch.setattr(ix, "_ALLOWED", [tmp_path.resolve()])
    out = json.loads(ix.batch_ocr(json.dumps([str(f)])))
    assert out["count"] == 1


# ---------------------------------------------------------- compare_engines ---
def test_compare_engines(monkeypatch):
    engines = {
        "rapidocr": FakeEngine("rapidocr", result=eng.OcrResult(engine="rapidocr", text="hello world")),
        "tesseract": FakeEngine("tesseract", result=eng.OcrResult(engine="tesseract", text="hello word")),
        "empty": FakeEngine("empty", result=eng.OcrResult(engine="empty", text="")),
        "broken": FakeEngine("broken", result=eng.OcrResult(engine="broken", error="x")),
        "unavail": FakeEngine("unavail", avail=(False, "no")),
    }
    monkeypatch.setattr(ix, "ENGINES", engines)
    out = json.loads(ix.compare_engines("x.png"))
    assert out["engines"]["unavail"] == {"available": False}
    assert out["engines"]["rapidocr"]["ok"] is True
    assert out["engines"]["broken"]["ok"] is False
    assert "consensus_engine" in out["agreement"]


# -------------------------------------------------------- evaluate_accuracy ---
def test_evaluate_accuracy_blocked(monkeypatch, tmp_path):
    monkeypatch.setattr(ix, "_ALLOWED", [tmp_path.resolve()])
    out = json.loads(ix.evaluate_accuracy("C:/evil/gt.txt", ocr_text="hi"))
    assert "error" in out


def test_evaluate_accuracy_gt_missing(fake_engines, monkeypatch):
    monkeypatch.setattr(ix, "_ALLOWED", [])
    out = json.loads(ix.evaluate_accuracy("missing_gt.txt", ocr_text="hi"))
    assert out["error"].startswith("ground truth not found")


def test_evaluate_accuracy_provided_text(fake_engines, monkeypatch, tmp_path):
    gt = tmp_path / "gt.txt"
    gt.write_text("hello world", encoding="utf-8")
    monkeypatch.setattr(ix, "_ALLOWED", [])
    out = json.loads(ix.evaluate_accuracy(str(gt), ocr_text="hello world"))
    assert out["cer"] == 0.0
    assert out["evaluated"] == {"source": "provided_text"}


def test_evaluate_accuracy_no_input(fake_engines, monkeypatch, tmp_path):
    gt = tmp_path / "gt.txt"
    gt.write_text("ref", encoding="utf-8")
    monkeypatch.setattr(ix, "_ALLOWED", [])
    out = json.loads(ix.evaluate_accuracy(str(gt)))
    assert out["error"] == "provide ocr_text or ocr_path"


def test_evaluate_accuracy_ocr_bad_engine(fake_engines, monkeypatch, tmp_path):
    gt = tmp_path / "gt.txt"
    gt.write_text("ref", encoding="utf-8")
    monkeypatch.setattr(ix, "_ALLOWED", [])
    out = json.loads(ix.evaluate_accuracy(str(gt), ocr_path="x.png", engine="nope"))
    assert "unknown engine" in out["error"]


def test_evaluate_accuracy_ocr_path_missing(fake_engines, monkeypatch, tmp_path):
    gt = tmp_path / "gt.txt"
    gt.write_text("ref", encoding="utf-8")
    monkeypatch.setattr(ix, "_ALLOWED", [])
    out = json.loads(ix.evaluate_accuracy(str(gt), ocr_path="missing.png"))
    assert out["error"].startswith("ocr_path not found")


def test_evaluate_accuracy_ocr_image(fake_engines, monkeypatch, tmp_path):
    gt = tmp_path / "gt.txt"
    gt.write_text("hello", encoding="utf-8")
    img = tmp_path / "i.png"
    img.write_text("x")
    monkeypatch.setattr(ix, "_ALLOWED", [])
    out = json.loads(ix.evaluate_accuracy(str(gt), ocr_path=str(img)))
    assert out["evaluated"] == {"source": "ocr", "engine": "rapidocr"}


def test_evaluate_accuracy_ocr_pdf(fake_engines, monkeypatch, tmp_path):
    gt = tmp_path / "gt.txt"
    gt.write_text("hello", encoding="utf-8")
    pdf = tmp_path / "i.pdf"
    pdf.write_text("x")
    monkeypatch.setattr(ix, "_ALLOWED", [])
    monkeypatch.setattr(ix._eval, "pdf_to_images", lambda *a, **k: (["p1.png", "p2.png"], []))
    out = json.loads(ix.evaluate_accuracy(str(gt), ocr_path=str(pdf)))
    assert out["evaluated"]["source"] == "ocr"
