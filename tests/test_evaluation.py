"""Full-coverage unit tests for evaluation.py (real jiwer/opencv/pymupdf)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import fitz
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluation as ev  # noqa: E402


# ---------------------------------------------------------------- evaluate ----
def test_evaluate_jiwer_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "jiwer", None)
    out = ev.evaluate("a", "b")
    assert out == {"error": "jiwer not installed: pip install jiwer"}


def test_evaluate_perfect_match():
    out = ev.evaluate("hello world", "hello world")
    assert out["cer"] == 0.0
    assert out["wer"] == 0.0
    assert out["char_accuracy_pct"] == 100.0
    assert out["word_accuracy_pct"] == 100.0
    assert out["ref_word_count"] == 2
    assert out["hyp_word_count"] == 2
    assert out["hits"] == 2


def test_evaluate_with_errors_and_empty_defaults():
    # empty hypothesis exercises the `hypothesis or ""` falsy branch
    out = ev.evaluate("", "reference text here")
    assert out["cer"] > 0.0
    assert out["wer"] == 1.0
    assert out["hyp_word_count"] == 0


# -------------------------------------------------------------- similarity ----
def test_similarity_identical():
    assert ev.similarity("abc", "abc") == 1.0


def test_similarity_empty_defaults():
    assert ev.similarity("", "") == 1.0
    assert ev.similarity("", "x") == 0.0


# ---------------------------------------------------------- cross_agreement ----
def test_cross_agreement_multi():
    out = ev.cross_agreement({"a": "hello", "b": "hello", "c": "world", "skip": ""})
    assert "a__vs__b" in out["pairwise_similarity"]
    assert out["consensus_engine"] in {"a", "b"}
    assert out["avg_similarity"]["a"] >= 0.0


def test_cross_agreement_single_engine_zero_avg():
    out = ev.cross_agreement({"only": "text"})
    assert out["avg_similarity"]["only"] == 0.0
    assert out["consensus_engine"] == "only"


def test_cross_agreement_empty():
    out = ev.cross_agreement({"a": "", "b": ""})
    assert out["consensus_engine"] is None
    assert out["pairwise_similarity"] == {}


# --------------------------------------------------------------- preprocess ----
def _write_png(tmp_path: Path, arr: np.ndarray) -> str:
    p = tmp_path / "in.png"
    cv2.imwrite(str(p), arr)
    return str(p)


def test_preprocess_opencv_missing(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setitem(sys.modules, "numpy", None)
    path, notes = ev.preprocess("nonexistent.png")
    assert notes == ["preprocessing skipped (opencv unavailable)"]
    assert path == "nonexistent.png"


def test_preprocess_unreadable_image(tmp_path):
    bad = tmp_path / "not_an_image.png"
    bad.write_text("garbage")
    path, notes = ev.preprocess(str(bad))
    assert notes == ["preprocessing skipped (could not read image)"]
    assert path == str(bad)


def test_preprocess_full_pipeline_with_deskew(tmp_path):
    # white canvas with a slanted black bar -> exercises grayscale+denoise+deskew
    img = np.full((200, 400, 3), 255, np.uint8)
    cv2.line(img, (40, 60), (360, 150), (0, 0, 0), 8)
    src = _write_png(tmp_path, img)
    out, notes = ev.preprocess(src, grayscale=True, denoise=True, deskew=True)
    assert "grayscale" in notes
    assert "denoise" in notes
    assert Path(out).is_file()


def test_preprocess_color_denoise_no_deskew(tmp_path):
    img = np.full((80, 120, 3), 200, np.uint8)
    src = _write_png(tmp_path, img)
    out, notes = ev.preprocess(src, grayscale=False, denoise=True, deskew=True)
    # color path: ndim==3 so deskew is skipped
    assert "denoise" in notes
    assert "grayscale" not in notes
    assert Path(out).is_file()


def test_preprocess_deskew_too_few_dark_pixels(tmp_path):
    # mostly-white grayscale-converted image: <=50 dark coords -> deskew skipped
    img = np.full((60, 60, 3), 255, np.uint8)
    img[0, 0] = (0, 0, 0)
    src = _write_png(tmp_path, img)
    out, notes = ev.preprocess(src, grayscale=True, denoise=False, deskew=True)
    assert "grayscale" in notes
    assert not any(n.startswith("deskew") for n in notes)


def test_preprocess_deskew_angle_below_threshold(monkeypatch, tmp_path):
    # many dark pixels (enters deskew) but a ~0 angle -> warpAffine skipped
    img = np.full((120, 120, 3), 255, np.uint8)
    cv2.rectangle(img, (20, 20), (100, 100), (0, 0, 0), -1)
    src = _write_png(tmp_path, img)
    monkeypatch.setattr(cv2, "minAreaRect", lambda coords: ((0.0, 0.0), (10.0, 10.0), 0.0))
    out, notes = ev.preprocess(src, grayscale=True, denoise=False, deskew=True)
    assert not any(n.startswith("deskew") for n in notes)
    assert Path(out).is_file()


def test_preprocess_write_failure(monkeypatch, tmp_path):
    img = np.full((40, 40, 3), 255, np.uint8)
    src = _write_png(tmp_path, img)
    monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out, notes = ev.preprocess(src, grayscale=False, denoise=False, deskew=False)
    assert out == src
    assert notes[0].startswith("preprocessing skipped (error:")


# ------------------------------------------------------------ pdf_to_images ----
def _make_pdf(tmp_path: Path, pages: int = 3) -> str:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"page {i + 1}")
    p = tmp_path / "doc.pdf"
    doc.save(str(p))
    doc.close()
    return str(p)


def test_pdf_to_images_all(tmp_path):
    pdf = _make_pdf(tmp_path, 2)
    imgs, notes = ev.pdf_to_images(pdf, pages="all", dpi=72)
    assert len(imgs) == 2
    assert all(Path(i).is_file() for i in imgs)
    assert any("rasterized 2/2" in n for n in notes)


def test_pdf_to_images_range_and_single(tmp_path):
    pdf = _make_pdf(tmp_path, 4)
    imgs, notes = ev.pdf_to_images(pdf, pages="1-2,4", dpi=72)
    assert len(imgs) == 3


def test_pdf_to_images_invalid_specs(tmp_path):
    pdf = _make_pdf(tmp_path, 2)
    imgs, notes = ev.pdf_to_images(pdf, pages="x-y,zz, ,99", dpi=72)
    # invalid range, invalid page, blank part skipped, out-of-range filtered
    assert imgs == []
    assert any("invalid page range" in n for n in notes)
    assert any("invalid page" in n for n in notes)
