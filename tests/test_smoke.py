"""Smoke test: render known text to a PNG, OCR it with RapidOCR, verify recovery
and that evaluate() reports a low CER against the known string."""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import engines as eng_mod  # noqa: E402
import evaluation as ev  # noqa: E402

# Live end-to-end checks against the real RapidOCR backend. Opt-in only: requires
# both the heavy onnxruntime/PaddleOCR stack AND OCR_MCP_LIVE=1 (the first call
# downloads ONNX models). The lean CI gate proves 100% coverage with mocked
# backends in the other test modules, so this stays skipped there.
pytestmark = pytest.mark.skipif(
    os.environ.get("OCR_MCP_LIVE") != "1" or importlib.util.find_spec("rapidocr_onnxruntime") is None,
    reason="live OCR smoke test (set OCR_MCP_LIVE=1 with rapidocr installed)",
)

KNOWN = "The quick brown fox jumps over the lazy dog 1234567890"


def _render(text: str) -> str:
    img = Image.new("RGB", (1000, 160), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except Exception:
        font = ImageFont.load_default()
    d.text((20, 55), text, fill="black", font=font)
    p = Path(tempfile.gettempdir()) / "ocrmcp_smoke.png"
    img.save(str(p))
    return str(p)


def test_rapidocr_recovers_text():
    img = _render(KNOWN)
    res = eng_mod.RapidOCREngine().ocr_image(img, lang="en")
    assert res.ok, f"engine errored: {res.error}"
    assert res.text.strip(), "no text recovered"
    # most distinctive tokens should appear
    low = res.text.lower()
    assert "quick" in low and "brown" in low and "fox" in low, f"got: {res.text!r}"


def test_evaluate_cer_low():
    img = _render(KNOWN)
    res = eng_mod.RapidOCREngine().ocr_image(img, lang="en")
    # join lines into one string for comparison
    hyp = " ".join(res.text.split())
    metrics = ev.evaluate(hyp, KNOWN)
    assert metrics["cer"] < 0.25, f"CER too high: {metrics}"


def test_list_engines_has_rapidocr():
    engines = eng_mod.build_engines()
    assert "rapidocr" in engines
    ok, status = engines["rapidocr"].available()
    assert ok, status
