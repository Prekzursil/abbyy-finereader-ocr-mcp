"""Multi-engine OCR MCP server.

Engines: RapidOCR (default, local/headless), Tesseract, ABBYY FineReader 16.
Tools: list_engines, ocr_image, ocr_pdf, batch_ocr, compare_engines, evaluate_accuracy.

Run: python index.py   (stdio MCP transport)
"""

from __future__ import annotations

import contextlib
import glob as _glob
import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

import engines as _engines
import evaluation as _eval

mcp = FastMCP("ocr")
ENGINES = _engines.build_engines()
DEFAULT_ENGINE = "rapidocr"

# Optional sandbox: if OCR_MCP_ALLOWED_DIRS is set (os.pathsep-separated), tools may
# only read files under those directories. Unset = read anything the process can
# (fine for a trusted local client; see the README security note).
_ALLOWED = [Path(p).resolve() for p in os.environ.get("OCR_MCP_ALLOWED_DIRS", "").split(os.pathsep) if p.strip()]


def _check_path(path: str) -> str | None:
    """Return an error string if `path` is outside the allowlist, else None."""
    if not _ALLOWED:
        return None
    try:
        p = Path(path).resolve()
    except Exception as e:
        return f"invalid path: {e}"
    if not any(p == a or a in p.parents for a in _ALLOWED):
        return f"path outside OCR_MCP_ALLOWED_DIRS: {p}"
    return None


def _resolve_engine(engine: str | None) -> str:
    if not engine or engine == "auto":
        return DEFAULT_ENGINE
    if engine not in ENGINES:
        raise ValueError(f"unknown engine '{engine}'. Available: {list(ENGINES)} (or 'auto').")
    return engine


@mcp.tool()
def list_engines() -> str:
    """List OCR engines and whether each is currently usable on this machine.

    Returns JSON: for each engine -> {available, status}. Call this first to see
    which engines compare_engines will actually run."""
    out = {}
    for name, eng in ENGINES.items():
        ok, status = eng.available()
        out[name] = {"available": ok, "status": status}
    return json.dumps({"engines": out, "default": DEFAULT_ENGINE}, indent=2)


@mcp.tool()
def ocr_image(path: str, engine: str = "auto", lang: str = "en", preprocess: bool = False) -> str:
    """OCR a single image file (PNG/JPG/TIFF/BMP).

    Args:
        path: absolute path to the image.
        engine: 'auto' (RapidOCR), 'rapidocr', 'tesseract', or 'finereader'.
        lang: ISO 639-1 code ('en','de','fr','ro',...). Mapped per-engine.
        preprocess: if true, apply grayscale/denoise/deskew first (needs opencv).

    Returns JSON: {engine, ok, text, mean_confidence, line_count,
    low_confidence_count, lines:[{text,confidence,bbox}], warnings}."""
    err = _check_path(path)
    if err:
        return json.dumps({"error": err})
    try:
        name = _resolve_engine(engine)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    use_path, pre_notes = (path, [])
    if preprocess:
        use_path, pre_notes = _eval.preprocess(path)
    res = ENGINES[name].ocr_image(use_path, lang=lang)
    d = res.to_dict()
    if pre_notes:
        d["preprocessing"] = pre_notes
    return json.dumps(d, indent=2, ensure_ascii=False)


@mcp.tool()
def ocr_pdf(path: str, engine: str = "auto", lang: str = "en", pages: str = "all", dpi: int = 300) -> str:
    """OCR a PDF by rasterizing pages (PyMuPDF) then running an engine per page.

    Args:
        path: absolute path to the PDF.
        engine/lang: see ocr_image.
        pages: 'all' or a range like '1-3,5'.
        dpi: rasterization DPI (default 300; higher = slower, more accurate).

    Returns JSON: {page_count, pages:[{page, ...ocr_image result...}], full_text}."""
    err = _check_path(path)
    if err:
        return json.dumps({"error": err})
    try:
        name = _resolve_engine(engine)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    if not Path(path).is_file():
        return json.dumps({"error": f"file not found: {path}"})
    try:
        imgs, notes = _eval.pdf_to_images(path, pages=pages, dpi=dpi)
    except Exception as e:
        return json.dumps({"error": f"pdf rasterization failed: {e}"})
    page_results = []
    texts = []
    try:
        for i, img in enumerate(imgs, start=1):
            res = ENGINES[name].ocr_image(img, lang=lang).to_dict()
            res["page"] = i
            page_results.append(res)
            texts.append(res.get("text", ""))
    finally:
        for img in imgs:  # clean up rasterized page temp files
            with contextlib.suppress(OSError):
                os.remove(img)
    return json.dumps(
        {
            "engine": name,
            "notes": notes,
            "page_count": len(imgs),
            "pages": page_results,
            "full_text": "\n\n".join(texts),
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool()
def batch_ocr(paths_or_glob: str, engine: str = "auto", lang: str = "en") -> str:
    """OCR many images. `paths_or_glob` is a glob (e.g. 'C:/scans/*.png') or a
    JSON list of absolute paths. Returns JSON: {count, results:[...]}."""
    try:
        name = _resolve_engine(engine)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    paths: list[str]
    s = paths_or_glob.strip()
    if s.startswith("["):
        try:
            paths = json.loads(s)
        except Exception as e:
            return json.dumps({"error": f"bad JSON path list: {e}"})
    else:
        paths = _glob.glob(s)
    if not paths:
        return json.dumps({"error": f"no files matched: {paths_or_glob}"})
    if _ALLOWED:
        blocked = [p for p in paths if _check_path(p)]
        if blocked:
            return json.dumps(
                {
                    "error": f"{len(blocked)} path(s) outside OCR_MCP_ALLOWED_DIRS",
                    "examples": blocked[:3],
                }
            )
    results = []
    for p in paths:
        res = ENGINES[name].ocr_image(p, lang=lang).to_dict()
        results.append(
            {
                "path": p,
                "text": res.get("text", ""),
                "mean_confidence": res.get("mean_confidence"),
                "ok": res.get("ok"),
                "error": res.get("error"),
            }
        )
    return json.dumps({"engine": name, "count": len(results), "results": results}, indent=2, ensure_ascii=False)


@mcp.tool()
def compare_engines(path: str, lang: str = "en") -> str:
    """Run ALL available engines on one image and compare them — the core
    accuracy tool when you have no ground truth.

    Returns JSON: per-engine {text, mean_confidence, ok}, plus pairwise text
    similarity, average agreement, and a 'consensus_engine' (the one whose output
    best agrees with the others)."""
    per = {}
    texts = {}
    for name, eng in ENGINES.items():
        ok, _ = eng.available()
        if not ok:
            per[name] = {"available": False}
            continue
        res = eng.ocr_image(path, lang=lang)
        per[name] = {
            "available": True,
            "ok": res.ok,
            "error": res.error,
            "text": res.text,
            "mean_confidence": res.mean_confidence,
            "elapsed_s": res.elapsed_s,
        }
        if res.ok and res.text:
            texts[name] = res.text
    agreement = _eval.cross_agreement(texts)
    return json.dumps({"path": path, "engines": per, "agreement": agreement}, indent=2, ensure_ascii=False)


@mcp.tool()
def evaluate_accuracy(
    ground_truth_path: str, ocr_text: str = "", ocr_path: str = "", engine: str = "auto", lang: str = "en"
) -> str:
    """Score OCR output against a ground-truth text file (CER/WER).

    Provide EITHER `ocr_text` (already-extracted text) OR `ocr_path` (an image/PDF
    to OCR now with `engine`). Compares against the UTF-8 text at
    `ground_truth_path`.

    Returns JSON: {cer, wer, char_accuracy_pct, word_accuracy_pct,
    substitutions, deletions, insertions, hits}."""
    for pth in (ground_truth_path, ocr_path):
        if pth and (err := _check_path(pth)):
            return json.dumps({"error": err})
    gt = Path(ground_truth_path)
    if not gt.is_file():
        return json.dumps({"error": f"ground truth not found: {ground_truth_path}"})
    reference = gt.read_text(encoding="utf-8", errors="replace")
    hypothesis = ocr_text
    used = {"source": "provided_text"}
    if not hypothesis and ocr_path:
        try:
            name = _resolve_engine(engine)
        except ValueError as e:
            return json.dumps({"error": str(e)})
        p = Path(ocr_path)
        if not p.is_file():
            return json.dumps({"error": f"ocr_path not found: {ocr_path}"})
        if p.suffix.lower() == ".pdf":
            imgs, _ = _eval.pdf_to_images(ocr_path)
            hypothesis = "\n\n".join(ENGINES[name].ocr_image(i, lang=lang).text for i in imgs)
        else:
            hypothesis = ENGINES[name].ocr_image(ocr_path, lang=lang).text
        used = {"source": "ocr", "engine": name}
    if not hypothesis:
        return json.dumps({"error": "provide ocr_text or ocr_path"})
    metrics = _eval.evaluate(hypothesis, reference)
    metrics["evaluated"] = used
    return json.dumps(metrics, indent=2, ensure_ascii=False)


if __name__ == "__main__":  # pragma: no cover - stdio entrypoint, not unit-testable
    mcp.run()
