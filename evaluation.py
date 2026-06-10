"""Accuracy evaluation + image preprocessing for the OCR MCP.

- CER / WER against ground truth via `jiwer`.
- Cross-engine agreement (character similarity) for `compare_engines`.
- Optional preprocessing (grayscale / denoise / deskew) via opencv, applied to a
  temp copy; degrades gracefully if opencv is unavailable.
"""
from __future__ import annotations

import difflib
import os
import tempfile
from pathlib import Path


def evaluate(hypothesis: str, reference: str) -> dict:
    """Compare OCR output (hypothesis) against ground truth (reference)."""
    try:
        import jiwer
    except ImportError:
        return {"error": "jiwer not installed: pip install jiwer"}

    hyp = hypothesis or ""
    ref = reference or ""
    # WER pipeline: normalize whitespace + case for a fair word comparison
    wer = jiwer.wer(ref, hyp)
    cer = jiwer.cer(ref, hyp)
    # Detailed word-level measures
    out = jiwer.process_words(ref, hyp)
    return {
        "cer": round(float(cer), 4),
        "wer": round(float(wer), 4),
        "char_accuracy_pct": round(max(0.0, 1.0 - float(cer)) * 100, 2),
        "word_accuracy_pct": round(max(0.0, 1.0 - float(wer)) * 100, 2),
        "ref_word_count": len(ref.split()),
        "hyp_word_count": len(hyp.split()),
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "hits": out.hits,
    }


def similarity(a: str, b: str) -> float:
    """Character-level similarity ratio in [0,1] between two strings."""
    return round(difflib.SequenceMatcher(None, a or "", b or "").ratio(), 4)


def cross_agreement(texts: dict[str, str]) -> dict:
    """Pairwise similarity between engine outputs + a consensus pick.

    Consensus = the engine whose output is, on average, most similar to the others
    (a cheap 'majority agreement' proxy when no ground truth is available).
    """
    names = [n for n, t in texts.items() if t]
    pairwise: dict[str, float] = {}
    avg: dict[str, list[float]] = {n: [] for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            s = similarity(texts[a], texts[b])
            pairwise[f"{a}__vs__{b}"] = s
            avg[a].append(s)
            avg[b].append(s)
    # single engine (no pairs) -> 0.0 rather than None so callers can sort numerically
    avg_sim = {n: round(sum(v) / len(v), 4) if v else 0.0 for n, v in avg.items()}
    consensus = max(avg_sim, key=lambda n: avg_sim[n]) if avg_sim else None
    return {"pairwise_similarity": pairwise, "avg_similarity": avg_sim, "consensus_engine": consensus}


def preprocess(path: str, grayscale: bool = True, denoise: bool = True, deskew: bool = True) -> tuple[str, list[str]]:
    """Return (path_to_use, notes). Writes a processed temp image; falls back to
    the original path with a note if opencv is missing or processing fails."""
    notes: list[str] = []
    try:
        import cv2
        import numpy as np
    except Exception:
        return path, ["preprocessing skipped (opencv unavailable)"]
    try:
        img = cv2.imread(path)
        if img is None:
            return path, ["preprocessing skipped (could not read image)"]
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            notes.append("grayscale")
        if denoise:
            img = cv2.fastNlMeansDenoising(img) if img.ndim == 2 else cv2.fastNlMeansDenoisingColored(img)
            notes.append("denoise")
        if deskew and img.ndim == 2:
            coords = np.column_stack(np.where(img < 128))
            if len(coords) > 50:
                angle = cv2.minAreaRect(coords)[-1]
                angle = -(90 + angle) if angle < -45 else -angle
                if abs(angle) > 0.3:
                    h, w = img.shape[:2]
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
                    notes.append(f"deskew({angle:.1f}deg)")
        out = Path(tempfile.gettempdir()) / f"ocrmcp_pre_{Path(path).stem}_{os.urandom(4).hex()}.png"
        cv2.imwrite(str(out), img)
        return str(out), notes
    except Exception as e:
        return path, [f"preprocessing skipped (error: {e})"]


def pdf_to_images(pdf_path: str, pages: str = "all", dpi: int = 300) -> tuple[list[str], list[str]]:
    """Rasterize PDF pages to PNGs (PyMuPDF). `pages`='all' or '1-3,5'. Returns
    (image_paths, notes)."""
    notes: list[str] = []
    import fitz  # PyMuPDF

    token = os.urandom(4).hex()
    out_paths: list[str] = []
    with fitz.open(pdf_path) as doc:  # context manager closes even on error
        total = doc.page_count
        if pages == "all":
            idx = list(range(total))
        else:
            idx = []
            for part in pages.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    try:
                        a, b = part.split("-", 1)
                        idx.extend(range(int(a) - 1, int(b)))
                    except ValueError:
                        notes.append(f"skipped invalid page range: {part!r}")
                else:
                    try:
                        idx.append(int(part) - 1)
                    except ValueError:
                        notes.append(f"skipped invalid page: {part!r}")
            idx = [i for i in idx if 0 <= i < total]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        tmp = Path(tempfile.gettempdir())
        for i in idx:
            pix = doc.load_page(i).get_pixmap(matrix=mat)
            p = tmp / f"ocrmcp_pdf_{Path(pdf_path).stem}_{token}_p{i + 1}.png"
            pix.save(str(p))
            out_paths.append(str(p))
        notes.append(f"rasterized {len(out_paths)}/{total} page(s) at {dpi}dpi")
    return out_paths, notes
