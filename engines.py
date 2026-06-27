"""OCR engine adapters for the multi-engine OCR MCP.

Each engine implements `available()` and `ocr_image(path, lang)` returning an
`OcrResult`. Engines fail soft: an unavailable or erroring engine reports the
problem in `OcrResult.warnings`/`error` rather than raising, so the server stays
up and `compare_engines` can run whichever engines work.

Engines:
  - RapidOCREngine  : PaddleOCR models on onnxruntime. Free, fully local/headless,
                      per-line confidence. Default engine. Works on Python 3.12+.
  - TesseractEngine : Google Tesseract via pytesseract. Needs tesseract.exe on PATH
                      or under C:\\Program Files\\Tesseract-OCR. Per-word confidence.
  - FineReaderEngine: ABBYY FineReader 16 (local). Standard license exposes only the
                      Regular CLI, which sends results to an app — we use
                      `/send Clipboard` then read the clipboard. Best-effort: the GUI
                      flashes and it is one-document-at-a-time. Headless file output
                      needs ABBYY's paid Extended CLI license (auto-used if present).
"""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# ---- language code maps (canonical = ISO 639-1 'en','de','fr',...) -----------
_TESS_LANG = {
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "nl": "nld",
    "ru": "rus",
    "ro": "ron",
    "pl": "pol",
    "ja": "jpn",
    "ko": "kor",
    "zh": "chi_sim",
    "ar": "ara",
    "uk": "ukr",
}
_FR_LANG = {  # ABBYY FineReader uses full English language names
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "ro": "Romanian",
    "pl": "Polish",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "ChineseSimplified",
    "ar": "Arabic",
    "uk": "Ukrainian",
}

LOW_CONFIDENCE = 0.60  # lines/words below this are flagged


@dataclass
class OcrLine:
    text: str
    confidence: float | None
    bbox: list | None = None  # [[x,y],...] or [x0,y0,x1,y1]


@dataclass
class OcrResult:
    engine: str
    text: str = ""
    mean_confidence: float | None = None
    lines: list[OcrLine] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_s: float | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def low_confidence_lines(self, threshold: float = LOW_CONFIDENCE) -> list[OcrLine]:
        return [ln for ln in self.lines if ln.confidence is not None and ln.confidence < threshold]

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "ok": self.ok,
            "error": self.error,
            "text": self.text,
            "mean_confidence": (round(self.mean_confidence, 4) if self.mean_confidence is not None else None),
            "line_count": len(self.lines),
            "low_confidence_count": len(self.low_confidence_lines()),
            "lines": [
                {
                    "text": ln.text,
                    "confidence": (round(ln.confidence, 4) if ln.confidence is not None else None),
                    "bbox": ln.bbox,
                }
                for ln in self.lines
            ],
            "warnings": self.warnings,
            "elapsed_s": (round(self.elapsed_s, 3) if self.elapsed_s is not None else None),
        }


class Engine:
    name = "base"

    def available(self) -> tuple[bool, str]:
        """Return (is_available, human-readable status)."""
        raise NotImplementedError

    def ocr_image(self, path: str, lang: str = "en") -> OcrResult:
        raise NotImplementedError

    @staticmethod
    def _mean(confs: list[float]) -> float | None:
        vals = [c for c in confs if c is not None]
        return float(sum(vals) / len(vals)) if vals else None


class RapidOCREngine(Engine):
    name = "rapidocr"

    def __init__(self) -> None:
        self._engine = None  # lazy-loaded (model load is slow)

    def _get(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    def available(self) -> tuple[bool, str]:
        try:
            import onnxruntime  # noqa: F401
            import rapidocr_onnxruntime  # noqa: F401

            return True, "ready (onnxruntime, PaddleOCR models, local/headless)"
        except Exception as e:  # pragma: no cover
            return False, f"import failed: {e}"

    def ocr_image(self, path: str, lang: str = "en") -> OcrResult:
        r = OcrResult(engine=self.name)
        if not Path(path).is_file():
            r.error = f"file not found: {path}"
            return r
        try:
            t0 = time.time()
            engine = self._get()
            out, _elapse = engine(path)
            r.elapsed_s = time.time() - t0
            if not out:
                r.warnings.append("no text detected")
                return r
            confs = []
            for item in out:
                # RapidOCR item = [box, text, score]
                box, txt, score = item[0], item[1], float(item[2])
                r.lines.append(OcrLine(text=str(txt), confidence=score, bbox=list(box)))
                confs.append(score)
            r.text = "\n".join(ln.text for ln in r.lines)
            r.mean_confidence = self._mean(confs)
        except Exception as e:
            r.error = f"rapidocr failure: {e}"
        return r


class TesseractEngine(Engine):
    name = "tesseract"

    def _exe(self) -> str | None:
        exe = shutil.which("tesseract")
        if exe:
            return exe
        cand = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        return cand if Path(cand).is_file() else None

    def available(self) -> tuple[bool, str]:
        try:
            import pytesseract  # noqa: F401
        except Exception as e:
            return False, f"pytesseract import failed: {e}"
        exe = self._exe()
        if not exe:
            return False, "tesseract.exe not found (install Tesseract-OCR and add to PATH)"
        return True, f"ready ({exe})"

    def ocr_image(self, path: str, lang: str = "en") -> OcrResult:
        r = OcrResult(engine=self.name)
        if not Path(path).is_file():
            r.error = f"file not found: {path}"
            return r
        exe = self._exe()
        if not exe:
            r.error = "tesseract.exe not found"
            return r
        try:
            import pytesseract
            from PIL import Image

            pytesseract.pytesseract.tesseract_cmd = exe
            tlang = _TESS_LANG.get(lang, lang)
            t0 = time.time()
            img = Image.open(path)
            data = pytesseract.image_to_data(img, lang=tlang, output_type=pytesseract.Output.DICT)
            r.elapsed_s = time.time() - t0
            confs = []
            n = len(data["text"])
            for i in range(n):
                txt = data["text"][i].strip()
                conf = float(data["conf"][i])
                if not txt or conf < 0:
                    continue
                c = conf / 100.0
                bbox = [
                    data["left"][i],
                    data["top"][i],
                    data["left"][i] + data["width"][i],
                    data["top"][i] + data["height"][i],
                ]
                r.lines.append(OcrLine(text=txt, confidence=c, bbox=bbox))
                confs.append(c)
            r.text = "\n".join(ln.text for ln in r.lines)
            r.mean_confidence = self._mean(confs)
            if not r.lines:
                r.warnings.append("no text detected")
        except Exception as e:
            r.error = f"tesseract failure: {e}"
        return r


class FineReaderEngine(Engine):
    name = "finereader"
    INSTALL = r"C:\Program Files\ABBYY FineReader 16"

    def __init__(self, timeout_s: int = 120) -> None:
        # Clipboard-poll timeout lives on the instance so ocr_image keeps the
        # base Engine.ocr_image(path, lang) signature (Liskov-compatible).
        self.timeout_s = timeout_s

    def _exe(self) -> str | None:
        cand = Path(self.INSTALL) / "FineReaderOCR.exe"
        if cand.is_file():
            return str(cand)
        alt = Path(self.INSTALL) / "FineReader.exe"
        return str(alt) if alt.is_file() else None

    def available(self) -> tuple[bool, str]:
        exe = self._exe()
        if not exe:
            return False, "ABBYY FineReader 16 not found at default path"
        return True, f"ready (Regular CLI clipboard mode; {exe})"

    def ocr_image(self, path: str, lang: str = "en") -> OcrResult:
        r = OcrResult(engine=self.name)
        if not Path(path).is_file():
            r.error = f"file not found: {path}"
            return r
        exe = self._exe()
        if not exe:
            r.error = "FineReader not found"
            return r
        timeout_s = self.timeout_s
        try:
            import pyperclip

            frlang = _FR_LANG.get(lang, "English")
            r.warnings.append("FineReader Regular CLI: GUI may flash; result captured via clipboard.")
            # Write a unique sentinel so any value != sentinel is unambiguously new.
            sentinel = f"__ocrmcp_sentinel_{uuid.uuid4().hex}"
            try:
                pyperclip.copy(sentinel)
            except Exception:
                sentinel = None
                r.warnings.append("could not seed clipboard sentinel; result detection is best-effort")
            t0 = time.time()
            # noqa S603: local, trusted FineReader executable resolved from a fixed
            # install dir; args are an OS-resolved file path and a fixed language name.
            proc = subprocess.Popen(  # noqa: S603
                [exe, str(Path(path).resolve()), "/lang", frlang, "/send", "Clipboard"]
            )
            # poll clipboard until it holds something other than the sentinel
            text = ""
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                time.sleep(1.0)
                try:
                    cur = pyperclip.paste()
                except Exception:
                    cur = ""
                if cur and cur.strip() and cur != sentinel:
                    text = cur
                    break
                if proc.poll() is not None and time.time() - t0 > 3:
                    time.sleep(1.5)  # process exited; let the clipboard settle
                    try:
                        cur = pyperclip.paste()
                    except Exception:
                        cur = ""
                    if cur and cur.strip() and cur != sentinel:
                        text = cur
                    break
            r.elapsed_s = time.time() - t0
            try:
                proc.terminate()
                proc.wait(timeout=5)  # reap so we don't leak a zombie/handle
            except Exception:  # noqa: S110 - best-effort cleanup; nothing actionable to log
                pass
            if not text:
                r.error = (
                    "no clipboard result (FineReader may need a logged-in "
                    "interactive session, or the doc produced no text)"
                )
                return r
            r.text = text.strip()
            r.lines = [OcrLine(text=line, confidence=None) for line in r.text.splitlines() if line.strip()]
            r.warnings.append("FineReader CLI does not expose confidence scores.")
        except Exception as e:
            r.error = f"finereader failure: {e}"
        return r


def build_engines() -> dict[str, Engine]:
    return {e.name: e for e in (RapidOCREngine(), TesseractEngine(), FineReaderEngine())}
