"""Full-coverage unit tests for engines.py.

The actual OCR backends (rapidocr_onnxruntime, onnxruntime, pytesseract,
pyperclip, FineReader.exe) are not installed/available in CI, so every backend
call is mocked. A controllable fake clock makes the FineReader clipboard-polling
loop deterministic without real sleeps.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engines as eng  # noqa: E402


# --------------------------------------------------------------- helpers ------
def _real_file(tmp_path: Path) -> str:
    from PIL import Image

    p = tmp_path / "img.png"
    Image.new("RGB", (4, 4), "white").save(str(p))
    return str(p)


class FakeClock:
    """Deterministic monotonic clock: sleep() advances time()."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def time(self) -> float:
        return self.now

    def sleep(self, secs: float) -> None:
        self.now += secs


class FakeProc:
    def __init__(self, poll_val=None, terminate_exc=None) -> None:
        self._poll = poll_val
        self._te = terminate_exc

    def poll(self):
        return self._poll

    def terminate(self):
        if self._te:
            raise self._te

    def wait(self, timeout=None):
        return None


# --------------------------------------------------------- OcrResult/Engine ---
def test_ocrresult_to_dict_and_helpers():
    r = eng.OcrResult(engine="x")
    r.lines = [eng.OcrLine("hi", 0.9), eng.OcrLine("lo", 0.2), eng.OcrLine("nc", None)]
    r.mean_confidence = 0.55
    r.elapsed_s = 1.23456
    d = r.to_dict()
    assert d["ok"] is True
    assert d["line_count"] == 3
    assert d["low_confidence_count"] == 1  # only the 0.2 line
    assert d["mean_confidence"] == 0.55
    assert d["elapsed_s"] == 1.235
    assert d["lines"][2]["confidence"] is None


def test_ocrresult_to_dict_none_values():
    d = eng.OcrResult(engine="y").to_dict()
    assert d["mean_confidence"] is None
    assert d["elapsed_s"] is None


def test_ocrresult_ok_false_on_error():
    r = eng.OcrResult(engine="z", error="boom")
    assert r.ok is False


def test_engine_base_not_implemented():
    base = eng.Engine()
    with pytest.raises(NotImplementedError):
        base.available()
    with pytest.raises(NotImplementedError):
        base.ocr_image("p")


def test_engine_mean_empty_and_values():
    assert eng.Engine._mean([]) is None
    assert eng.Engine._mean([None]) is None
    assert eng.Engine._mean([0.2, 0.4]) == pytest.approx(0.3)


# ------------------------------------------------------------- RapidOCR -------
@pytest.fixture
def fake_rapid(monkeypatch):
    mod = types.ModuleType("rapidocr_onnxruntime")

    class _Rapid:
        result = ([[[[0, 0]], "hello", 0.9], [[[1, 1]], "world", 0.8]], 0.01)

        def __call__(self, path):
            return self.result

    mod.RapidOCR = _Rapid
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", mod)
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    return _Rapid


def test_rapidocr_available(fake_rapid):
    ok, status = eng.RapidOCREngine().available()
    assert ok is True
    assert "ready" in status


def test_rapidocr_file_not_found():
    r = eng.RapidOCREngine().ocr_image("nope.png")
    assert r.error.startswith("file not found")


def test_rapidocr_success_and_get_caching(fake_rapid, tmp_path):
    e = eng.RapidOCREngine()
    img = _real_file(tmp_path)
    r1 = e.ocr_image(img)  # _engine is None -> imports
    r2 = e.ocr_image(img)  # _engine cached -> reuse branch
    assert r1.ok and r2.ok
    assert "hello" in r1.text and "world" in r1.text
    assert r1.mean_confidence == pytest.approx(0.85)


def test_rapidocr_no_text(fake_rapid, tmp_path, monkeypatch):
    fake_rapid.result = ([], 0.0)
    r = eng.RapidOCREngine().ocr_image(_real_file(tmp_path))
    assert r.warnings == ["no text detected"]
    assert r.text == ""


def test_rapidocr_engine_exception(tmp_path, monkeypatch):
    e = eng.RapidOCREngine()
    monkeypatch.setattr(e, "_get", lambda: (_ for _ in ()).throw(RuntimeError("kaboom")))
    r = e.ocr_image(_real_file(tmp_path))
    assert r.error.startswith("rapidocr failure")


# ------------------------------------------------------------- Tesseract ------
def _fake_pytesseract(data=None, raises=None):
    mod = types.ModuleType("pytesseract")
    inner = types.SimpleNamespace(tesseract_cmd="")
    mod.pytesseract = inner

    class _Out:
        DICT = "dict"

    mod.Output = _Out

    def image_to_data(img, lang=None, output_type=None):
        if raises:
            raise raises
        return data

    mod.image_to_data = image_to_data
    return mod


def test_tesseract_exe_on_path(monkeypatch):
    monkeypatch.setattr(eng.shutil, "which", lambda n: "/usr/bin/tesseract")
    assert eng.TesseractEngine()._exe() == "/usr/bin/tesseract"


def test_tesseract_exe_program_files(monkeypatch):
    monkeypatch.setattr(eng.shutil, "which", lambda n: None)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    assert eng.TesseractEngine()._exe().endswith("tesseract.exe")


def test_tesseract_exe_none(monkeypatch):
    monkeypatch.setattr(eng.shutil, "which", lambda n: None)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    assert eng.TesseractEngine()._exe() is None


def test_tesseract_available_import_fail(monkeypatch):
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    ok, status = eng.TesseractEngine().available()
    assert ok is False and "import failed" in status


def test_tesseract_available_no_exe(monkeypatch):
    monkeypatch.setitem(sys.modules, "pytesseract", _fake_pytesseract())
    monkeypatch.setattr(eng.TesseractEngine, "_exe", lambda self: None)
    ok, status = eng.TesseractEngine().available()
    assert ok is False and "not found" in status


def test_tesseract_available_ready(monkeypatch):
    monkeypatch.setitem(sys.modules, "pytesseract", _fake_pytesseract())
    monkeypatch.setattr(eng.TesseractEngine, "_exe", lambda self: "/bin/tesseract")
    ok, status = eng.TesseractEngine().available()
    assert ok is True and "ready" in status


def test_tesseract_ocr_file_not_found():
    r = eng.TesseractEngine().ocr_image("missing.png")
    assert r.error.startswith("file not found")


def test_tesseract_ocr_no_exe(monkeypatch, tmp_path):
    monkeypatch.setattr(eng.TesseractEngine, "_exe", lambda self: None)
    r = eng.TesseractEngine().ocr_image(_real_file(tmp_path))
    assert r.error == "tesseract.exe not found"


def test_tesseract_ocr_success(monkeypatch, tmp_path):
    data = {
        "text": ["good", "", "lowneg"],
        "conf": ["95", "50", "-1"],
        "left": [1, 0, 0],
        "top": [2, 0, 0],
        "width": [3, 0, 0],
        "height": [4, 0, 0],
    }
    monkeypatch.setitem(sys.modules, "pytesseract", _fake_pytesseract(data=data))
    monkeypatch.setattr(eng.TesseractEngine, "_exe", lambda self: "/bin/tesseract")
    r = eng.TesseractEngine().ocr_image(_real_file(tmp_path), lang="en")
    assert r.text == "good"  # empty txt and conf<0 skipped
    assert r.lines[0].bbox == [1, 2, 4, 6]
    assert r.mean_confidence == pytest.approx(0.95)


def test_tesseract_ocr_no_lines_warning(monkeypatch, tmp_path):
    data = {"text": [""], "conf": ["-1"], "left": [0], "top": [0], "width": [0], "height": [0]}
    monkeypatch.setitem(sys.modules, "pytesseract", _fake_pytesseract(data=data))
    monkeypatch.setattr(eng.TesseractEngine, "_exe", lambda self: "/bin/tesseract")
    # unknown lang falls through the _TESS_LANG.get default branch
    r = eng.TesseractEngine().ocr_image(_real_file(tmp_path), lang="xx")
    assert r.warnings == ["no text detected"]


def test_tesseract_ocr_exception(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "pytesseract", _fake_pytesseract(raises=RuntimeError("x")))
    monkeypatch.setattr(eng.TesseractEngine, "_exe", lambda self: "/bin/tesseract")
    r = eng.TesseractEngine().ocr_image(_real_file(tmp_path))
    assert r.error.startswith("tesseract failure")


# ------------------------------------------------------------ FineReader ------
def test_finereader_exe_primary(monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda self: self.name == "FineReaderOCR.exe")
    assert eng.FineReaderEngine()._exe().endswith("FineReaderOCR.exe")


def test_finereader_exe_alt(monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda self: self.name == "FineReader.exe")
    assert eng.FineReaderEngine()._exe().endswith("FineReader.exe")


def test_finereader_exe_none(monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    assert eng.FineReaderEngine()._exe() is None


def test_finereader_available_none(monkeypatch):
    monkeypatch.setattr(eng.FineReaderEngine, "_exe", lambda self: None)
    ok, status = eng.FineReaderEngine().available()
    assert ok is False and "not found" in status


def test_finereader_available_ready(monkeypatch):
    monkeypatch.setattr(eng.FineReaderEngine, "_exe", lambda self: r"C:\fr\FineReaderOCR.exe")
    ok, status = eng.FineReaderEngine().available()
    assert ok is True and "ready" in status


def test_finereader_file_not_found():
    r = eng.FineReaderEngine().ocr_image("absent.png")
    assert r.error.startswith("file not found")


def test_finereader_no_exe(monkeypatch, tmp_path):
    monkeypatch.setattr(eng.FineReaderEngine, "_exe", lambda self: None)
    r = eng.FineReaderEngine().ocr_image(_real_file(tmp_path))
    assert r.error == "FineReader not found"


def _setup_finereader(monkeypatch, paste_behaviour, copy_exc=None, poll_val=None, terminate_exc=None, popen_exc=None):
    monkeypatch.setattr(eng.FineReaderEngine, "_exe", lambda self: r"C:\fr\FineReaderOCR.exe")
    monkeypatch.setattr(eng, "time", FakeClock())

    clip = types.ModuleType("pyperclip")

    def copy(val):
        if copy_exc:
            raise copy_exc

    clip.copy = copy
    clip.paste = paste_behaviour
    monkeypatch.setitem(sys.modules, "pyperclip", clip)

    def popen(*a, **k):
        if popen_exc:
            raise popen_exc
        return FakeProc(poll_val=poll_val, terminate_exc=terminate_exc)

    monkeypatch.setattr(eng.subprocess, "Popen", popen)


def test_finereader_success_first_poll(monkeypatch, tmp_path):
    _setup_finereader(monkeypatch, paste_behaviour=lambda: "RECOGNIZED TEXT\nline two")
    r = eng.FineReaderEngine().ocr_image(_real_file(tmp_path), lang="en")
    assert r.error is None
    assert r.text == "RECOGNIZED TEXT\nline two"
    assert len(r.lines) == 2
    assert any("confidence" in w for w in r.warnings)


def test_finereader_copy_sentinel_fails(monkeypatch, tmp_path):
    _setup_finereader(monkeypatch, paste_behaviour=lambda: "TEXT", copy_exc=RuntimeError("noclip"))
    r = eng.FineReaderEngine().ocr_image(_real_file(tmp_path))
    assert r.text == "TEXT"
    assert any("sentinel" in w for w in r.warnings)


def test_finereader_proc_exit_settle_success(monkeypatch, tmp_path):
    # paste returns "" until the settle branch, where it returns text
    calls = {"n": 0}

    def paste():
        calls["n"] += 1
        return "SETTLED TEXT" if calls["n"] > 4 else ""

    _setup_finereader(monkeypatch, paste_behaviour=paste, poll_val=0, terminate_exc=RuntimeError("reap"))
    r = eng.FineReaderEngine(timeout_s=10).ocr_image(_real_file(tmp_path))
    assert r.text == "SETTLED TEXT"


def test_finereader_paste_raises_then_settle_empty_error(monkeypatch, tmp_path):
    def paste():
        raise RuntimeError("clip dead")

    _setup_finereader(monkeypatch, paste_behaviour=paste, poll_val=0)
    r = eng.FineReaderEngine(timeout_s=10).ocr_image(_real_file(tmp_path))
    assert r.error.startswith("no clipboard result")


def test_finereader_timeout_no_result(monkeypatch, tmp_path):
    _setup_finereader(monkeypatch, paste_behaviour=lambda: "", poll_val=None)
    r = eng.FineReaderEngine(timeout_s=2).ocr_image(_real_file(tmp_path))
    assert r.error.startswith("no clipboard result")


def test_finereader_outer_exception(monkeypatch, tmp_path):
    _setup_finereader(monkeypatch, paste_behaviour=lambda: "T", popen_exc=RuntimeError("spawn fail"))
    r = eng.FineReaderEngine().ocr_image(_real_file(tmp_path))
    assert r.error.startswith("finereader failure")


# ------------------------------------------------------------ build_engines ---
def test_build_engines():
    engines = eng.build_engines()
    assert set(engines) == {"rapidocr", "tesseract", "finereader"}
