# ocr-mcp — Multi-Engine OCR MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io) server that gives AI
assistants (Claude Code, Codex, Cursor, …) **OCR with first-class accuracy handling
and evaluation**. It wraps three engines behind one interface and can score and
compare them:

| Engine | Backend | Local? | Confidence | Notes |
|--------|---------|--------|-----------|-------|
| **RapidOCR** (default) | PaddleOCR models on [onnxruntime](https://onnxruntime.ai) | ✅ fully local/headless | per-line | No GPU/torch needed; great default |
| **Tesseract** | Google [Tesseract](https://github.com/tesseract-ocr/tesseract) via `pytesseract` | ✅ local | per-word | Needs `tesseract.exe` on PATH |
| **ABBYY FineReader 16** | local FineReader Regular CLI (`/send Clipboard`) | ✅ local | — | Best accuracy; GUI flashes, 1 doc at a time. Headless file output needs ABBYY's paid Extended CLI |

> **Why multi-engine?** No single OCR engine wins on every document. This server
> lets the model run several, **compare their agreement**, and **score them against
> ground truth (CER/WER)** — so you can pick the right engine per job instead of
> guessing.

## Features

- 📄 OCR images **and** PDFs (PDFs rasterized via PyMuPDF, per-page OCR).
- 🎯 **Confidence scores** per line/word, with low-confidence flagging.
- ⚖️ **`compare_engines`** — run every available engine on one document and report
  pairwise agreement + a consensus pick (no ground truth required).
- 📏 **`evaluate_accuracy`** — CER / WER, char/word accuracy %, and edit breakdown
  (substitutions/deletions/insertions) against a ground-truth text file.
- 🧹 Optional preprocessing (grayscale / denoise / deskew via OpenCV).
- 🧱 Fails soft: an unavailable engine is reported, never crashes the server.

## Tools

| Tool | Description |
|------|-------------|
| `list_engines()` | Which engines are usable on this machine + status. Call first. |
| `ocr_image(path, engine="auto", lang="en", preprocess=False)` | OCR one image. |
| `ocr_pdf(path, engine="auto", lang="en", pages="all", dpi=300)` | OCR a PDF. |
| `batch_ocr(paths_or_glob, engine="auto", lang="en")` | OCR many images (glob or JSON list). |
| `compare_engines(path, lang="en")` | Run all engines, compare agreement + consensus. |
| `evaluate_accuracy(ground_truth_path, ocr_text="" \| ocr_path="", engine, lang)` | CER/WER vs ground truth. |

`engine` ∈ `auto` (=RapidOCR) · `rapidocr` · `tesseract` · `finereader`.
`lang` is an ISO-639-1 code (`en`, `de`, `fr`, `ro`, `zh`, …), mapped per engine.

## Requirements

- **Python ≥ 3.12** (3.12 recommended — all wheels mature; 3.14 also works for the
  core RapidOCR path but OpenCV/PyMuPDF wheels may lag).
- **Tesseract** (optional): install [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki)
  and add `tesseract.exe` to PATH for that engine.
- **ABBYY FineReader 16** (optional): a local install enables the FineReader engine
  (Regular-CLI clipboard mode). Headless file output requires ABBYY's Extended CLI license.

## Install

```bash
git clone https://github.com/Prekzursil/ocr-mcp
cd ocr-mcp
uv venv --python 3.12
uv pip install -e .
# (first OCR call downloads the small RapidOCR ONNX models, ~?? MB, cached locally)
```

## Configure

### Claude Code
```bash
claude mcp add ocr -s user -- "/abs/path/ocr-mcp/.venv/Scripts/python.exe" "/abs/path/ocr-mcp/index.py"
```

### Codex (`~/.codex/config.toml`)
```toml
[mcp_servers.ocr]
command = "D:\\path\\ocr-mcp\\.venv\\Scripts\\python.exe"
args = ["D:\\path\\ocr-mcp\\index.py"]
startup_timeout_sec = 60
tool_timeout_sec = 300

[mcp_servers.ocr.env]
PYTHONUTF8 = "1"
PYTHONUNBUFFERED = "1"
```

### Generic MCP client (`mcp.json`)
```json
{
  "mcpServers": {
    "ocr": { "command": "/abs/path/.venv/bin/python", "args": ["/abs/path/index.py"] }
  }
}
```

## Usage examples

```
> OCR this scan and tell me how confident you are.
  → ocr_image("C:/scans/invoice.png")  → text + mean_confidence + low-confidence lines

> Which engine reads this receipt best?
  → compare_engines("C:/scans/receipt.jpg")  → per-engine text + agreement + consensus

> How accurate is RapidOCR on this page vs my transcript?
  → evaluate_accuracy("truth.txt", ocr_path="page.png", engine="rapidocr") → CER/WER
```

## Evaluation methodology

`evaluate_accuracy` uses [`jiwer`](https://github.com/jitsi/jiwer) for **CER**
(character error rate) and **WER** (word error rate). Lower is better;
`char_accuracy_pct = (1 − CER)·100`. Keep ground-truth `.txt` files next to your
test images to track engine accuracy over time. `compare_engines` is the
no-ground-truth fallback: it reports how much the engines agree and which one is
the consensus.

## Development

```bash
uv pip install -e ".[test]"
pytest        # renders known text → OCR → asserts recovery + low CER
```

## Security

This server reads any file path the MCP client gives it — i.e. **any file readable by
the server process**. There is no sandbox by default. Run it only with a **trusted MCP
client**, and be aware that an LLM driving the tools could be prompted to read arbitrary
local files.

For defense-in-depth, set **`OCR_MCP_ALLOWED_DIRS`** (an `os.pathsep`-separated list of
directories) to restrict all tools to files under those roots:

```toml
[mcp_servers.ocr.env]
OCR_MCP_ALLOWED_DIRS = "D:\\scans;D:\\documents"
```

Also note: `batch_ocr` with a recursive glob (`**/*.png`) can match very large file
sets — scope your globs. The FineReader engine shells out to the local
`FineReaderOCR.exe` (list-form args, no shell) and reads the OS clipboard.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

[RapidOCR](https://github.com/RapidAI/RapidOCR) · [Tesseract](https://github.com/tesseract-ocr/tesseract) ·
[ABBYY FineReader](https://pdf.abbyy.com) · [jiwer](https://github.com/jitsi/jiwer) ·
[PyMuPDF](https://github.com/pymupdf/PyMuPDF) · [MCP](https://modelcontextprotocol.io)
