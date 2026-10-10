# CodeSage

CodeSage is a research-oriented codebase intelligence and semantic indexing system.
This milestone (Milestone 1) provides safe repository scanning, syntax-aware code unit extraction, and deterministic dataset generation without modifying existing application source code.

---

## 1. Architecture & Package Structure

CodeSage is structured as an isolated Python package:

```text
codesage/
├── README.md                 # Complete documentation and setup guide
├── requirements.txt          # Pinned Python dependencies
├── pytest.ini                # Test runner configuration (pythonpath=src)
├── .gitignore                # Conservative ignore rules for artifacts & secrets
├── src/
│   └── codesage/
│       ├── __init__.py       # Package definition
│       ├── config.py         # Config dataclasses, extensions, exclusion rules
│       ├── scanner.py        # Safe, deterministic filesystem scanner
│       ├── extractors.py     # Tree-sitter & AST syntax-aware code-unit extractors
│       └── indexer.py        # Pipeline orchestrator and manifest writer
├── scripts/
│   └── build_index.py        # CLI entry point for index generation
├── tests/
│   ├── test_scanner.py       # Scanner test suite (synthetic fixtures)
│   └── test_extractors.py    # Extractor, boundary, and parser test suite
└── data/
    ├── .gitkeep              # Directory marker (data files ignored by default)
    ├── code_units.jsonl      # Generated code units dataset (ignored in Git)
    └── manifest.json         # Indexing metadata and statistics (ignored in Git)
```

---

## 2. Environment Setup (Windows PowerShell)

Ensure Python 3.10+ (tested on Python 3.13) is installed. Run all commands from the `codesage` project directory.

### Step 2.1: Create Virtual Environment
```powershell
python -m venv .venv
```

### Step 2.2: Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating).*

### Step 2.3: Install Dependencies
```powershell
pip install -r requirements.txt
```

Pinned dependencies in `requirements.txt`:
- `tree-sitter==0.26.0`: Core Tree-sitter parsing engine.
- `tree-sitter-javascript==0.25.0`: Maintained JavaScript grammar for syntax-aware AST parsing.
- `tree-sitter-dart==0.1.0`: Native Dart grammar for Flutter codebase AST parsing.
- `pytest==9.1.1`: Automated test execution framework.

---

## 3. CLI Usage

The ingestion CLI is invoked via `scripts/build_index.py`.

### Basic Ingestion
To index DocuCharm (located at `../DocuCharm` or `..`):
```powershell
python scripts/build_index.py --repo-root ..
```

### Dry Run (Scan & Preview Without Writing Files)
```powershell
python scripts/build_index.py --repo-root .. --dry-run
```

### Ingestion With File Limits
```powershell
python scripts/build_index.py --repo-root .. --max-files 5
```

### Available CLI Options
- `--repo-root <PATH>`: Target repository directory to scan (default: `..`). Resolves safely to target repository without traversing outside.
- `--output-dir <PATH>`: Destination directory for index artifacts (default: `data/`). Enforces safe output boundary checks.
- `--max-files <N>`: Maximum number of source files to index.
- `--max-file-size-kb <KB>`: Maximum individual file size in kilobytes (default: `512`).
- `--max-unit-lines <LINES>`: Threshold at which an oversized unit is deterministically divided into non-overlapping chunks with stable IDs (default: `250`).
- `--dry-run`: Performs discovery and unit extraction reporting without writing output files.

---

## 4. Supported Languages and Limitations

### Supported Languages
1. **Dart** (`.dart`):
   - Syntax-aware parsing using Tree-sitter.
   - Extracts top-level functions, classes, and methods (including constructor signatures and class getters/methods) with parent class context.
2. **JavaScript** (`.js`, `.jsx`, `.mjs`, `.cjs`):
   - Syntax-aware parsing using Tree-sitter.
   - Extracts function declarations, arrow functions assigned to variables, classes, and class methods.
3. **Python** (`.py`):
   - Syntax-aware parsing using Python's built-in `ast` module.
   - Extracts module-level functions, classes, and class methods with parent symbol tracking.
4. **Fallback Chunker** (`.ts`, `.tsx`, or files without recognized symbols):
   - Deterministic, non-overlapping line chunking (default chunk size: 50 lines).
   - Tagged with `extraction_method="line_chunk"` and `parser_status="fallback"`.

### Limitations
- Dynamic evaluation patterns (e.g. `eval()`, complex runtime metaprogramming) are indexed by their syntactic boundaries rather than execution semantics.
- Very large functions exceeding `--max-unit-lines` (default 250 lines) are split into non-overlapping sub-chunks with warnings recorded in metadata.

---

## 5. Output Schema

The indexer writes two artifacts into `data/`:

### 5.1 `data/code_units.jsonl`
Each line is a self-contained JSON object representing an extracted code unit:

```json
{
  "unit_id": "backend/controllers/pdfController.js:function:mergePdfs:9_36",
  "repository_relative_path": "backend/controllers/pdfController.js",
  "language": "javascript",
  "unit_type": "function",
  "symbol_name": "mergePdfs",
  "parent_symbol": null,
  "start_line": 9,
  "end_line": 36,
  "source_text": "const mergePdfs = async (req, res) => {\n  ...\n};\n",
  "content_sha256": "d108600641b2413f27c6fe60c689b9c25f48f505296ceec028546503f408e60a",
  "extraction_method": "tree_sitter",
  "parser_status": "success",
  "warnings": []
}
```

#### Field Definitions
- `unit_id`: Deterministic identifier in the format `<path>:<type>:<symbol>:<start>_<end>[:part<N>]`.
- `repository_relative_path`: Path relative to the repository root (POSIX forward slashes).
- `language`: Canonical language name (`dart`, `javascript`, `python`).
- `unit_type`: Unit category (`function`, `method`, `class`, `line_chunk`).
- `symbol_name`: Extracted symbol name, or `null` for line chunks.
- `parent_symbol`: Contextual enclosing class or container, or `null`.
- `start_line`: 1-based inclusive start line number.
- `end_line`: 1-based inclusive end line number.
- `source_text`: Exact un-truncated source text.
- `content_sha256`: SHA-256 hex digest of the UTF-8 source text.
- `extraction_method`: Extractor used (`tree_sitter`, `python_ast`, `line_chunk`).
- `parser_status`: Status indicator (`success`, `partial`, `fallback`).
- `warnings`: List of warning messages (e.g., chunk splitting notes).

### 5.2 `data/manifest.json`
Provides reproducibility metadata, timestamp, file counts, and schema verification:

```json
{
  "schema_version": "1.0.0",
  "generated_at": "2026-10-10T17:19:29.695047+00:00",
  "repository_label": "DocuCharm",
  "counts": {
    "total_files_discovered": 187,
    "total_files_indexed": 16,
    "total_files_skipped": 171,
    "total_units_extracted": 89,
    "by_language": {
      "dart": 80,
      "javascript": 9
    },
    "by_unit_type": {
      "class": 27,
      "function": 9,
      "line_chunk": 2,
      "method": 51
    },
    "by_extraction_method": {
      "line_chunk": 2,
      "tree_sitter": 87
    },
    "by_parser_status": {
      "fallback": 2,
      "success": 87
    }
  },
  "skipped_files_by_reason": {
    "binary_or_document": 43,
    "extension_not_supported": 114,
    "ignored_directory": 11,
    "lockfile_or_generated": 3
  },
  "parser_failures": [],
  "configuration": {
    "supported_extensions": [".dart", ".js", ".jsx", ".py", ".ts", ".tsx"],
    "max_files": null,
    "max_file_size_bytes": 524288,
    "max_unit_lines": 250,
    "dry_run": false
  }
}
```

---

## 6. Running Tests

Automated tests use isolated temporary directories and synthetic fixtures to ensure existing application files are never touched.

Run pytest:
```powershell
pytest -v
```

All 15 tests verify:
- Scanner exclusions (dependencies, build caches, `.git`, `.venv`).
- Privacy and secrets exclusions (`.env*`, `.pem`, `.key`, credentials).
- Binary and PDF filtering (user documents excluded from code index).
- Tree-sitter extraction for Dart and JavaScript.
- Python AST extraction.
- Deterministic ordering and stable unit IDs.
- Oversized unit splitting into non-overlapping parts.
- Safe output directory containment.

---

## 7. Privacy and Preservation Rules

1. **Secrets & Environment Isolation**: Scanner excludes all `.env*`, `.key`, `.pem`, `id_rsa`, and credential patterns. Secrets are neither printed nor indexed.
2. **User Data & Binaries**: PDF files, images, compiled binaries, and output uploads are strictly excluded from source indexing.
3. **Non-Destructive Execution**: The target repository is only read; existing application configurations, code, and dependencies are left untouched.
4. **Git Protection**: Generated datasets (`data/code_units.jsonl`, `data/manifest.json`) and the `.venv/` folder are ignored by Git. Only `data/.gitkeep` is tracked.
