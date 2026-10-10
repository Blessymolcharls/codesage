"""Automated unit tests for CodeSage extractors and unit integrity."""

from pathlib import Path
import pytest

from codesage.config import CodeSageConfig
from codesage.extractors import (
    CodeUnitExtractor,
    LineChunkExtractor,
    PythonAstExtractor,
    TreeSitterExtractor,
    compute_sha256,
    make_unit_id,
    split_large_unit_if_needed,
)
from codesage.indexer import CodeSageIndexer


def test_javascript_extraction(tmp_path: Path):
    """Extract JavaScript functions, classes, and methods using Tree-sitter."""
    js_code = """class Calculator {
  constructor(initial) {
    this.value = initial;
  }

  add(n) {
    return this.value + n;
  }
}

function topHelper(x) {
  return x * 2;
}

const arrowFunc = (a, b) => {
  return a + b;
};
"""
    f = tmp_path / "calc.js"
    f.write_text(js_code, encoding="utf-8")

    extractor = CodeUnitExtractor()
    units, status, warnings = extractor.extract_from_file(tmp_path, f)

    assert status == "success"
    types = [u.unit_type for u in units]
    names = [u.symbol_name for u in units]

    assert "class" in types
    assert "method" in types
    assert "function" in types

    assert "Calculator" in names
    assert "add" in names
    assert "topHelper" in names
    assert "arrowFunc" in names

    # Check method parent symbol context
    add_method = [u for u in units if u.symbol_name == "add"][0]
    assert add_method.parent_symbol == "Calculator"
    assert add_method.unit_type == "method"
    assert add_method.extraction_method == "tree_sitter"

    # Line range and text check
    lines = js_code.splitlines(keepends=True)
    expected_slice = "".join(lines[add_method.start_line - 1 : add_method.end_line])
    assert add_method.source_text == expected_slice
    assert add_method.content_sha256 == compute_sha256(expected_slice)


def test_dart_extraction(tmp_path: Path):
    """Extract Dart classes, methods, and functions with Tree-sitter."""
    dart_code = """class Greeter {
  final String greeting;

  Greeter(this.greeting);

  String greet(String name) {
    return '$greeting, $name!';
  }
}

void main() {
  final g = Greeter('Hello');
  print(g.greet('World'));
}
"""
    f = tmp_path / "main.dart"
    f.write_text(dart_code, encoding="utf-8")

    extractor = CodeUnitExtractor()
    units, status, warnings = extractor.extract_from_file(tmp_path, f)

    assert status == "success"
    names = [u.symbol_name for u in units]
    types = [u.unit_type for u in units]

    assert "Greeter" in names
    assert "greet" in names
    assert "main" in names

    assert "class" in types
    assert "method" in types
    assert "function" in types

    greet_unit = [u for u in units if u.symbol_name == "greet"][0]
    assert greet_unit.parent_symbol == "Greeter"
    assert greet_unit.start_line < greet_unit.end_line

    lines = dart_code.splitlines(keepends=True)
    expected_slice = "".join(lines[greet_unit.start_line - 1 : greet_unit.end_line])
    assert greet_unit.source_text == expected_slice


def test_python_ast_extraction(tmp_path: Path):
    """Extract Python functions, classes, and methods using built-in AST."""
    py_code = """class Worker:
    def __init__(self, name: str):
        self.name = name

    def execute(self) -> bool:
        return True

def run_worker() -> None:
    w = Worker("agent")
    w.execute()
"""
    f = tmp_path / "worker.py"
    f.write_text(py_code, encoding="utf-8")

    extractor = CodeUnitExtractor()
    units, status, warnings = extractor.extract_from_file(tmp_path, f)

    assert status == "success"
    names = [u.symbol_name for u in units]
    assert "Worker" in names
    assert "execute" in names
    assert "run_worker" in names

    exec_unit = [u for u in units if u.symbol_name == "execute"][0]
    assert exec_unit.parent_symbol == "Worker"
    assert exec_unit.unit_type == "method"
    assert exec_unit.extraction_method == "python_ast"

    run_unit = [u for u in units if u.symbol_name == "run_worker"][0]
    assert run_unit.parent_symbol is None
    assert run_unit.unit_type == "function"


def test_line_chunk_fallback(tmp_path: Path):
    """Files without grammar support or unparseable syntax fall back to line chunks."""
    chunker = LineChunkExtractor(chunk_size=10)
    sample_text = "\n".join([f"line {i}" for i in range(1, 26)])
    units = chunker.extract("sample.ts", sample_text, "typescript")

    assert len(units) == 3
    assert units[0].start_line == 1
    assert units[0].end_line == 10
    assert units[1].start_line == 11
    assert units[1].end_line == 20
    assert units[2].start_line == 21
    assert units[2].end_line == 25

    for u in units:
        assert u.unit_type == "line_chunk"
        assert u.extraction_method == "line_chunk"
        assert u.parser_status == "fallback"


def test_oversized_unit_splitting(tmp_path: Path):
    """Units exceeding max_unit_lines are split into non-overlapping sub-chunks."""
    py_lines = ["def huge():\n"] + [f"    x = {i}\n" for i in range(1, 150)]
    py_code = "".join(py_lines)
    f = tmp_path / "huge.py"
    f.write_text(py_code, encoding="utf-8")

    extractor = CodeUnitExtractor()
    units, status, warnings = extractor.extract_from_file(tmp_path, f, max_unit_lines=50)

    assert len(units) > 1
    # Check that parts are non-overlapping and contiguous
    for i in range(len(units) - 1):
        assert units[i].end_line + 1 == units[i + 1].start_line
        assert ":part" in units[i].unit_id
        assert len(units[i].warnings) > 0


def test_safe_output_path_enforcement(tmp_path: Path):
    """Indexer refuses to write outside allowed filesystem boundaries."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "index.js").write_text("console.log(1);", encoding="utf-8")

    root_anchor = Path(Path().resolve().anchor)
    config = CodeSageConfig(
        repo_root=repo_dir,
        output_dir=root_anchor,
    )
    indexer = CodeSageIndexer(config)
    with pytest.raises(PermissionError):
        indexer.build_index()
