"""Automated unit and integration tests for CodeSage scanner."""

import os
from pathlib import Path

import pytest

from codesage.config import CodeSageConfig
from codesage.indexer import CodeSageIndexer
from codesage.scanner import RepositoryScanner


def test_scan_empty_repo(tmp_path: Path):
    """Empty repository yields zero source files without error."""
    config = CodeSageConfig(
        repo_root=tmp_path,
        output_dir=tmp_path / "data",
    )
    scanner = RepositoryScanner(config)
    result = scanner.scan()

    assert len(result.source_files) == 0
    assert len(result.skipped_files) == 0
    assert len(result.relative_paths) == 0


def test_scan_supported_extensions_and_skips(tmp_path: Path):
    """Only supported source extensions are indexed; others skipped."""
    (tmp_path / "main.dart").write_text("void main() {}", encoding="utf-8")
    (tmp_path / "server.js").write_text("console.log('test');", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('test')", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("plain text", encoding="utf-8")
    (tmp_path / "styles.css").write_text("body {}", encoding="utf-8")

    config = CodeSageConfig(
        repo_root=tmp_path,
        output_dir=tmp_path / "data",
    )
    scanner = RepositoryScanner(config)
    result = scanner.scan()

    indexed_names = [f.name for f in result.source_files]
    assert "main.dart" in indexed_names
    assert "server.js" in indexed_names
    assert "app.py" in indexed_names
    assert "notes.txt" not in indexed_names
    assert "styles.css" not in indexed_names

    # Check skipped reasons
    assert result.skip_reasons_count.get("extension_not_supported", 0) >= 2


def test_scan_excludes_dependency_and_build_dirs(tmp_path: Path):
    """Dependency and build folders must not have their files indexed."""
    # Source file
    (tmp_path / "index.js").write_text("const x = 1;", encoding="utf-8")

    # Excluded directories
    node_modules = tmp_path / "node_modules" / "pkg"
    node_modules.mkdir(parents=True)
    (node_modules / "lib.js").write_text("module.exports = {};", encoding="utf-8")

    build_dir = tmp_path / "build" / "app"
    build_dir.mkdir(parents=True)
    (build_dir / "gen.dart").write_text("void gen() {}", encoding="utf-8")

    dart_tool = tmp_path / ".dart_tool"
    dart_tool.mkdir(parents=True)
    (dart_tool / "tool.dart").write_text("void t() {}", encoding="utf-8")

    git_dir = tmp_path / ".git" / "hooks"
    git_dir.mkdir(parents=True)
    (git_dir / "hook.js").write_text("console.log('hook');", encoding="utf-8")

    config = CodeSageConfig(
        repo_root=tmp_path,
        output_dir=tmp_path / "data",
    )
    scanner = RepositoryScanner(config)
    result = scanner.scan()

    indexed_rel = result.relative_paths
    assert "index.js" in indexed_rel
    assert not any("node_modules" in p for p in indexed_rel)
    assert not any("build" in p for p in indexed_rel)
    assert not any(".dart_tool" in p for p in indexed_rel)
    assert not any(".git" in p for p in indexed_rel)
    assert result.skip_reasons_count.get("ignored_directory", 0) >= 4


def test_scan_excludes_secrets_and_env_files(tmp_path: Path):
    """Secrets, .env files, certificates, and keys must never be indexed."""
    (tmp_path / ".env").write_text("SECRET=12345", encoding="utf-8")
    (tmp_path / ".env.local").write_text("KEY=abc", encoding="utf-8")
    (tmp_path / ".env.production").write_text("DB=prod", encoding="utf-8")
    (tmp_path / "server.key").write_text("PRIVATE KEY DATA", encoding="utf-8")
    (tmp_path / "cert.pem").write_text("CERT DATA", encoding="utf-8")
    (tmp_path / "valid.dart").write_text("void run() {}", encoding="utf-8")

    config = CodeSageConfig(
        repo_root=tmp_path,
        output_dir=tmp_path / "data",
    )
    scanner = RepositoryScanner(config)
    result = scanner.scan()

    assert result.relative_paths == ["valid.dart"]
    assert result.skip_reasons_count.get("secret_or_env_pattern", 0) >= 5


def test_scan_excludes_lockfiles_and_binaries(tmp_path: Path):
    """Lockfiles, PDFs, and binary files must never be treated as source code."""
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pubspec.lock").write_text("packages: {}", encoding="utf-8")
    (tmp_path / "document.pdf").write_bytes(b"%PDF-1.4 binary document")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n binary image")
    (tmp_path / "valid.js").write_text("const a = 1;", encoding="utf-8")

    config = CodeSageConfig(
        repo_root=tmp_path,
        output_dir=tmp_path / "data",
    )
    scanner = RepositoryScanner(config)
    result = scanner.scan()

    assert result.relative_paths == ["valid.js"]
    assert result.skip_reasons_count.get("lockfile_or_generated", 0) == 2
    assert result.skip_reasons_count.get("binary_or_document", 0) == 2


def test_scan_file_size_limit(tmp_path: Path):
    """Files exceeding max_file_size_bytes must be skipped."""
    small_file = tmp_path / "small.js"
    small_file.write_text("const x = 1;", encoding="utf-8")

    large_file = tmp_path / "large.js"
    large_file.write_text("x" * 2000, encoding="utf-8")

    config = CodeSageConfig(
        repo_root=tmp_path,
        output_dir=tmp_path / "data",
        max_file_size_bytes=1000,  # 1 KB limit
    )
    scanner = RepositoryScanner(config)
    result = scanner.scan()

    assert "small.js" in result.relative_paths
    assert "large.js" not in result.relative_paths
    assert result.skip_reasons_count.get("file_too_large", 0) == 1


def test_scan_max_files_limit(tmp_path: Path):
    """Configuring max_files limits the count of indexed files."""
    for i in range(10):
        (tmp_path / f"file_{i:02d}.js").write_text(f"const f{i} = {i};", encoding="utf-8")

    config = CodeSageConfig(
        repo_root=tmp_path,
        output_dir=tmp_path / "data",
        max_files=4,
    )
    scanner = RepositoryScanner(config)
    result = scanner.scan()

    assert len(result.source_files) == 4
    assert result.skip_reasons_count.get("max_files_limit_reached", 0) == 6


def test_scan_deterministic_ordering(tmp_path: Path):
    """Scanned files must always be returned in deterministic alphabetical order."""
    names = ["z.js", "m.dart", "a.js", "b.py", "c.dart"]
    for n in names:
        (tmp_path / n).write_text("// test", encoding="utf-8")

    config = CodeSageConfig(
        repo_root=tmp_path,
        output_dir=tmp_path / "data",
    )
    scanner = RepositoryScanner(config)
    result = scanner.scan()

    expected = sorted(names)
    assert result.relative_paths == expected


def test_indexer_manifest_and_jsonl_output(tmp_path: Path):
    """Indexer generates valid code_units.jsonl and manifest.json without leaking secrets."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    out_dir = tmp_path / "out_data"

    (repo_dir / "index.js").write_text("function greet() { return 'hi'; }", encoding="utf-8")
    (repo_dir / ".env").write_text("SUPER_SECRET_KEY=123", encoding="utf-8")

    config = CodeSageConfig(
        repo_root=repo_dir,
        output_dir=out_dir,
        dry_run=False,
    )
    indexer = CodeSageIndexer(config)
    result = indexer.build_index()

    assert result.jsonl_path is not None and result.jsonl_path.exists()
    assert result.manifest_path is not None and result.manifest_path.exists()

    manifest = result.manifest
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["counts"]["total_files_indexed"] == 1
    assert manifest["counts"]["total_units_extracted"] >= 1
    assert manifest["counts"]["by_language"]["javascript"] >= 1

    # Ensure secret is not in manifest or code units
    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    jsonl_text = result.jsonl_path.read_text(encoding="utf-8")
    assert "SUPER_SECRET_KEY" not in manifest_text
    assert "SUPER_SECRET_KEY" not in jsonl_text
    assert str(repo_dir) not in manifest_text  # No absolute path leaked
