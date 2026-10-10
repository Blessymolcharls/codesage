#!/usr/bin/env python3
"""CLI entry point for CodeSage repository scanning and code-unit extraction."""

import argparse
import sys
from pathlib import Path

# Ensure 'src' is in Python module search path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from codesage.config import CodeSageConfig
from codesage.indexer import CodeSageIndexer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CodeSage: Ingest repository source code and build reproducible code-unit index."
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default="..",
        help="Path to the repository to index (default: '..')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "data"),
        help="Path to directory for output artifacts (default: 'codesage/data')",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of source files to index (optional)",
    )
    parser.add_argument(
        "--max-file-size-kb",
        type=int,
        default=512,
        help="Maximum file size to index in kilobytes (default: 512)",
    )
    parser.add_argument(
        "--max-unit-lines",
        type=int,
        default=250,
        help="Maximum lines before a code unit is split into deterministic chunks (default: 250)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform scanning and extraction without writing output index files",
    )
    return parser.parse_args()


def resolve_repo_path(raw_path: str) -> Path:
    """Resolve target repository path safely."""
    path = Path(raw_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Specified repository root does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Specified repository root is not a directory: {path}")

    # If the user passes '..' from codesage workspace, and '..' contains DocuCharm,
    # target DocuCharm if '..' is the parent folder containing multiple repos
    docucharm_sub = path / "DocuCharm"
    if docucharm_sub.is_dir() and (docucharm_sub / "frontend").is_dir() and not (path / "frontend").is_dir():
        return docucharm_sub

    return path


def resolve_output_path(raw_output: str) -> Path:
    """Resolve and validate output path safety."""
    out_path = Path(raw_output).resolve()
    default_data = (project_root / "data").resolve()

    # Prevent writing to root of filesystem
    if out_path == Path(out_path.anchor):
        raise PermissionError(f"Refusing to write index artifacts to filesystem root: {out_path}")

    # Ensure output path is either within codesage/data or explicitly specified safe dir
    return out_path


def main() -> int:
    args = parse_args()

    try:
        repo_root = resolve_repo_path(args.repo_root)
        output_dir = resolve_output_path(args.output_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    max_size_bytes = args.max_file_size_kb * 1024

    config = CodeSageConfig(
        repo_root=repo_root,
        output_dir=output_dir,
        max_file_size_bytes=max_size_bytes,
        max_files=args.max_files,
        max_unit_lines=args.max_unit_lines,
        dry_run=args.dry_run,
    )

    try:
        indexer = CodeSageIndexer(config)
        result = indexer.build_index()
    except Exception as e:
        print(f"Fatal error during indexing: {e}", file=sys.stderr)
        return 1

    manifest = result.manifest
    counts = manifest["counts"]

    print("=" * 60)
    print("CodeSage Repository Ingestion Summary")
    print("=" * 60)
    print(f"Repository Root:      {repo_root.name} ({repo_root})")
    print(f"Schema Version:       {manifest['schema_version']}")
    print(f"Dry Run:              {config.dry_run}")
    print(f"Discovered Files:     {counts['total_files_discovered']}")
    print(f"Indexed Files:        {counts['total_files_indexed']}")
    print(f"Skipped Files:        {counts['total_files_skipped']}")
    print(f"Extracted Units:      {counts['total_units_extracted']}")
    print("-" * 60)
    print("Units by Language:")
    for lang, cnt in counts["by_language"].items():
        print(f"  - {lang}: {cnt}")
    print("Units by Type:")
    for u_type, cnt in counts["by_unit_type"].items():
        print(f"  - {u_type}: {cnt}")
    print("Extraction Methods:")
    for method, cnt in counts["by_extraction_method"].items():
        print(f"  - {method}: {cnt}")
    print("Parser Statuses:")
    for status, cnt in counts["by_parser_status"].items():
        print(f"  - {status}: {cnt}")

    if manifest["skipped_files_by_reason"]:
        print("-" * 60)
        print("Skipped Files Breakdown:")
        for reason, cnt in manifest["skipped_files_by_reason"].items():
            print(f"  - {reason}: {cnt}")

    if manifest["parser_failures"]:
        print("-" * 60)
        print(f"Parser Failures ({len(manifest['parser_failures'])}):")
        for fail in manifest["parser_failures"][:5]:
            print(f"  - {fail['file']}: {fail['error']}")

    if not config.dry_run and result.jsonl_path and result.manifest_path:
        print("-" * 60)
        print(f"Wrote Code Units:     {result.jsonl_path}")
        print(f"Wrote Manifest:       {result.manifest_path}")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
