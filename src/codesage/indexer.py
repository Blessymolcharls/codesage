"""Core indexing pipeline and artifact generation for CodeSage."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from codesage.config import CodeSageConfig
from codesage.extractors import CodeUnit, CodeUnitExtractor
from codesage.scanner import RepositoryScanner, ScanResult


SCHEMA_VERSION = "1.0.0"


@dataclass
class IndexResult:
    """Result of repository indexing."""

    manifest: Dict[str, Any]
    code_units: List[CodeUnit]
    scan_result: ScanResult
    jsonl_path: Optional[Path] = None
    manifest_path: Optional[Path] = None


class CodeSageIndexer:
    """Coordinates repository scanning, unit extraction, and index persistence."""

    def __init__(self, config: CodeSageConfig):
        self.config = config
        self.config.validate()
        self.scanner = RepositoryScanner(config)
        self.extractor = CodeUnitExtractor()

    def build_index(self) -> IndexResult:
        """Run complete ingestion and extraction workflow."""
        scan_res = self.scanner.scan()

        all_units: List[CodeUnit] = []
        parser_failures: List[Dict[str, str]] = []
        by_language: Dict[str, int] = {}
        by_unit_type: Dict[str, int] = {}
        by_extraction_method: Dict[str, int] = {}
        by_parser_status: Dict[str, int] = {}

        for file_path in scan_res.source_files:
            try:
                rel_path = file_path.resolve().relative_to(self.config.repo_root.resolve()).as_posix()
            except ValueError:
                rel_path = file_path.name

            units, parser_status, warnings = self.extractor.extract_from_file(
                repo_root=self.config.repo_root,
                file_path=file_path,
                max_unit_lines=self.config.max_unit_lines,
            )

            if parser_status == "fallback" and any("syntax error" in w.lower() or "exception" in w.lower() for w in warnings):
                parser_failures.append({
                    "file": rel_path,
                    "error": "; ".join(warnings),
                })

            for u in units:
                all_units.append(u)
                by_language[u.language] = by_language.get(u.language, 0) + 1
                by_unit_type[u.unit_type] = by_unit_type.get(u.unit_type, 0) + 1
                by_extraction_method[u.extraction_method] = by_extraction_method.get(u.extraction_method, 0) + 1
                by_parser_status[u.parser_status] = by_parser_status.get(u.parser_status, 0) + 1

        # Deterministic sorting of code units
        all_units.sort(
            key=lambda u: (
                u.repository_relative_path,
                u.start_line,
                u.end_line,
                u.unit_id,
            )
        )

        repo_label = self.config.repo_root.resolve().name or "repository"
        now_utc = datetime.now(timezone.utc).isoformat()

        manifest: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": now_utc,
            "repository_label": repo_label,
            "counts": {
                "total_files_discovered": len(scan_res.source_files) + len(scan_res.skipped_files),
                "total_files_indexed": len(scan_res.source_files),
                "total_files_skipped": len(scan_res.skipped_files),
                "total_units_extracted": len(all_units),
                "by_language": dict(sorted(by_language.items())),
                "by_unit_type": dict(sorted(by_unit_type.items())),
                "by_extraction_method": dict(sorted(by_extraction_method.items())),
                "by_parser_status": dict(sorted(by_parser_status.items())),
            },
            "skipped_files_by_reason": dict(sorted(scan_res.skip_reasons_count.items())),
            "parser_failures": parser_failures,
            "configuration": {
                "supported_extensions": sorted(list(self.config.supported_extensions)),
                "max_files": self.config.max_files,
                "max_file_size_bytes": self.config.max_file_size_bytes,
                "max_unit_lines": self.config.max_unit_lines,
                "dry_run": self.config.dry_run,
            },
        }

        jsonl_path: Optional[Path] = None
        manifest_path: Optional[Path] = None

        if not self.config.dry_run:
            self._validate_safe_output_dir(self.config.output_dir)
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

            jsonl_path = self.config.output_dir / "code_units.jsonl"
            manifest_path = self.config.output_dir / "manifest.json"

            with open(jsonl_path, "w", encoding="utf-8") as f:
                for u in all_units:
                    f.write(json.dumps(u.to_dict(), ensure_ascii=False) + "\n")

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
                f.write("\n")

        return IndexResult(
            manifest=manifest,
            code_units=all_units,
            scan_result=scan_res,
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
        )

    def _validate_safe_output_dir(self, output_dir: Path) -> None:
        """Prevent writing outside CodeSage data or designated output boundary."""
        resolved = output_dir.resolve()
        # Ensure it cannot target root paths or traversal escapes
        if resolved == Path(resolved.anchor):
            raise PermissionError(f"Refusing to write index artifacts to filesystem root: {output_dir}")
