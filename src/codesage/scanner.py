"""Safe repository scanner for source code extraction."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from codesage.config import (
    BINARY_AND_DOC_EXTENSIONS,
    CodeSageConfig,
    DEFAULT_SOURCE_EXTENSIONS,
    EXCLUDED_DIRECTORIES,
    EXCLUDED_FILENAMES,
    SECRET_EXTENSIONS,
    SECRET_FILENAME_PREFIXES,
)


@dataclass
class SkippedFileRecord:
    """Record of a skipped file and the reason."""
    path: str
    reason: str


@dataclass
class ScanResult:
    """Outcome of repository scanning."""
    repo_root: Path
    source_files: List[Path] = field(default_factory=list)
    relative_paths: List[str] = field(default_factory=list)
    skipped_files: List[SkippedFileRecord] = field(default_factory=list)
    skip_reasons_count: Dict[str, int] = field(default_factory=dict)

    def record_skip(self, path_str: str, reason: str) -> None:
        self.skipped_files.append(SkippedFileRecord(path=path_str, reason=reason))
        self.skip_reasons_count[reason] = self.skip_reasons_count.get(reason, 0) + 1


class RepositoryScanner:
    """Scans a repository tree safely, deterministically, and conservatively."""

    def __init__(self, config: CodeSageConfig):
        self.config = config
        self.repo_root = config.repo_root.resolve()
        self.output_dir = config.output_dir.resolve()

    def is_safe_path(self, path: Path) -> bool:
        """Verify that path does not escape the repository root via symlinks."""
        try:
            resolved = path.resolve()
            return resolved == self.repo_root or self.repo_root in resolved.parents
        except (OSError, RuntimeError):
            return False

    def is_output_dir(self, dir_path: Path) -> bool:
        """Check if directory is the configured CodeSage output directory."""
        try:
            resolved = dir_path.resolve()
            return resolved == self.output_dir or self.output_dir in resolved.parents
        except (OSError, RuntimeError):
            return False

    def scan(self) -> ScanResult:
        """Perform deterministic repository scan."""
        result = ScanResult(repo_root=self.repo_root)

        if not self.repo_root.exists() or not self.repo_root.is_dir():
            return result

        # We traverse with os.walk, sorting for deterministic discovery
        for root_str, dirs, files in os.walk(str(self.repo_root), followlinks=False):
            root_path = Path(root_str)

            # Prevent traversing outside repo root via symlink or anomalous paths
            if not self.is_safe_path(root_path):
                rel_root = self._to_relative(root_path)
                result.record_skip(rel_root, "symlink_outside_root")
                dirs.clear()
                continue

            # Filter subdirectories in-place deterministically
            dirs.sort()
            filtered_dirs = []
            for d in dirs:
                dir_path = root_path / d
                rel_d = self._to_relative(dir_path)

                # Check if symlinked directory points outside
                if dir_path.is_symlink() and not self.is_safe_path(dir_path):
                    result.record_skip(rel_d, "symlink_outside_root")
                    continue

                # Check if directory is the CodeSage output directory
                if self.is_output_dir(dir_path):
                    result.record_skip(rel_d, "codesage_output_directory")
                    continue

                # Check conservative excluded directory names
                if d in EXCLUDED_DIRECTORIES:
                    result.record_skip(rel_d, "ignored_directory")
                    continue

                filtered_dirs.append(d)

            dirs[:] = filtered_dirs

            # Process files deterministically
            files.sort()
            for f in files:
                file_path = root_path / f
                rel_path = self._to_relative(file_path)

                # Check max files limit
                if (
                    self.config.max_files is not None
                    and len(result.source_files) >= self.config.max_files
                ):
                    result.record_skip(rel_path, "max_files_limit_reached")
                    continue

                # Check symlink safety
                if file_path.is_symlink() and not self.is_safe_path(file_path):
                    result.record_skip(rel_path, "symlink_outside_root")
                    continue

                # Check secrets / environment files
                if self._is_secret_or_env(f, file_path):
                    result.record_skip(rel_path, "secret_or_env_pattern")
                    continue

                # Check lockfiles & generated exclusions
                if f in EXCLUDED_FILENAMES:
                    result.record_skip(rel_path, "lockfile_or_generated")
                    continue

                # Check binary & user documents (e.g. PDFs)
                suffix_lower = file_path.suffix.lower()
                if suffix_lower in BINARY_AND_DOC_EXTENSIONS:
                    result.record_skip(rel_path, "binary_or_document")
                    continue

                # Check extension allowlist
                if suffix_lower not in self.config.supported_extensions:
                    result.record_skip(rel_path, "extension_not_supported")
                    continue

                # Check file size
                try:
                    size = file_path.stat().st_size
                    if size > self.config.max_file_size_bytes:
                        result.record_skip(rel_path, "file_too_large")
                        continue
                except OSError:
                    result.record_skip(rel_path, "stat_error")
                    continue

                # Check readability
                try:
                    with open(file_path, "rb") as test_f:
                        chunk = test_f.read(8192)
                        # Check for null bytes indicating binary content
                        if b"\x00" in chunk:
                            result.record_skip(rel_path, "binary_content_detected")
                            continue
                except OSError:
                    result.record_skip(rel_path, "unreadable_file")
                    continue

                # Valid source file found
                result.source_files.append(file_path)
                result.relative_paths.append(rel_path)

        # Deterministic sorting of final source file list
        paired = sorted(
            zip(result.relative_paths, result.source_files),
            key=lambda item: item[0],
        )
        if paired:
            result.relative_paths, result.source_files = zip(*paired)  # type: ignore
            result.relative_paths = list(result.relative_paths)
            result.source_files = list(result.source_files)
        else:
            result.relative_paths = []
            result.source_files = []

        return result

    def _is_secret_or_env(self, filename: str, path: Path) -> bool:
        """Detect potential secrets, credentials, or environment files."""
        name_lower = filename.lower()
        for prefix in SECRET_FILENAME_PREFIXES:
            if name_lower == prefix or name_lower.startswith(f"{prefix}."):
                return True
        if path.suffix.lower() in SECRET_EXTENSIONS:
            return True
        return False

    def _to_relative(self, path: Path) -> str:
        """Convert path to normalized relative POSIX string."""
        try:
            rel = path.resolve().relative_to(self.repo_root)
            return rel.as_posix()
        except ValueError:
            return path.name
