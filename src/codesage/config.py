"""Configuration constants and classes for CodeSage repository ingestion."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set


DEFAULT_SOURCE_EXTENSIONS: Set[str] = {
    ".dart",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".py",
}

# Directories that should never be searched for application source units
EXCLUDED_DIRECTORIES: Set[str] = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "build",
    "dist",
    "out",
    "target",
    ".dart_tool",
    ".pub",
    ".pub-cache",
    ".idea",
    ".vscode",
    ".pytest_cache",
    "__pycache__",
    "coverage",
    ".nyc_output",
    "ephemeral",
    "Pods",
    "uploads",
    "outputs",
}

# Dependency lockfiles and generated files to exclude
EXCLUDED_FILENAMES: Set[str] = {
    "package-lock.json",
    "pubspec.lock",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Gemfile.lock",
    "Cargo.lock",
    "local.properties",
    "secrets.xml",
}

# Binary files, images, and user documents
BINARY_AND_DOC_EXTENSIONS: Set[str] = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".bmp",
    ".tiff",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".mp3",
    ".mp4",
    ".wav",
}

# Patterns indicating secrets, credentials, or environment configs
SECRET_FILENAME_PREFIXES: tuple = (
    ".env",
    "id_rsa",
    "id_ed25519",
)

SECRET_EXTENSIONS: Set[str] = {
    ".pem",
    ".key",
    ".crt",
    ".cert",
    ".p12",
    ".pfx",
    ".keystore",
    ".jks",
}


@dataclass
class CodeSageConfig:
    """Configurable settings for scanning and indexing."""

    repo_root: Path
    output_dir: Path
    supported_extensions: Set[str] = field(
        default_factory=lambda: set(DEFAULT_SOURCE_EXTENSIONS)
    )
    max_file_size_bytes: int = 512 * 1024  # 512 KB
    max_files: Optional[int] = None
    max_unit_lines: int = 250
    dry_run: bool = False

    def validate(self) -> None:
        """Validate paths and configuration parameters."""
        if not self.repo_root.exists() or not self.repo_root.is_dir():
            raise ValueError(f"Repository root does not exist or is not a directory: {self.repo_root}")

        if self.max_file_size_bytes <= 0:
            raise ValueError(f"max_file_size_bytes must be positive, got {self.max_file_size_bytes}")

        if self.max_files is not None and self.max_files <= 0:
            raise ValueError(f"max_files must be positive if set, got {self.max_files}")

        if self.max_unit_lines <= 0:
            raise ValueError(f"max_unit_lines must be positive, got {self.max_unit_lines}")
