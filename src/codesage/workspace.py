"""Repository workspace abstraction for multi-repository indexing and retrieval.

Provides isolated workspace storage, lifecycle management, safe path validation,
and index/graph/cache separation per repository codebase.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from codesage.config import CodeSageConfig
from codesage.graph import RepositoryGraph, RepositoryGraphBuilder
from codesage.indexer import CodeSageIndexer
from codesage.retrieval import SearchResult, TfidfRetriever, load_code_units_from_jsonl

# Regex enforcing safe repository identifiers:
# - Must start with alphanumeric character [a-zA-Z0-9]
# - Can contain letters, numbers, hyphens, underscores, and dots (1-64 characters)
# - Strictly prevents path traversal ("..", "/", "\\"), dots-only (".", ".."), and metacharacters
SAFE_REPO_ID_REGEX = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-\.]{0,63}$")

# Reserved identifiers: system endpoints, device names, and canonical baseline
RESERVED_WORKSPACE_IDS: Set[str] = {
    "default",
    "docucharm",
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


class WorkspaceError(Exception):
    """Base exception for workspace operations."""
    pass


class WorkspaceNotFoundError(WorkspaceError):
    """Raised when a requested workspace does not exist."""
    pass


class InvalidWorkspaceIdError(WorkspaceError):
    """Raised when a workspace identifier violates safety constraints."""
    pass


class UnsafePathError(WorkspaceError):
    """Raised when a repository path escapes allowed boundaries or poses security risks."""
    pass


@dataclass
class WorkspaceMetadata:
    """Metadata describing a repository workspace and its index state."""
    id: str
    name: str
    display_path: str
    repo_path: str
    created_at: str
    indexed_at: Optional[str] = None
    status: str = "unindexed"  # "unindexed" | "indexing" | "ready" | "error"
    unit_count: int = 0
    file_count: int = 0
    languages: List[str] = field(default_factory=list)
    last_error: Optional[str] = None
    source_fingerprint: Optional[str] = None
    is_canonical: bool = False

    def to_dict(self, mask_absolute_path: bool = True) -> Dict[str, Any]:
        """Convert metadata to dictionary, masking host filesystem paths for privacy and security."""
        data = asdict(self)
        if mask_absolute_path:
            # Mask sensitive host username/drive details for all workspaces
            data["repo_path"] = self.display_path
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkspaceMetadata:
        """Create WorkspaceMetadata instance from dictionary."""
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class Workspace:
    """Represents an isolated repository workspace containing index, graph, and cache artifacts."""

    def __init__(
        self,
        metadata: WorkspaceMetadata,
        workspace_dir: Path,
        units_path: Path,
        graph_path: Path,
        manifest_path: Path,
        cache_dir: Path,
    ):
        self.metadata = metadata
        self.workspace_dir = workspace_dir
        self.units_path = units_path
        self.graph_path = graph_path
        self.manifest_path = manifest_path
        self.cache_dir = cache_dir

    @property
    def id(self) -> str:
        return self.metadata.id

    @property
    def is_ready(self) -> bool:
        return self.metadata.status == "ready" and self.units_path.exists()

    def load_code_units(self) -> List[Dict[str, Any]]:
        """Load code units from this workspace's index file."""
        if not self.units_path.exists():
            return []
        return load_code_units_from_jsonl(self.units_path)

    def load_graph(self) -> Optional[RepositoryGraph]:
        """Load repository relationship graph for this workspace."""
        if not self.graph_path.exists():
            return None
        return RepositoryGraph.load_json(self.graph_path)


class WorkspaceManager:
    """Manages multi-repository workspaces, registration, storage isolation, and lifecycles."""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir).resolve()
        self.workspaces_dir = (self.data_dir / "workspaces").resolve()
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)

        # Canonical research workspace paths
        self.canonical_units_path = self.data_dir / "code_units.jsonl"
        self.canonical_graph_path = self.data_dir / "relationship_graph.json"
        self.canonical_manifest_path = self.data_dir / "manifest.json"
        self.canonical_cache_dir = self.data_dir / "cache"

        # Concurrency guards: per-workspace threading locks
        self._workspace_locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _get_workspace_lock(self, repo_id: str) -> threading.Lock:
        """Retrieve or create an isolated threading lock for a specific workspace ID."""
        clean_key = repo_id.strip().lower()
        with self._locks_guard:
            if clean_key not in self._workspace_locks:
                self._workspace_locks[clean_key] = threading.Lock()
            return self._workspace_locks[clean_key]

    @staticmethod
    def validate_workspace_id(repo_id: str, allow_canonical: bool = False) -> str:
        """Validate repository ID against strict safety constraints preventing path traversal."""
        if not isinstance(repo_id, str):
            raise InvalidWorkspaceIdError("Workspace ID must be a string.")

        clean_id = repo_id.strip()
        if not clean_id:
            raise InvalidWorkspaceIdError("Workspace ID cannot be empty.")

        # Reject null bytes and control characters
        if "\x00" in clean_id or any(ord(c) < 32 for c in clean_id):
            raise InvalidWorkspaceIdError("Workspace ID contains illegal control characters.")

        # Reject path separators and traversal tokens
        if "/" in clean_id or "\\" in clean_id or ".." in clean_id:
            raise InvalidWorkspaceIdError(
                f"Invalid workspace ID '{repo_id}'. Must not contain path separators or traversal sequences."
            )

        # Reject single or double dots
        if clean_id in (".", ".."):
            raise InvalidWorkspaceIdError("Workspace ID cannot be '.' or '..'.")

        # Reject ending with a period (Windows filename normalization bypass risk)
        if clean_id.endswith("."):
            raise InvalidWorkspaceIdError(f"Workspace ID '{repo_id}' cannot end with a period.")

        # Match strict alphanumeric prefix and allowed characters
        if not SAFE_REPO_ID_REGEX.match(clean_id):
            raise InvalidWorkspaceIdError(
                f"Invalid workspace ID '{repo_id}'. Must be 1-64 characters matching [a-zA-Z0-9][a-zA-Z0-9_.-]*."
            )

        # Check reserved system and device names
        lower_id = clean_id.lower()
        if lower_id in RESERVED_WORKSPACE_IDS:
            if not (allow_canonical and lower_id in ("default", "docucharm")):
                raise InvalidWorkspaceIdError(
                    f"Workspace ID '{repo_id}' is a reserved system identifier or device name."
                )

        return clean_id

    def _resolve_workspace_dir(self, repo_id: str) -> Path:
        """Deterministically resolve and bound workspace directory path."""
        clean_id = self.validate_workspace_id(repo_id)
        resolved_root = self.workspaces_dir.resolve()
        target = (self.workspaces_dir / clean_id).resolve()

        # Strict boundary containment check
        try:
            rel = target.relative_to(resolved_root)
            if len(rel.parts) != 1 or rel.parts[0] != clean_id:
                raise UnsafePathError(f"Workspace path traversal attempt detected for ID: {clean_id}")
        except ValueError:
            raise UnsafePathError(f"Workspace path traversal attempt detected for ID: {clean_id}")

        if target == resolved_root or target.parent != resolved_root:
            raise UnsafePathError(f"Workspace directory escapes allowed root: {clean_id}")

        return target

    def _get_canonical_workspace(self) -> Workspace:
        """Construct the canonical research baseline workspace pointing to immutable benchmark data."""
        unit_count = 0
        if self.canonical_units_path.exists():
            with open(self.canonical_units_path, "r", encoding="utf-8") as f:
                unit_count = sum(1 for line in f if line.strip())

        meta = WorkspaceMetadata(
            id="default",
            name="DocuCharm (Canonical Research Baseline)",
            display_path="../DocuCharm",
            repo_path="../DocuCharm",
            created_at="2026-10-14T12:00:00Z",
            indexed_at="2026-10-14T12:12:17Z",
            status="ready" if self.canonical_units_path.exists() else "unindexed",
            unit_count=unit_count,
            file_count=16,
            languages=["dart", "javascript"],
            is_canonical=True,
        )

        return Workspace(
            metadata=meta,
            workspace_dir=self.data_dir,
            units_path=self.canonical_units_path,
            graph_path=self.canonical_graph_path,
            manifest_path=self.canonical_manifest_path,
            cache_dir=self.canonical_cache_dir,
        )

    def list_workspaces(self) -> List[WorkspaceMetadata]:
        """List all available workspaces (canonical baseline + user workspaces)."""
        workspaces: List[WorkspaceMetadata] = []

        # 1. Canonical research baseline
        canonical = self._get_canonical_workspace()
        workspaces.append(canonical.metadata)

        # 2. Dynamically registered workspaces
        if self.workspaces_dir.exists():
            for child in sorted(self.workspaces_dir.iterdir()):
                if child.is_dir() and (child / "metadata.json").exists():
                    try:
                        with open(child / "metadata.json", "r", encoding="utf-8") as f:
                            data = json.load(f)
                            workspaces.append(WorkspaceMetadata.from_dict(data))
                    except Exception:
                        continue

        return workspaces

    def get_workspace(self, repo_id: str) -> Workspace:
        """Get workspace instance by ID. Supports 'default' and 'docucharm' aliases."""
        clean_id = repo_id.strip().lower()
        if clean_id in ("default", "docucharm"):
            return self._get_canonical_workspace()

        valid_id = self.validate_workspace_id(repo_id)
        ws_dir = self._resolve_workspace_dir(valid_id)
        meta_file = ws_dir / "metadata.json"

        if not ws_dir.exists() or not meta_file.exists():
            raise WorkspaceNotFoundError(f"Workspace '{repo_id}' not found.")

        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = WorkspaceMetadata.from_dict(data)
        units_path = ws_dir / "code_units.jsonl"
        graph_path = ws_dir / "relationship_graph.json"
        manifest_path = ws_dir / "manifest.json"
        cache_dir = ws_dir / "cache"

        return Workspace(
            metadata=meta,
            workspace_dir=ws_dir,
            units_path=units_path,
            graph_path=graph_path,
            manifest_path=manifest_path,
            cache_dir=cache_dir,
        )

    def register_workspace(
        self,
        repo_id: str,
        repo_path: str | Path,
        name: Optional[str] = None,
    ) -> Workspace:
        """Register a new repository workspace.
        
        Validates target repository path exists and ensures workspace storage is isolated.
        """
        valid_id = self.validate_workspace_id(repo_id)
        if valid_id.lower() in ("default", "docucharm"):
            raise InvalidWorkspaceIdError("The ID 'default' and 'docucharm' are reserved for the canonical baseline.")

        target_path = Path(repo_path).resolve()
        if not target_path.exists() or not target_path.is_dir():
            raise UnsafePathError(f"Target repository path does not exist or is not a directory: {repo_path}")

        # Protect CodeSage internal data and workspace roots from being registered as codebases
        resolved_data = self.data_dir.resolve()
        resolved_workspaces = self.workspaces_dir.resolve()
        if (
            target_path in (resolved_data, resolved_workspaces)
            or resolved_data in target_path.parents
            or resolved_workspaces in target_path.parents
        ):
            raise UnsafePathError("Target repository path cannot be inside the CodeSage internal data directory.")

        ws_dir = self._resolve_workspace_dir(valid_id)
        if target_path == ws_dir.resolve():
            raise UnsafePathError("Target repository path cannot be inside its own workspace directory.")

        ws_dir.mkdir(parents=True, exist_ok=True)
        (ws_dir / "cache").mkdir(exist_ok=True)

        display_name = name.strip() if name and name.strip() else valid_id
        # Derive relative display path to avoid leaking absolute host paths
        try:
            display_path = target_path.relative_to(self.data_dir.parent).as_posix()
        except ValueError:
            display_path = target_path.name

        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta = WorkspaceMetadata(
            id=valid_id,
            name=display_name,
            display_path=display_path,
            repo_path=str(target_path),
            created_at=now_iso,
            status="unindexed",
            is_canonical=False,
        )

        meta_file = ws_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(mask_absolute_path=False), f, indent=2)

        return self.get_workspace(valid_id)

    def delete_workspace(self, repo_id: str) -> None:
        """Delete a registered workspace's generated index artifacts.
        
        IMPORTANT: The target source repository is NEVER touched or deleted.
        Canonical research baseline cannot be deleted.
        """
        clean_id = repo_id.strip().lower()
        if clean_id in ("default", "docucharm"):
            raise WorkspaceError("The canonical research baseline workspace cannot be deleted.")

        valid_id = self.validate_workspace_id(repo_id)
        with self._get_workspace_lock(valid_id):
            ws_dir = self._resolve_workspace_dir(valid_id)
            if not ws_dir.exists():
                raise WorkspaceNotFoundError(f"Workspace '{repo_id}' not found.")

            # Validate target immediately before destructive removal
            resolved_ws = ws_dir.resolve()
            resolved_parent = self.workspaces_dir.resolve()
            if resolved_ws == resolved_parent or resolved_ws.parent != resolved_parent:
                raise UnsafePathError(f"Refusing to delete unsafe workspace directory: {ws_dir}")

            # If the workspace directory itself is a symlink/junction, unlink without traversal
            if ws_dir.is_symlink():
                ws_dir.unlink()
            else:
                shutil.rmtree(ws_dir)

    def _publish_staged_artifacts(self, staging_dir: Path, ws: Workspace) -> None:
        """Atomically validate and publish candidate index artifacts to target workspace directory."""
        ws.workspace_dir.mkdir(parents=True, exist_ok=True)

        artifacts = [
            ("code_units.jsonl", ws.units_path),
            ("relationship_graph.json", ws.graph_path),
            ("manifest.json", ws.manifest_path),
            ("file_hashes.json", ws.workspace_dir / "file_hashes.json"),
        ]

        # 1. Validation pass: ensure all staged artifacts are well-formed before touching destination
        for name, dst in artifacts:
            staged_file = staging_dir / name
            if not staged_file.exists():
                continue

            if name.endswith(".json"):
                with open(staged_file, "r", encoding="utf-8") as f:
                    json.load(f)
            elif name.endswith(".jsonl"):
                with open(staged_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            json.loads(line)

        # 2. Atomic publish pass using os.replace
        for name, dst in artifacts:
            staged_file = staging_dir / name
            if staged_file.exists():
                os.replace(staged_file, dst)

    def index_workspace(
        self,
        repo_id: str,
        max_files: Optional[int] = None,
        max_file_size_kb: int = 512,
        max_unit_lines: int = 250,
    ) -> Workspace:
        """Run full repository scanning, code-unit extraction, and relationship graph construction.
        
        Uses transactional staging to prevent partial or corrupted index publication.
        """
        valid_id = self.validate_workspace_id(repo_id)
        with self._get_workspace_lock(valid_id):
            ws = self.get_workspace(valid_id)
            if ws.metadata.is_canonical:
                raise WorkspaceError("The canonical research baseline is immutable and cannot be reindexed.")

            repo_root = Path(ws.metadata.repo_path).resolve()
            if not repo_root.exists() or not repo_root.is_dir():
                raise UnsafePathError(f"Target repository path no longer exists: {ws.metadata.repo_path}")

            # Mark status as indexing
            ws.metadata.status = "indexing"
            self._save_metadata(ws)

            staging_dir = ws.workspace_dir / ".staging_tmp"
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            staging_dir.mkdir(parents=True, exist_ok=True)

            try:
                # 1. Run Indexer writing into staging_dir
                config = CodeSageConfig(
                    repo_root=repo_root,
                    output_dir=staging_dir,
                    max_files=max_files,
                    max_file_size_bytes=max_file_size_kb * 1024,
                    max_unit_lines=max_unit_lines,
                )
                indexer = CodeSageIndexer(config)
                index_result = indexer.build_index()

                # 2. Build Relationship Graph into staging_dir
                graph_builder = RepositoryGraphBuilder(repo_root=repo_root)
                raw_units = [u.to_dict() if hasattr(u, "to_dict") else u for u in index_result.code_units]
                graph = graph_builder.build(raw_units)
                graph.save(staging_dir / "relationship_graph.json")

                # 3. Save file hashes into staging_dir
                file_hashes: Dict[str, str] = {}
                if hasattr(index_result, "scan_result") and index_result.scan_result:
                    for fp in index_result.scan_result.source_files:
                        try:
                            rel_p = fp.resolve().relative_to(repo_root).as_posix()
                            with open(fp, "rb") as bf:
                                file_hashes[rel_p] = hashlib.sha256(bf.read()).hexdigest()
                        except Exception:
                            continue
                with open(staging_dir / "file_hashes.json", "w", encoding="utf-8") as f:
                    json.dump(file_hashes, f, indent=2)

                # 4. Atomic publication: validate candidate files and publish atomically
                self._publish_staged_artifacts(staging_dir, ws)

                # 5. Update Metadata upon successful publication
                languages = sorted(list(set(u.language for u in index_result.code_units if u.language)))
                now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

                ws.metadata.status = "ready"
                ws.metadata.indexed_at = now_iso
                ws.metadata.unit_count = len(index_result.code_units)
                if hasattr(index_result, "scan_result") and index_result.scan_result:
                    ws.metadata.file_count = len(index_result.scan_result.source_files)
                elif isinstance(index_result.manifest, dict):
                    ws.metadata.file_count = index_result.manifest.get("counts", {}).get("total_files_indexed", 0)
                else:
                    ws.metadata.file_count = 0
                ws.metadata.languages = languages
                ws.metadata.last_error = None
                self._save_metadata(ws)

                return ws
            except Exception as e:
                # Retain previous ready state if valid index exists on disk, otherwise mark error
                ws.metadata.status = "error" if not ws.units_path.exists() else "ready"
                ws.metadata.last_error = str(e)
                self._save_metadata(ws)
                raise WorkspaceError(f"Indexing failed for workspace '{repo_id}': {e}") from e
            finally:
                if staging_dir.exists():
                    shutil.rmtree(staging_dir, ignore_errors=True)

    def incremental_index(
        self,
        repo_id: str,
        max_file_size_kb: int = 512,
        max_unit_lines: int = 250,
    ) -> Tuple[Workspace, Dict[str, Any]]:
        """Perform reliable incremental reindexing on an existing workspace.
        
        Detects added, modified, and deleted files by inspecting file hashes.
        Reprocesses only affected code units and rebuilds the relationship graph.
        Uses transactional staging to prevent partial or corrupted publication.
        Falls back safely to full indexing if previous manifest/index is missing or diff is large.
        """
        valid_id = self.validate_workspace_id(repo_id)
        with self._get_workspace_lock(valid_id):
            ws = self.get_workspace(valid_id)
            if ws.metadata.is_canonical:
                raise WorkspaceError("The canonical research baseline is immutable and cannot be reindexed.")

            if not ws.units_path.exists() or not ws.manifest_path.exists():
                ws = self.index_workspace(valid_id, max_file_size_kb=max_file_size_kb, max_unit_lines=max_unit_lines)
                return ws, {"mode": "full_fallback", "added": 0, "modified": 0, "deleted": 0}

            repo_root = Path(ws.metadata.repo_path).resolve()
            if not repo_root.exists() or not repo_root.is_dir():
                raise UnsafePathError(f"Target repository path no longer exists: {ws.metadata.repo_path}")

            staging_dir = ws.workspace_dir / ".staging_tmp"
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

            try:
                with open(ws.manifest_path, "r", encoding="utf-8") as f:
                    prev_manifest = json.load(f)

                hashes_file = ws.workspace_dir / "file_hashes.json"
                if hashes_file.exists():
                    with open(hashes_file, "r", encoding="utf-8") as f:
                        prev_files = json.load(f)
                else:
                    prev_files = {}
                    for u in ws.load_code_units():
                        rel_p = u.get("repository_relative_path") or u.get("rel_path")
                        if rel_p and rel_p not in prev_files:
                            prev_files[rel_p] = u.get("content_sha256", "")

                config = CodeSageConfig(
                    repo_root=repo_root,
                    output_dir=ws.workspace_dir,
                    max_file_size_bytes=max_file_size_kb * 1024,
                    max_unit_lines=max_unit_lines,
                )
                from codesage.scanner import RepositoryScanner
                scanner = RepositoryScanner(config)
                scan_result = scanner.scan()

                curr_files: Dict[str, str] = {}
                for file_path in scan_result.source_files:
                    try:
                        rel_p = file_path.resolve().relative_to(repo_root).as_posix()
                        with open(file_path, "rb") as bf:
                            curr_files[rel_p] = hashlib.sha256(bf.read()).hexdigest()
                    except Exception:
                        continue

                added = [p for p in curr_files if p not in prev_files]
                deleted = [p for p in prev_files if p not in curr_files]
                modified = [p for p in curr_files if p in prev_files and curr_files[p] != prev_files[p]]

                stats = {
                    "mode": "incremental",
                    "added": len(added),
                    "modified": len(modified),
                    "deleted": len(deleted),
                    "unchanged": len(curr_files) - len(added) - len(modified),
                }

                if not added and not deleted and not modified:
                    return ws, stats

                # If large portion of files changed, safely fall back to full rebuild
                if len(added) + len(deleted) + len(modified) > max(10, len(curr_files) // 2):
                    ws = self.index_workspace(valid_id, max_file_size_kb=max_file_size_kb, max_unit_lines=max_unit_lines)
                    stats["mode"] = "full_fallback_due_to_large_diff"
                    return ws, stats

                staging_dir.mkdir(parents=True, exist_ok=True)

                affected_paths = set(deleted) | set(modified)
                existing_units = ws.load_code_units()
                surviving_units = [u for u in existing_units if u.get("repository_relative_path") not in affected_paths]

                from codesage.extractors import CodeUnitExtractor
                extractor = CodeUnitExtractor()
                fresh_units = []

                for rel_p in added + modified:
                    full_path = repo_root / rel_p
                    if full_path.exists():
                        # Prevent symlink traversal outside repo boundary during extraction
                        try:
                            resolved_p = full_path.resolve()
                            if resolved_p != repo_root and repo_root not in resolved_p.parents:
                                continue
                        except Exception:
                            continue

                        try:
                            ext_units, _, _ = extractor.extract_from_file(
                                repo_root=repo_root,
                                file_path=full_path,
                                max_unit_lines=max_unit_lines,
                            )
                            for u in ext_units:
                                fresh_units.append(u.to_dict())
                        except Exception:
                            pass

                all_units = surviving_units + fresh_units
                all_units.sort(key=lambda u: (u.get("repository_relative_path", ""), u.get("start_line", 0), u.get("unit_id", "")))

                # Write candidate files to staging directory
                with open(staging_dir / "code_units.jsonl", "w", encoding="utf-8") as f:
                    for u in all_units:
                        f.write(json.dumps(u, ensure_ascii=False) + "\n")

                graph_builder = RepositoryGraphBuilder(repo_root=repo_root)
                graph = graph_builder.build(all_units)
                graph.save(staging_dir / "relationship_graph.json")

                if "counts" not in prev_manifest:
                    prev_manifest["counts"] = {}
                prev_manifest["counts"]["total_files_indexed"] = len(curr_files)
                prev_manifest["counts"]["total_units_extracted"] = len(all_units)
                prev_manifest["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                with open(staging_dir / "manifest.json", "w", encoding="utf-8") as f:
                    json.dump(prev_manifest, f, indent=2, ensure_ascii=False)
                    f.write("\n")

                with open(staging_dir / "file_hashes.json", "w", encoding="utf-8") as f:
                    json.dump(curr_files, f, indent=2)

                # Atomic publication
                self._publish_staged_artifacts(staging_dir, ws)

                now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                ws.metadata.indexed_at = now_iso
                ws.metadata.unit_count = len(all_units)
                ws.metadata.file_count = len(curr_files)
                ws.metadata.languages = sorted(list(set(u["language"] for u in all_units if u.get("language"))))
                ws.metadata.status = "ready"
                ws.metadata.last_error = None
                self._save_metadata(ws)

                return ws, stats
            except Exception as e:
                # If staging or publication fails, fall back to clean full rebuild
                try:
                    ws = self.index_workspace(valid_id, max_file_size_kb=max_file_size_kb, max_unit_lines=max_unit_lines)
                    return ws, {"mode": "full_fallback_due_to_error", "error": str(e)}
                except Exception as fb_err:
                    ws.metadata.last_error = f"Incremental failed: {e}; Fallback failed: {fb_err}"
                    self._save_metadata(ws)
                    raise WorkspaceError(ws.metadata.last_error) from fb_err
            finally:
                if staging_dir.exists():
                    shutil.rmtree(staging_dir, ignore_errors=True)

    def _save_metadata(self, ws: Workspace) -> None:
        """Persist workspace metadata to disk."""
        if ws.metadata.is_canonical:
            return
        meta_file = ws.workspace_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(ws.metadata.to_dict(mask_absolute_path=False), f, indent=2)
