"""Repository relationship graph construction and analysis for CodeSage.

Extracts syntax-aware relationships across indexed code units and files:
- function-to-class membership (MEMBER_OF, CONTAINS_METHOD)
- class inheritance (EXTENDS)
- file definitions (DEFINES)
- intra-repo module and file imports (IMPORTS_FILE, IMPORTS_SYMBOL)
- reliable, syntax-resolved calls (CALLS, CONSTRUCTS)
- explicit tracking of unresolved external/dynamic references (without guessing)
"""

import ast
import json
import posixpath
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class RelationshipType(str, Enum):
    """Typed relationship edges in the repository graph."""
    DEFINES = "DEFINES"                    # file -> unit
    MEMBER_OF = "MEMBER_OF"                # method -> class
    CONTAINS_METHOD = "CONTAINS_METHOD"    # class -> method
    EXTENDS = "EXTENDS"                    # class -> base class
    IMPORTS_FILE = "IMPORTS_FILE"          # file -> file
    IMPORTS_SYMBOL = "IMPORTS_SYMBOL"      # file/unit -> target unit
    CALLS = "CALLS"                        # caller unit -> callee unit
    CONSTRUCTS = "CONSTRUCTS"              # caller unit -> class unit (instantiation)


@dataclass(frozen=True)
class GraphNode:
    """Stable node representing a file or a code unit."""
    id: str                            # Stable identifier: 'file:<path>' or 'unit:<unit_id>'
    node_type: str                     # 'file' or 'code_unit'
    name: str                          # Symbol name or filename
    path: str                          # Repository-relative path
    language: str                      # 'javascript', 'dart', etc.
    unit_type: Optional[str] = None    # 'class', 'function', 'method', 'line_chunk'
    start_line: Optional[int] = None
    end_line: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphEdge:
    """Typed relationship edge between two graph nodes."""
    source: str                        # Source node ID
    target: str                        # Target node ID
    relationship: str                  # RelationshipType value
    confidence: float = 1.0            # 1.0 for reliable syntax resolution
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class UnresolvedReference:
    """Explicitly recorded reference that could not be resolved to an indexed unit."""
    caller_id: str                     # Node ID where reference occurred
    symbol_name: str                   # Identifier or call target
    call_type: str                     # 'external_module', 'third_party_package', 'dynamic_dispatch', 'unindexed_symbol'
    line: Optional[int] = None         # Line number if identifiable
    reason: str = ""                   # Why resolution was not guessed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RepositoryGraph:
    """In-memory representation of the repository relationship graph."""

    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.unresolved_references: List[UnresolvedReference] = []

        # Adjacency indexes for fast traversal
        self.outgoing: Dict[str, List[GraphEdge]] = {}
        self.incoming: Dict[str, List[GraphEdge]] = {}
        self._edge_keys: Set[Tuple[str, str, str]] = set()

    def add_node(self, node: GraphNode) -> None:
        """Add a node if not already present."""
        if node.id not in self.nodes:
            self.nodes[node.id] = node
            self.outgoing[node.id] = []
            self.incoming[node.id] = []

    def add_edge(self, edge: GraphEdge) -> None:
        """Add a typed edge, preventing duplicates and ensuring node registration."""
        edge_key = (edge.source, edge.relationship, edge.target)
        if edge_key in self._edge_keys:
            return

        self._edge_keys.add(edge_key)
        self.edges.append(edge)

        self.outgoing.setdefault(edge.source, []).append(edge)
        self.incoming.setdefault(edge.target, []).append(edge)

    def add_unresolved(self, ref: UnresolvedReference) -> None:
        """Record an unresolved reference explicitly."""
        self.unresolved_references.append(ref)

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self.nodes.get(node_id)

    def get_unit_node(self, unit_id: str) -> Optional[GraphNode]:
        return self.nodes.get(f"unit:{unit_id}")

    def get_neighbors(
        self,
        node_id: str,
        rel_types: Optional[Set[str]] = None,
        direction: str = "both",
    ) -> List[Tuple[str, str, float]]:
        """Get connected neighbor IDs with relationship type and confidence.
        
        direction: 'outgoing', 'incoming', or 'both'.
        """
        results: List[Tuple[str, str, float]] = []

        if direction in ("outgoing", "both") and node_id in self.outgoing:
            for e in self.outgoing[node_id]:
                if rel_types is None or e.relationship in rel_types:
                    results.append((e.target, e.relationship, e.confidence))

        if direction in ("incoming", "both") and node_id in self.incoming:
            for e in self.incoming[node_id]:
                if rel_types is None or e.relationship in rel_types:
                    results.append((e.source, e.relationship, e.confidence))

        return results

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to deterministic JSON-serializable dictionary."""
        sorted_nodes = [self.nodes[k].to_dict() for k in sorted(self.nodes.keys())]
        sorted_edges = [
            e.to_dict()
            for e in sorted(self.edges, key=lambda x: (x.source, x.relationship, x.target))
        ]
        sorted_unresolved = [
            u.to_dict()
            for u in sorted(
                self.unresolved_references,
                key=lambda x: (x.caller_id, x.symbol_name, x.line or 0),
            )
        ]

        # Summary statistics
        edge_counts: Dict[str, int] = {}
        for e in self.edges:
            edge_counts[e.relationship] = edge_counts.get(e.relationship, 0) + 1

        unresolved_counts: Dict[str, int] = {}
        for u in self.unresolved_references:
            unresolved_counts[u.call_type] = unresolved_counts.get(u.call_type, 0) + 1

        return {
            "version": "1.0.0",
            "counts": {
                "total_nodes": len(sorted_nodes),
                "total_edges": len(sorted_edges),
                "total_unresolved": len(sorted_unresolved),
                "edges_by_relationship": edge_counts,
                "unresolved_by_type": unresolved_counts,
            },
            "nodes": sorted_nodes,
            "edges": sorted_edges,
            "unresolved_references": sorted_unresolved,
        }

    def save_json(self, output_path: Path | str) -> None:
        """Serialize graph to formatted JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save(self, output_path: Path | str) -> None:
        """Alias for save_json."""
        self.save_json(output_path)

    @classmethod
    def load(cls, json_path: Path | str) -> "RepositoryGraph":
        """Alias for load_json."""
        return cls.load_json(json_path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepositoryGraph":
        """Reconstruct graph from dictionary."""
        graph = cls()
        for nd in data.get("nodes", []):
            node = GraphNode(
                id=nd["id"],
                node_type=nd["node_type"],
                name=nd["name"],
                path=nd["path"],
                language=nd["language"],
                unit_type=nd.get("unit_type"),
                start_line=nd.get("start_line"),
                end_line=nd.get("end_line"),
            )
            graph.add_node(node)

        for ed in data.get("edges", []):
            edge = GraphEdge(
                source=ed["source"],
                target=ed["target"],
                relationship=ed["relationship"],
                confidence=ed.get("confidence", 1.0),
                metadata=ed.get("metadata", {}),
            )
            graph.add_edge(edge)

        for ud in data.get("unresolved_references", []):
            unres = UnresolvedReference(
                caller_id=ud["caller_id"],
                symbol_name=ud["symbol_name"],
                call_type=ud["call_type"],
                line=ud.get("line"),
                reason=ud.get("reason", ""),
            )
            graph.add_unresolved(unres)

        return graph

    @classmethod
    def load_json(cls, json_path: Path | str) -> "RepositoryGraph":
        """Load graph from JSON file."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# Regex patterns for reliable syntax extraction without guesswork
_JS_REQUIRE_PATTERN = re.compile(
    r"""(?:const|let|var)\s+(?:\{([^}]+)\}|([a-zA-Z0-9_$]+))\s*=\s*require\(\s*['"]([^'"]+)['"]\s*\)"""
)
_DART_IMPORT_PATTERN = re.compile(
    r"""import\s+['"]([^'"]+)['"](?:\s+as\s+([a-zA-Z0-9_$]+))?(?:\s+show\s+([^;]+))?;"""
)
_DART_CLASS_EXTENDS_PATTERN = re.compile(
    r"""class\s+([a-zA-Z0-9_$]+)(?:<[^>]+>)?\s+extends\s+([a-zA-Z0-9_$]+)"""
)
_CALL_IDENT_PATTERN = re.compile(r"""\b([a-zA-Z0-9_$]+)\s*\(""")


class RepositoryGraphBuilder:
    """Deterministic, syntax-aware builder for repository relationship graphs."""

    def __init__(self, repo_root: Optional[Path | str] = None) -> None:
        self.repo_root = Path(repo_root) if repo_root else None

    def build(self, units: List[Any]) -> RepositoryGraph:
        """Build relationship graph from indexed code units and repository files."""
        graph = RepositoryGraph()
        units = [u.to_dict() if hasattr(u, "to_dict") else u for u in units]

        # 1. Index code units and files
        file_paths: Set[str] = set()
        units_by_id: Dict[str, Dict[str, Any]] = {}
        units_by_file: Dict[str, List[Dict[str, Any]]] = {}
        classes_by_file_and_name: Dict[Tuple[str, str], Dict[str, Any]] = {}
        functions_by_file_and_name: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for u in units:
            uid = u["unit_id"]
            path = u.get("repository_relative_path") or u.get("rel_path", "")
            units_by_id[uid] = u
            file_paths.add(path)
            units_by_file.setdefault(path, []).append(u)

            sym_name = u.get("symbol_name")
            unit_type = u.get("unit_type")

            if sym_name:
                if unit_type == "class":
                    classes_by_file_and_name[(path, sym_name)] = u
                elif unit_type in ("function", "method"):
                    functions_by_file_and_name[(path, sym_name)] = u

            # Add Unit Node
            u_node = GraphNode(
                id=f"unit:{uid}",
                node_type="code_unit",
                name=sym_name or "anon",
                path=path,
                language=u.get("language", "unknown"),
                unit_type=unit_type,
                start_line=u.get("start_line"),
                end_line=u.get("end_line"),
            )
            graph.add_node(u_node)

        # Add File Nodes and DEFINES edges
        for p in sorted(file_paths):
            f_lang = units_by_file[p][0].get("language", "unknown") if p in units_by_file else "unknown"
            f_node = GraphNode(
                id=f"file:{p}",
                node_type="file",
                name=Path(p).name,
                path=p,
                language=f_lang,
            )
            graph.add_node(f_node)

            for u in units_by_file.get(p, []):
                graph.add_edge(
                    GraphEdge(
                        source=f"file:{p}",
                        target=f"unit:{u['unit_id']}",
                        relationship=RelationshipType.DEFINES.value,
                    )
                )

        # 2. Extract Class-Method Membership & Class Inheritance
        self._extract_class_memberships_and_inheritance(
            units, classes_by_file_and_name, graph
        )

        # 3. Extract File Imports & Module Dependencies
        file_to_file_imports, imported_symbols_by_file = self._extract_imports(
            file_paths, units_by_file, graph
        )

        # 4. Resolve Reliable Calls & Record Unresolved Invocations
        self._extract_calls(
            units,
            units_by_file,
            classes_by_file_and_name,
            functions_by_file_and_name,
            file_to_file_imports,
            imported_symbols_by_file,
            graph,
        )

        return graph

    def _extract_class_memberships_and_inheritance(
        self,
        units: List[Dict[str, Any]],
        classes_by_file_and_name: Dict[Tuple[str, str], Dict[str, Any]],
        graph: RepositoryGraph,
    ) -> None:
        """Resolve function-to-class membership and class inheritance."""
        for u in units:
            path = u.get("repository_relative_path") or u.get("rel_path", "")
            unit_type = u.get("unit_type")
            parent_symbol = u.get("parent_symbol")
            uid = u["unit_id"]

            # Method to Class Membership
            if unit_type == "method" and parent_symbol:
                class_unit = classes_by_file_and_name.get((path, parent_symbol))
                if class_unit:
                    class_id = class_unit["unit_id"]
                    # MEMBER_OF: method -> class
                    graph.add_edge(
                        GraphEdge(
                            source=f"unit:{uid}",
                            target=f"unit:{class_id}",
                            relationship=RelationshipType.MEMBER_OF.value,
                        )
                    )
                    # CONTAINS_METHOD: class -> method
                    graph.add_edge(
                        GraphEdge(
                            source=f"unit:{class_id}",
                            target=f"unit:{uid}",
                            relationship=RelationshipType.CONTAINS_METHOD.value,
                        )
                    )

            # Class Inheritance (extends)
            if unit_type == "class":
                source_code = u.get("source_text") or u.get("source_code", "")
                m = _DART_CLASS_EXTENDS_PATTERN.search(source_code)
                if m:
                    sub_class, base_class = m.group(1), m.group(2)
                    # Look for base class in same file or elsewhere in repo
                    base_unit = classes_by_file_and_name.get((path, base_class))
                    if not base_unit:
                        for (bp, bc), bu in classes_by_file_and_name.items():
                            if bc == base_class:
                                base_unit = bu
                                break

                    if base_unit:
                        graph.add_edge(
                            GraphEdge(
                                source=f"unit:{uid}",
                                target=f"unit:{base_unit['unit_id']}",
                                relationship=RelationshipType.EXTENDS.value,
                                metadata={"sub_class": sub_class, "base_class": base_class},
                            )
                        )
                    else:
                        # Base class is external (e.g. StatelessWidget, StatefulWidget, ChangeNotifier)
                        graph.add_unresolved(
                            UnresolvedReference(
                                caller_id=f"unit:{uid}",
                                symbol_name=base_class,
                                call_type="third_party_package",
                                line=u.get("start_line"),
                                reason="Base class extends external framework class",
                            )
                        )

    def _resolve_relative_path(self, src_path: str, import_str: str, file_paths: Set[str]) -> Optional[str]:
        """Deterministically resolve relative require/import paths to indexed repository files."""
        src_dir = posixpath.dirname(src_path.replace("\\", "/"))

        candidates: List[str] = []

        # Handle Dart 'package:<pkg>/...' imports
        if import_str.startswith("package:"):
            pkg_parts = import_str[len("package:"):].split("/", 1)
            if len(pkg_parts) == 2:
                pkg_name, rest = pkg_parts
                # Standard Dart package structure maps package:<pkg>/<rest> to lib/<rest>
                candidates.append(f"lib/{rest}")
                candidates.append(f"frontend/lib/{rest}")
                # Dynamically locate any indexed file ending with lib/<rest>
                for fp in file_paths:
                    norm_fp = fp.replace("\\", "/")
                    if norm_fp.endswith(f"/lib/{rest}") or norm_fp == f"lib/{rest}":
                        candidates.append(norm_fp)
            if not candidates:
                # External Flutter/Dart package (e.g. package:flutter/...)
                return None
        elif import_str.startswith("."):
            # Relative imports in JS or Dart
            candidates = [
                posixpath.normpath(posixpath.join(src_dir, import_str)),
                posixpath.normpath(posixpath.join(src_dir, f"{import_str}.js")),
                posixpath.normpath(posixpath.join(src_dir, f"{import_str}.dart")),
                posixpath.normpath(posixpath.join(src_dir, import_str, "index.js")),
            ]
        else:
            # Bare import like 'express', 'multer', 'pdf-lib', etc.
            return None

        # Check candidate match against known repository file paths
        normalized_file_paths = {p.replace("\\", "/"): p for p in file_paths}

        for cand in candidates:
            norm_cand = cand.replace("\\", "/")
            if norm_cand.startswith("./"):
                norm_cand = norm_cand[2:]

            if norm_cand in normalized_file_paths:
                return normalized_file_paths[norm_cand]

        return None

    def _extract_imports(
        self,
        file_paths: Set[str],
        units_by_file: Dict[str, List[Dict[str, Any]]],
        graph: RepositoryGraph,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, str]]]:
        """Extract file imports and imported symbol maps."""
        file_to_file_imports: Dict[str, Set[str]] = {p: set() for p in file_paths}
        imported_symbols_by_file: Dict[str, Dict[str, str]] = {p: {} for p in file_paths}

        for p in file_paths:
            # Gather full text either from filesystem or by assembling units
            content = ""
            if self.repo_root:
                fp = self.repo_root / p
                if fp.exists():
                    try:
                        content = fp.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        content = ""

            if not content:
                content = "\n".join(u.get("source_text", "") for u in units_by_file.get(p, []))

            # 1. JavaScript requires and imports
            if p.endswith(".js") or p.endswith(".jsx"):
                for m in _JS_REQUIRE_PATTERN.finditer(content):
                    named_symbols_raw, default_symbol, import_target = m.groups()
                    resolved_file = self._resolve_relative_path(p, import_target, file_paths)

                    if resolved_file:
                        file_to_file_imports[p].add(resolved_file)
                        graph.add_edge(
                            GraphEdge(
                                source=f"file:{p}",
                                target=f"file:{resolved_file}",
                                relationship=RelationshipType.IMPORTS_FILE.value,
                            )
                        )

                        # Handle named symbol imports: const { mergePdfs, ... } = require(...)
                        if named_symbols_raw:
                            sym_names = [s.strip() for s in named_symbols_raw.split(",") if s.strip()]
                            for sym in sym_names:
                                # Look for symbol unit in resolved_file
                                target_unit = None
                                for tu in units_by_file.get(resolved_file, []):
                                    if tu.get("symbol_name") == sym:
                                        target_unit = tu
                                        break
                                if target_unit:
                                    imported_symbols_by_file[p][sym] = target_unit["unit_id"]
                                    graph.add_edge(
                                        GraphEdge(
                                            source=f"file:{p}",
                                            target=f"unit:{target_unit['unit_id']}",
                                            relationship=RelationshipType.IMPORTS_SYMBOL.value,
                                            metadata={"symbol": sym},
                                        )
                                    )
                        elif default_symbol:
                            imported_symbols_by_file[p][default_symbol] = f"file:{resolved_file}"
                    else:
                        # External package like 'express', 'multer', 'pdf-lib'
                        graph.add_unresolved(
                            UnresolvedReference(
                                caller_id=f"file:{p}",
                                symbol_name=import_target,
                                call_type="external_module",
                                reason="External Node.js package or built-in module",
                            )
                        )

            # 2. Dart imports
            elif p.endswith(".dart"):
                for m in _DART_IMPORT_PATTERN.finditer(content):
                    import_target = m.group(1)
                    resolved_file = self._resolve_relative_path(p, import_target, file_paths)

                    if resolved_file:
                        file_to_file_imports[p].add(resolved_file)
                        graph.add_edge(
                            GraphEdge(
                                source=f"file:{p}",
                                target=f"file:{resolved_file}",
                                relationship=RelationshipType.IMPORTS_FILE.value,
                            )
                        )
                    else:
                        # External package like 'package:flutter/material.dart'
                        graph.add_unresolved(
                            UnresolvedReference(
                                caller_id=f"file:{p}",
                                symbol_name=import_target,
                                call_type="third_party_package",
                                reason="External Flutter/Dart package or SDK library",
                            )
                        )

        return file_to_file_imports, imported_symbols_by_file

    def _extract_calls(
        self,
        units: List[Dict[str, Any]],
        units_by_file: Dict[str, List[Dict[str, Any]]],
        classes_by_file_and_name: Dict[Tuple[str, str], Dict[str, Any]],
        functions_by_file_and_name: Dict[Tuple[str, str], Dict[str, Any]],
        file_to_file_imports: Dict[str, Set[str]],
        imported_symbols_by_file: Dict[str, Dict[str, str]],
        graph: RepositoryGraph,
    ) -> None:
        """Resolve reliable call relationships and record unresolved invocations."""
        for u in units:
            uid = u["unit_id"]
            path = u.get("repository_relative_path") or u.get("rel_path", "")
            source_text = u.get("source_text") or u.get("source_code", "")
            parent_symbol = u.get("parent_symbol")
            u_name = u.get("symbol_name")

            # Collect call candidate identifiers in this unit
            call_matches = _CALL_IDENT_PATTERN.findall(source_text)

            for callee_name in set(call_matches):
                if callee_name in ("if", "for", "while", "switch", "catch", "return", "require", "super"):
                    continue

                if callee_name == u_name:
                    # Direct recursive call
                    graph.add_edge(
                        GraphEdge(
                            source=f"unit:{uid}",
                            target=f"unit:{uid}",
                            relationship=RelationshipType.CALLS.value,
                            metadata={"type": "recursive"},
                        )
                    )
                    continue

                target_unit_id: Optional[str] = None
                rel_type = RelationshipType.CALLS.value

                # 1. Check if callee is an intra-class method
                if parent_symbol:
                    sibling_unit = functions_by_file_and_name.get((path, callee_name))
                    if sibling_unit and sibling_unit.get("parent_symbol") == parent_symbol:
                        target_unit_id = sibling_unit["unit_id"]

                # 2. Check if callee is a top-level function in the same file
                if not target_unit_id:
                    file_func = functions_by_file_and_name.get((path, callee_name))
                    if file_func and file_func["unit_id"] != uid:
                        target_unit_id = file_func["unit_id"]

                # 3. Check if callee is a class in the same file (instantiation)
                if not target_unit_id:
                    file_cls = classes_by_file_and_name.get((path, callee_name))
                    if file_cls:
                        target_unit_id = file_cls["unit_id"]
                        rel_type = RelationshipType.CONSTRUCTS.value

                # 4. Check if callee is an explicitly imported symbol in this file
                if not target_unit_id and callee_name in imported_symbols_by_file.get(path, {}):
                    sym_ref = imported_symbols_by_file[path][callee_name]
                    if sym_ref.startswith("unit:"):
                        target_unit_id = sym_ref.replace("unit:", "")
                    elif not sym_ref.startswith("file:"):
                        target_unit_id = sym_ref

                # 5. Check if callee is a class/function in an imported file
                if not target_unit_id:
                    imported_files = file_to_file_imports.get(path, set())
                    matching_imported_units = []
                    for imp_f in imported_files:
                        if (imp_f, callee_name) in classes_by_file_and_name:
                            matching_imported_units.append((classes_by_file_and_name[(imp_f, callee_name)], RelationshipType.CONSTRUCTS.value))
                        elif (imp_f, callee_name) in functions_by_file_and_name:
                            matching_imported_units.append((functions_by_file_and_name[(imp_f, callee_name)], RelationshipType.CALLS.value))

                    if len(matching_imported_units) == 1:
                        target_unit_id = matching_imported_units[0][0]["unit_id"]
                        rel_type = matching_imported_units[0][1]
                    elif len(matching_imported_units) > 1:
                        # Ambiguous: multiple imported targets define the same symbol name
                        graph.add_unresolved(
                            UnresolvedReference(
                                caller_id=f"unit:{uid}",
                                symbol_name=callee_name,
                                call_type="ambiguous_dispatch",
                                reason=f"Symbol '{callee_name}' defined in multiple imported files",
                            )
                        )
                        continue

                # Add resolved edge or record unresolved invocation
                if target_unit_id:
                    graph.add_edge(
                        GraphEdge(
                            source=f"unit:{uid}",
                            target=f"unit:{target_unit_id}",
                            relationship=rel_type,
                            metadata={"symbol": callee_name},
                        )
                    )
                else:
                    graph.add_unresolved(
                        UnresolvedReference(
                            caller_id=f"unit:{uid}",
                            symbol_name=callee_name,
                            call_type="unindexed_symbol",
                            reason="Unresolved external invocation or dynamic member dispatch",
                        )
                    )
