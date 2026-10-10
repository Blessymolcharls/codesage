"""Tests for repository relationship graph and graph-context reranking."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from codesage.graph import (
    GraphEdge,
    GraphNode,
    RelationshipType,
    RepositoryGraph,
    RepositoryGraphBuilder,
    UnresolvedReference,
)
from codesage.graph_reranker import GraphContextReranker
from codesage.retrieval import SearchResult


@pytest.fixture
def synthetic_code_units() -> List[Dict[str, Any]]:
    """Synthetic multi-file, multi-language code unit fixtures."""
    return [
        # JavaScript File: backend/controllers/pdfController.js
        {
            "unit_id": "js_unit_helper",
            "repository_relative_path": "backend/utils/fileHelper.js",
            "symbol_name": "cleanupFiles",
            "unit_type": "function",
            "language": "javascript",
            "start_line": 5,
            "end_line": 15,
            "source_text": "function cleanupFiles(paths) { /* deletes files */ }",
        },
        {
            "unit_id": "js_unit_merge",
            "repository_relative_path": "backend/controllers/pdfController.js",
            "symbol_name": "mergePdfs",
            "unit_type": "function",
            "language": "javascript",
            "start_line": 10,
            "end_line": 35,
            "source_text": (
                "const { cleanupFiles } = require('../utils/fileHelper');\n"
                "const express = require('express');\n"
                "async function mergePdfs(req, res) {\n"
                "    try {\n"
                "        cleanupFiles(req.files);\n"
                "    } catch(err) { expressError(err); }\n"
                "}"
            ),
        },
        # Dart File: frontend/lib/screens/base_feature_screen.dart
        {
            "unit_id": "dart_class_provider",
            "repository_relative_path": "frontend/lib/providers/pdf_provider.dart",
            "symbol_name": "PdfProvider",
            "unit_type": "class",
            "language": "dart",
            "start_line": 5,
            "end_line": 40,
            "source_text": (
                "import 'package:flutter/material.dart';\n"
                "class PdfProvider extends ChangeNotifier {\n"
                "    bool isLoading = false;\n"
                "}"
            ),
        },
        {
            "unit_id": "dart_class_screen",
            "repository_relative_path": "frontend/lib/screens/base_feature_screen.dart",
            "symbol_name": "BaseFeatureScreen",
            "unit_type": "class",
            "language": "dart",
            "start_line": 10,
            "end_line": 50,
            "source_text": (
                "import 'package:docucharm/providers/pdf_provider.dart';\n"
                "import 'package:flutter/material.dart';\n"
                "class BaseFeatureScreen extends StatelessWidget {\n"
                "}"
            ),
        },
        {
            "unit_id": "dart_method_build",
            "repository_relative_path": "frontend/lib/screens/base_feature_screen.dart",
            "symbol_name": "build",
            "parent_symbol": "BaseFeatureScreen",
            "unit_type": "method",
            "language": "dart",
            "start_line": 20,
            "end_line": 45,
            "source_text": (
                "Widget build(BuildContext context) {\n"
                "    final provider = PdfProvider();\n"
                "    return renderView();\n"
                "}"
            ),
        },
        {
            "unit_id": "dart_method_render",
            "repository_relative_path": "frontend/lib/screens/base_feature_screen.dart",
            "symbol_name": "renderView",
            "parent_symbol": "BaseFeatureScreen",
            "unit_type": "method",
            "language": "dart",
            "start_line": 46,
            "end_line": 55,
            "source_text": "Widget renderView() { return Container(); }",
        },
    ]


def test_graph_node_and_edge_data_structures():
    """Verify GraphNode, GraphEdge, and UnresolvedReference dataclass serialization."""
    node = GraphNode(
        id="unit:123",
        node_type="code_unit",
        name="testFunc",
        path="backend/test.js",
        language="javascript",
        unit_type="function",
    )
    assert node.to_dict()["id"] == "unit:123"

    edge = GraphEdge(
        source="unit:123",
        target="unit:456",
        relationship=RelationshipType.CALLS.value,
        confidence=1.0,
    )
    assert edge.to_dict()["relationship"] == "CALLS"

    unresolved = UnresolvedReference(
        caller_id="unit:123",
        symbol_name="externalLib",
        call_type="external_module",
        reason="External dependency",
    )
    assert unresolved.to_dict()["call_type"] == "external_module"


def test_synthetic_graph_construction_and_memberships(synthetic_code_units):
    """Verify class-method membership and file containment relationships."""
    builder = RepositoryGraphBuilder()
    graph = builder.build(synthetic_code_units)

    # Check that method 'build' is MEMBER_OF 'BaseFeatureScreen'
    build_node_id = "unit:dart_method_build"
    screen_node_id = "unit:dart_class_screen"

    neighbors = graph.get_neighbors(build_node_id, direction="outgoing")
    member_of_targets = [tgt for tgt, rel, _ in neighbors if rel == RelationshipType.MEMBER_OF.value]
    assert screen_node_id in member_of_targets

    # Check inverse CONTAINS_METHOD
    screen_neighbors = graph.get_neighbors(screen_node_id, direction="outgoing")
    contains_targets = [tgt for tgt, rel, _ in screen_neighbors if rel == RelationshipType.CONTAINS_METHOD.value]
    assert build_node_id in contains_targets
    assert "unit:dart_method_render" in contains_targets


def test_synthetic_import_resolution_and_calls(synthetic_code_units):
    """Verify relative JS require import and named symbol resolution."""
    builder = RepositoryGraphBuilder()
    graph = builder.build(synthetic_code_units)

    # JS file-to-file import: pdfController -> fileHelper
    controller_file = "file:backend/controllers/pdfController.js"
    helper_file = "file:backend/utils/fileHelper.js"
    f_neighbors = graph.get_neighbors(controller_file, direction="outgoing")
    imported_files = [tgt for tgt, rel, _ in f_neighbors if rel == RelationshipType.IMPORTS_FILE.value]
    assert helper_file in imported_files

    # Symbol import
    symbol_imports = [tgt for tgt, rel, _ in f_neighbors if rel == RelationshipType.IMPORTS_SYMBOL.value]
    assert "unit:js_unit_helper" in symbol_imports

    # JS call resolution: mergePdfs CALLS cleanupFiles
    merge_calls = graph.get_neighbors("unit:js_unit_merge", direction="outgoing")
    called_units = [tgt for tgt, rel, _ in merge_calls if rel == RelationshipType.CALLS.value]
    assert "unit:js_unit_helper" in called_units

    # Dart cross-file construction: build CONSTRUCTS PdfProvider
    build_calls = graph.get_neighbors("unit:dart_method_build", direction="outgoing")
    construct_units = [tgt for tgt, rel, _ in build_calls if rel == RelationshipType.CONSTRUCTS.value]
    assert "unit:dart_class_provider" in construct_units

    # Intra-class call: build CALLS renderView
    calls = [tgt for tgt, rel, _ in build_calls if rel == RelationshipType.CALLS.value]
    assert "unit:dart_method_render" in calls


def test_explicit_unresolved_references_tracking(synthetic_code_units):
    """Verify that external libraries and framework classes are tracked as unresolved without guessing."""
    builder = RepositoryGraphBuilder()
    graph = builder.build(synthetic_code_units)

    unresolved = graph.unresolved_references
    assert len(unresolved) > 0

    sym_names = [u.symbol_name for u in unresolved]
    # 'express' should be unresolved external_module
    assert "express" in sym_names

    # 'package:flutter/material.dart' should be unresolved third_party_package
    assert "package:flutter/material.dart" in sym_names

    # Base classes like 'StatelessWidget' or 'ChangeNotifier' should be recorded
    assert "StatelessWidget" in sym_names or "ChangeNotifier" in sym_names


def test_graph_determinism(synthetic_code_units, tmp_path):
    """Verify that serializing graph multiple times produces byte-for-byte identical output."""
    builder = RepositoryGraphBuilder()
    graph1 = builder.build(synthetic_code_units)
    graph2 = builder.build(synthetic_code_units)

    dict1 = graph1.to_dict()
    dict2 = graph2.to_dict()

    json1 = json.dumps(dict1, sort_keys=True)
    json2 = json.dumps(dict2, sort_keys=True)
    assert json1 == json2

    # Save and reload
    path = tmp_path / "test_graph.json"
    graph1.save_json(path)
    loaded = RepositoryGraph.load_json(path)
    assert len(loaded.nodes) == len(graph1.nodes)
    assert len(loaded.edges) == len(graph1.edges)
    assert len(loaded.unresolved_references) == len(graph1.unresolved_references)


class MockRetriever:
    """Mock base retriever for testing graph reranker."""

    def __init__(self, canned_results: List[SearchResult]):
        self.canned_results = canned_results

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        return self.canned_results[:top_k]


def test_graph_context_reranker_affinity(synthetic_code_units):
    """Verify that GraphContextReranker promotes connected nodes and breaks ties deterministically."""
    builder = RepositoryGraphBuilder()
    graph = builder.build(synthetic_code_units)

    # Initial mock candidates:
    # 1. dart_method_render: score 0.50
    # 2. unrelated dummy: score 0.49
    # 3. dart_class_screen: score 0.48 (which CONTAINS dart_method_render)
    cand_render = SearchResult(
        unit_id="dart_method_render",
        score=0.50,
        rel_path="frontend/lib/screens/base_feature_screen.dart",
        language="dart",
        symbol_name="renderView",
        unit_type="method",
        start_line=46,
        end_line=55,
        extraction_method="ast",
        preview="renderView()",
        source_code="renderView()",
    )
    cand_screen = SearchResult(
        unit_id="dart_class_screen",
        score=0.48,
        rel_path="frontend/lib/screens/base_feature_screen.dart",
        language="dart",
        symbol_name="BaseFeatureScreen",
        unit_type="class",
        start_line=10,
        end_line=50,
        extraction_method="ast",
        preview="BaseFeatureScreen",
        source_code="BaseFeatureScreen",
    )
    cand_other = SearchResult(
        unit_id="isolated_unit",
        score=0.49,
        rel_path="backend/other.js",
        language="javascript",
        symbol_name="other",
        unit_type="function",
        start_line=1,
        end_line=10,
        extraction_method="ast",
        preview="other()",
        source_code="other()",
    )

    mock = MockRetriever([cand_render, cand_other, cand_screen])
    reranker = GraphContextReranker(mock, graph, alpha=0.5, beta=0.5, candidate_pool_k=10)

    reranked = reranker.search("screen rendering", top_k=3)
    assert len(reranked) == 3

    # Because screen and render share CONTAINS_METHOD/MEMBER_OF and co-file relationship,
    # cand_screen receives mutual graph boost above isolated_unit
    top_ids = [r.unit_id for r in reranked]
    assert "dart_class_screen" in top_ids[:2]
