#!/usr/bin/env python3
"""Build and export the repository relationship graph for CodeSage."""

import argparse
import json
import sys
import time
from pathlib import Path

# Add 'src' to python path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from codesage.graph import RepositoryGraphBuilder
from codesage.retrieval import load_code_units_from_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic repository relationship graph for CodeSage."
    )
    parser.add_argument(
        "--index",
        type=str,
        default=str(project_root / "data" / "code_units.jsonl"),
        help="Path to code_units.jsonl index (default: 'data/code_units.jsonl')",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(project_root.parent / "DocuCharm"),
        help="Path to target repository root (default: '../DocuCharm')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(project_root / "data" / "relationship_graph.json"),
        help="Output path for relationship graph JSON (default: 'data/relationship_graph.json')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = Path(args.index)
    output_path = Path(args.output)
    repo_root = Path(args.repo_root) if args.repo_root else None

    if not index_path.exists():
        print(f"Error: Index file not found at '{index_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading code units from: {index_path}")
    units = load_code_units_from_jsonl(index_path)
    print(f"Loaded {len(units)} code units.")

    print(f"Building relationship graph (repo root: {repo_root})...")
    t0 = time.perf_counter()
    builder = RepositoryGraphBuilder(repo_root=repo_root)
    graph = builder.build(units)
    elapsed = time.perf_counter() - t0

    graph_dict = graph.to_dict()
    counts = graph_dict["counts"]

    print(f"Graph construction completed in {elapsed:.3f}s:")
    print(f"  - Total nodes: {counts['total_nodes']}")
    print(f"  - Total edges: {counts['total_edges']}")
    print(f"  - Unresolved references: {counts['total_unresolved']}")
    print("\nEdges by relationship:")
    for rel, count in counts["edges_by_relationship"].items():
        print(f"  - {rel}: {count}")

    print("\nUnresolved references by type:")
    for call_type, count in counts["unresolved_by_type"].items():
        print(f"  - {call_type}: {count}")

    print(f"\nSaving graph to: {output_path}")
    graph.save_json(output_path)
    print("Done.")


if __name__ == "__main__":
    main()
