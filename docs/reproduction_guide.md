# CodeSage Reproduction & Experiment Guide

This guide provides deterministic, step-by-step instructions to reproduce all CodeSage artifacts, indexes, benchmarks, evaluations, and web services from scratch.

---

## 1. System Requirements & Prerequisites

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.10 – 3.13
- **Node.js**: >= 18.17 (LTS recommended)
- **Disk Space**: ~2 GB (for PyTorch CPU dependencies and sentence-transformer weights)
- **RAM**: >= 4 GB RAM recommended

---

## 2. Environment Setup

### 2.1 Python Environment
Clone the repository and create an isolated virtual environment:

```bash
# Navigate to the repository root
cd codesage

# Create and activate virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

Install backend dependencies:
```bash
pip install -r requirements.txt
```

### 2.2 Next.js Web Frontend Environment
Install web application dependencies:

```bash
cd web
npm install
cd ..
```

---

## 3. Step-by-Step Reproduction Pipeline

### Step 1: Repository Ingestion & Code-Unit Extraction
Scans the DocuCharm application (or any configured target) and extracts grounded code units with Tree-sitter parsers:

```bash
python -m codesage.scanner --repo-path ../DocuCharm --output data/code_units.jsonl --manifest data/manifest.json
```
- **Output**: `data/code_units.jsonl` (89 units) and `data/manifest.json`.
- **Validation**: Checks deterministic ordering, skips secrets and lockfiles, bounds oversized units.

---

### Step 2: Build the Repository Relationship Graph
Constructs the syntax-aware relationship graph linking functions, classes, methods, imports, and calls:

```bash
python scripts/build_graph.py --units data/code_units.jsonl --repo ../DocuCharm --output data/relationship_graph.json
```
- **Output**: `data/relationship_graph.json` (105 nodes, 378 edges, 876 explicit unresolved references).
- **Validation**: Verified deterministic SHA-256 graph generation across repeated runs.

---

### Step 3: Run Interactive CLI Code Search
Test lexical and semantic retrieval directly from the terminal:

```bash
# TF-IDF Lexical Search
python scripts/search.py "merge multiple pdf files" --model tfidf --top-k 5

# General-Purpose Semantic Search (all-MiniLM-L6-v2)
python scripts/search.py "upload files with progress" --model semantic --top-k 5

# Code-Specific Transformer (st-codesearch-distilroberta-base)
python scripts/search.py "health check endpoint" --model code_transformer --top-k 5

# Search with Graph Context Reranking
python scripts/search.py "merge multiple pdf files" --model tfidf --graph data/relationship_graph.json --top-k 5
```

---

### Step 4: Run the Research Evaluation & Ablation Study
Executes the unified evaluation runner across all model variants and splits (dev, test, negative control):

```bash
python scripts/run_experiments.py --units data/code_units.jsonl --queries data/evaluation_queries.jsonl --output-dir data/experiments
```
- **Outputs**:
  - `data/experiments/study_summary.json` (aggregate metrics, bootstrap 95% CIs)
  - `data/experiments/study_summary.csv`
  - `data/experiments/per_query_rankings.csv`
  - `data/experiments/ablation_significance.json` (paired Wilcoxon and t-tests)

---

### Step 5: Run the Graph Context Reranking Benchmark
Compares base retrieval against graph-boosted retrieval on the same evaluation dataset:

```bash
python scripts/evaluate_graph.py --units data/code_units.jsonl --queries data/evaluation_queries.jsonl --graph data/relationship_graph.json --output-dir data/experiments
```
- **Outputs**:
  - `data/experiments/graph_reranking_results.json`
  - `data/experiments/graph_reranking_results.csv`

---

## 4. Running the Web Application

The CodeSage application provides an interactive visual dashboard for real-time repository search, three-way side-by-side model comparison, and empirical evaluation metrics.

### 4.1 Start the FastAPI Backend
```bash
python scripts/serve_api.py --port 8000
```
- **API URL**: `http://127.0.0.1:8000`
- **Interactive OpenAPI Documentation**: `http://127.0.0.1:8000/docs`

### 4.2 Start the Next.js Frontend
In a separate terminal:
```bash
cd web
npm run dev
```
- **Web Interface**: `http://localhost:3000`
- **Features**:
  - `/`: Natural language search, model selector, top-$K$ controls, three-way side-by-side comparison, syntax-highlighted code viewers, graph neighborhood modal.
  - `/evaluation`: Interactive research benchmark tables, confidence intervals, per-category breakdowns, and graph reranking metrics.

---

## 5. Running the Complete Automated Test Suite

Run all 65 unit, integration, and API tests:

```bash
pytest -v
```

Expected output:
```text
================== 65 passed, 1 warning in ~90s ==================
```

Test coverage includes:
- Scanner boundary checks, secrets exclusion, and manifest creation (`tests/test_scanner.py`)
- Tree-sitter JavaScript and Dart extraction (`tests/test_extractors.py`)
- Lexical retrieval, tokenization, and character n-grams (`tests/test_retrieval.py`)
- General semantic embeddings and embedding cache invalidation (`tests/test_semantic.py`)
- Code transformer embeddings and determinism (`tests/test_code_semantic.py`)
- Metric calculation accuracy (MRR, Recall@K, nDCG@K) (`tests/test_evaluation.py`)
- Dataset audit, split isolation, and bootstrap math (`tests/test_study.py`)
- Syntax-aware graph construction and reranker affinity (`tests/test_graph.py`)
- FastAPI endpoints, CORS, input bounds, and 404/422 validation (`tests/test_api.py`)
