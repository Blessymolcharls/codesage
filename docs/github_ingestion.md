# CodeSage — Secure GitHub Repository Ingestion

**Document Version:** 1.0.0  
**Status:** Implemented & Verified (Milestone 3.1)  
**Date:** 2026-10-20  
**Scope:** Public GitHub Repository Ingestion, Isolated Workspace Publication Protocol, and Subprocess Hardening

---

## 1. Executive Summary

CodeSage Milestone 3.1 extends the multi-repository indexing platform to ingest, index, benchmark, and refresh public GitHub repositories as isolated workspaces. The implementation enforces strict security boundaries, sandboxes subprocess interactions, applies early resource constraints, guarantees source/index atomicity via a recoverable publication protocol, and prevents disclosure of host paths or credentials.

All existing platform capabilities—including the DocuCharm canonical baseline research artifacts, incremental indexing, Hybrid Reciprocal Rank Fusion (RRF), and graph context reranking—are preserved without regression.

---

## 2. Security Architecture & Threat Model

### 2.1 URL Parsing and Validation Engine (`src/codesage/github.py`)

Untrusted user input is evaluated against a strict canonical policy before any network or filesystem operation:

- **Protocol**: Exclusively `https://`. All other protocols (`http://`, `git://`, `ssh://`, `file://`, `ftp://`, SCP-style `git@...`) are rejected.
- **Remote Host**: Strictly `github.com`. Deceptive hostnames (e.g., `github.com.attacker.com`, `evil-github.com`), alternative git hosts (`gitlab.com`, `bitbucket.org`), subdomains (`gist.github.com`, `raw.githubusercontent.com`, `api.github.com`), and IP addresses are rejected.
- **Authentication / Credentials**: Explicitly rejects URLs containing userinfo or passwords (`https://user:pass@github.com/...`). CodeSage does not accept or store private credentials.
- **Port Policy**: Non-standard ports (e.g. `:8080`, `:443`) are rejected to eliminate SSRF against host services.
- **Query Strings & Fragments**: Strict rejection of `?branch=...` and `#fragment` prevents unintended parameter manipulation.
- **Path Traversal & Slug Validation**: Path must consist of exactly two segments: `/{owner}/{repo}`. Traversal tokens (`..`, `.`) and non-alphanumeric boundaries are rejected. Trailing slashes and `.git` suffixes are cleanly normalized.
- **Deterministic Workspace ID**: Collision-resistant identifier derived as `gh-{owner}-{repo}` (truncated to 64 chars). Any collision with local or reserved workspaces triggers numeric suffixing (`gh-{owner}-{repo}-2`).

### 2.2 Git Subprocess Hardening

When fetching remote repositories, CodeSage interacts with Git through hardened boundaries:

1. **No Shell Execution**: Git is invoked using an explicit argument array (`["git", ...]`) with `shell=False`.
2. **Interactive Prompting Disabled**: Environment variable `GIT_TERMINAL_PROMPT=0` prevents the subprocess from blocking on interactive credential prompts.
3. **Controlled Configuration Environment**:
   - `GIT_CONFIG_NOSYSTEM=1`: Disables system-wide gitconfig injection.
   - `-c core.hooksPath=""`: Overrides and disables client-side repository hooks.
   - `-c protocol.allow=never` & `-c protocol.https.allow=always`: Restricts Git's transport layer strictly to HTTPS, blocking protocol smuggling (e.g., `ext::`, `file::`).
4. **Dedicated Temporary Destination**: Clone targets a unique temporary directory inside the workspace directory (`data/workspaces/<id>/.candidate_source_tmp`), never a shared `/tmp` or root path.
5. **Bounded Timeouts**: All Git operations execute under an enforceable timeout (default: 60s). Subprocess timeouts trigger immediate SIGKILL and recursive purge of candidate files.
6. **No Code Execution**: CodeSage strictly treats repositories as passive text/code. No build tools (`npm`, `pip`, `make`, `setup.py`) or repository scripts are ever executed.
7. **Immediate `.git` Purge**: Once the pinned commit SHA is resolved via `git rev-parse HEAD`, the entire `.git/` directory is immediately deleted, eliminating repository hooks, submodules, and stored Git objects.
8. **Path Sanitization**: Stderr and stdout streams are scrubbed of host absolute paths before logging or returning error responses.

> [!WARNING]
> Disabling Git hooks alone does not make Git safe for arbitrary untrusted repositories. Security requires combining shell-less invocations, HTTPS-only protocol restrictions, prompt suppression, destination boundary validation, immediate `.git` purge, and post-clone filesystem audits.

### 2.3 Filesystem & Resource Limits

To defend against Denial-of-Service (DoS) and malicious repositories (e.g., billion laughs zip bombs or symlink escape attacks):

| Control | Limit | Action on Violation |
| :--- | :--- | :--- |
| **Max Files** | 2,500 files (configurable) | Aborts immediately; candidate purged; returns HTTP 413 |
| **Max Total Size** | 50 MB | Aborts immediately; candidate purged; returns HTTP 413 |
| **Max File Size** | 10 MB per file | Aborts immediately; candidate purged; returns HTTP 413 |
| **Symlink Containment**| Must resolve within candidate root | Aborts immediately; candidate purged; returns HTTP 422 |
| **Symlink Following** | `os.walk(..., followlinks=False)` | Symlinks are never traversed during scanning |
| **Deletion Guard** | Must resolve inside `data/workspaces/` | Refuses deletion if resolved path escapes workspace boundary |

---

## 3. Recoverable Publication Protocol

To prevent workspace corruption, CodeSage implements a two-phase staging and publication protocol for GitHub workspaces. The active source checkout and its corresponding index are **never replaced** before the candidate index is fully generated and validated.

```mermaid
sequenceDiagram
    participant C as Client / API
    participant M as WorkspaceManager
    participant T as .candidate_source_tmp
    participant R as .candidate_source_ready
    participant S as .staging_tmp
    participant L as Live source/ & Index

    C->>M: POST /api/workspaces/github
    Note over M: 1. Register metadata & acquire workspace lock
    M->>T: Safe Git shallow clone (--depth 1)
    M->>T: Resolve HEAD SHA & purge .git/
    M->>T: Audit symlinks, file count & size limits
    M->>S: Run Indexer & GraphBuilder on candidate source
    Note over M: 2. Validate staged units, graph & manifest
    M->>R: os.replace(T, R) (Candidate ready)
    opt Live source exists
        M->>M: os.replace(Live source, .previous_source_backup)
    end
    M->>L: os.replace(R, Live source)
    M->>L: _publish_staged_artifacts(S, Live index)
    Note over M: 3. Delete backup & persist updated metadata
    M-->>C: 201 Created (Ready)
```

### 3.1 Failure Recovery Behavior

If any step fails (network drop, timeout, limit breach, indexing exception, or crash):
- **Candidate Purge**: All `.candidate_source_tmp`, `.candidate_source_ready`, and `.staging_tmp` directories are cleaned up.
- **Rollback**: If `.previous_source_backup` exists, it is restored back to `source/`.
- **State Preservation**: The workspace retains its prior working source, valid index, and active `commit_sha`.
- **Automatic Recovery on Restart**: Whenever a workspace is loaded, `_recover_workspace_staging` cleans any stale candidate directories left by process termination or power failure.

### 3.2 Refresh Semantics

When `POST /api/workspaces/{id}/refresh` is triggered:
1. CodeSage clones the remote repository to `.candidate_source_tmp` and extracts the remote commit SHA.
2. If the remote commit SHA matches the active `commit_sha` and the index is healthy, the candidate is discarded and the API returns `status: "up_to_date"`, avoiding redundant reindexing.
3. If a new commit SHA is found, the candidate is indexed into `.staging_tmp`, verified, and published atomically.
4. If the refresh fails, the prior version remains completely untouched and active.

---

## 4. API Endpoints

### 4.1 Register & Ingest GitHub Repository

- **Endpoint**: `POST /api/workspaces/github`
- **Status Code**: `201 Created`
- **Request Body**:
```json
{
  "url": "https://github.com/psf/requests",
  "name": "Requests HTTP Library",
  "auto_index": true,
  "max_files": 2500,
  "max_file_size_kb": 512
}
```
- **Response**:
```json
{
  "id": "gh-psf-requests",
  "name": "Requests HTTP Library",
  "display_path": "https://github.com/psf/requests",
  "source_type": "github",
  "github_url": "https://github.com/psf/requests",
  "github_owner": "psf",
  "github_repo": "requests",
  "commit_sha": "b5a93259e863fa9ddcd06e8b4e727e2a9b6db5ad",
  "unit_count": 312,
  "file_count": 28,
  "status": "ready"
}
```

### 4.2 Refresh Remote Workspace

- **Endpoint**: `POST /api/workspaces/{workspace_id}/refresh`
- **Status Code**: `200 OK`
- **Response (Unchanged)**:
```json
{
  "status": "up_to_date",
  "commit_sha": "b5a93259e863fa9ddcd06e8b4e727e2a9b6db5ad",
  "message": "Repository is already at latest commit."
}
```
- **Response (Updated)**:
```json
{
  "status": "updated",
  "previous_commit": "b5a93259e863fa9ddcd06e8b4e727e2a9b6db5ad",
  "new_commit": "f901a1c3d2890e1bc001928374a5893d9821ef34",
  "unit_count": 325
}
```

---

## 5. Verification & Test Coverage

The implementation includes 65 dedicated automated tests across 4 suites, plus 81 existing platform tests:

1. **`tests/test_github_url.py`** (39 tests):
   - Canonical normalization, `.git` stripping, trailing slash handling.
   - Scheme validation, deceptive host rejection, userinfo/credential rejection, port rejection.
   - Query string & fragment rejection, path traversal rejection, reserved identifier protection, collision derivation.
2. **`tests/test_github_fetch.py`** (8 tests):
   - Subprocess execution with `shell=False`, argument list, and restricted Git environment.
   - Timeout handling and cleanup.
   - Error message path sanitization.
   - Immediate `.git/` purge after SHA extraction.
   - File count, total size, and individual file size limits.
   - Symlink escape detection and rejection.
3. **`tests/test_github_workspace.py`** (9 tests):
   - GitHub workspace registration, duplicate URL rejection (409), ID collision suffixing.
   - End-to-end fetch, index, search, and graph generation.
   - Fetch failure leaves candidate cleanly purged.
   - Refresh skips reindexing when commit is unchanged.
   - Refresh updates atomically on new commit.
   - Failed refresh preserves previous source, index, and commit SHA.
   - Interrupted publication recovery on restart.
   - Remote-to-local and remote-to-remote workspace isolation.
4. **`tests/test_github_api.py`** (9 tests):
   - `POST /api/workspaces/github` validation errors (400), conflicts (409), fetch failures (502), timeouts (504), oversized payloads (413), and unsafe content (422).
   - End-to-end API registration, details endpoint, and cross-model search queries on remote workspaces.
   - API refresh lifecycle handling.

**Full Test Suite Result:** 146 passed in 55s.  
**Frontend Compilation (`npm run build`):** Clean build (5/5 static pages generated).

---

## 6. Canonical Artifact Immutability Verification

All 8 canonical research benchmark artifacts have been re-verified by cryptographic SHA-256 fingerprinting:

| Artifact Path | SHA-256 Fingerprint | Status |
| :--- | :--- | :--- |
| `data/evaluation_queries.jsonl` | `fa86fc9df2924509dfd8413bea65626212df20d0d0b9200d3695f27ea6fab397` | Unmodified |
| `data/code_units.jsonl` | `1e388d99649574213cdf8ddc11423407cba61b7ab14cef4a04c9891b34e7e869` | Unmodified |
| `data/manifest.json` | `b7f5a829388b6be12b3f3bc2e3592fa3f0f67daddf940351d4bb2d236f6f08f1` | Unmodified |
| `data/relationship_graph.json` | `96329d9e144ade4b11904c5f55a07c6b827dd2cc5bc2f5d788609512648454a7` | Unmodified |
| `data/benchmark_comparison.json` | `39c63721e9035d250adf7547e9335de10a071dd16a610d81fdae1ae528ad1b5d` | Unmodified |
| `data/evaluation_results.json` | `480e1ecd8fb3bebcf924f53005a15d2eabe03924989fd56cfb3f86b2697bcb0c` | Unmodified |
| `data/experiments/study_results.json` | `6b651007c31eb54e4a884d8a29e1c63a9eac5eb2d8b359aa0bb93b2d89bf5178` | Unmodified |
| `data/experiments/hybrid_fusion_results.json` | `e28208422b10c79f2712bdaf3f5355fa651da8216241901064ae22d6e460fb03` | Unmodified |
| `data/experiments/graph_reranking_results.json` | `cbd2b340c4a7afef55f788d599c68a169b693463f5f3fc3b6b9605ec090fd9d8` | Unmodified |

---

## 7. Known Limitations & Future Work

1. **Public Repositories Only**: CodeSage Milestone 3.1 intentionally restricts ingestion to public GitHub repositories. Ingestion of private repositories requiring SSH keys, GitHub Apps, or personal access tokens is deferred to Phase 2.
2. **Git Submodules**: Submodule recursion (`--recurse-submodules`) is explicitly disabled to avoid unvetted remote fetches and credential-leak vectors.
3. **Repository Size Ceiling**: Repositories containing more than 2,500 source files or exceeding 50 MB total footprint are rejected. Expanding these limits will require background worker queues (e.g. Celery / RQ) to avoid HTTP request timeouts.
