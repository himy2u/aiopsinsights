# Bronze Ingestion: Building the Golden Dataset Foundation

*Published: November 19, 2025*  
*Part 2 of 6 in the "Building a Production-Grade LLM Triage System" series*

## Objectives
- Ingest synthetic + public medical dialog into a lakehouse Bronze layer.
- Redact PII and normalize records into a Silver schema ready for labeling.
- Produce initial Golden Dataset seed: `symptom_text → {specialty, urgency}`.
- Make all steps testable, cost-aware, and reproducible.

## Scope (This Part)
- Data generation (synthetic), format contracts, PII redaction, validations, CI checks.
- Outputs: `bronze/`, `silver/`, tests, pre-commit, Dockerfile, orchestration via GitHub Actions.

## FinOps BOM (Low Volume: 10k rows/day)
| Tool | Option | $/mo (AWS) | $/mo (GCP) | $/mo (Azure) | Free Tier? | Notes |
|------|--------|------------|------------|--------------|------------|-------|
| Object Storage | S3 / GCS / Azure Blob | $1–$3 | $1–$3 | $1–$3 | Yes | Data at rest + egress negligible at 10k/day |
| Compute (batch) | GitHub Actions + small runner | $0–$10 | $0–$10 | $0–$10 | Yes | CI minutes mostly free at low volume |
| Orchestration | GitHub Actions | $0 | $0 | $0 | Yes | Re-usable across stages |
| PII Redaction | Presidio (self-host) | $0 | $0 | $0 | Yes | CPU-only acceptable for regex/NLP-lite |
| Catalog/Schema | Great Expectations (OSS) | $0 | $0 | $0 | Yes | Checks run in CI |

Gate: Total Stage 2 (Bronze) <= $25/mo → Approved.

## Decision Matrix (Ingestion/Transport)
| Option | Cost | Latency | Operability | Lock-in Risk | Score |
|--------|------|---------|-------------|--------------|-------|
| Files via batch (daily) | 10 | 6 | 9 | 10 | 35 |
| Kafka managed (EH/Kinesis/PubSub) | 4 | 9 | 7 | 6 | 26 |
| Self-host Kafka | 3 | 8 | 3 | 9 | 23 |
Winner: Batch files for initial scale (10k/day), revisit stream later.

## Data Contracts
- Input (Bronze raw):
```json
{
  "id": "uuid",
  "timestamp": "iso8601",
  "channel": "web|sms|phone",
  "patient_text": "string",
  "zip": "string",
  "age": 42,
  "gender": "F|M|X|U"
}
```
- Output (Silver normalized):
```json
{
  "id": "uuid",
  "ts": "iso8601",
  "source": "web|sms|phone",
  "symptom_text": "string",
  "geo": { "zip": "string" },
  "demographics": { "age": 42, "gender": "U" },
  "pii_redacted": true
}
```

## Architecture
```mermaid
flowchart LR
  gen[Generate Synthetic Data] --> bronze[(Object Storage: bronze/)]
  public[Public Datasets] --> bronze
  bronze --> redact[PII Redaction (Presidio/regex)]
  redact --> validate[Great Expectations Checks]
  validate --> silver[(Object Storage: silver/)]
  CI[GitHub Actions CI] -->|pre-commit, tests| validate
```

## File Tree (New/Updated)
```
docs/ai-engineering/llmops-healthcare-triage/
  ├─ part2-bronze-ingestion.md
src/
  └─ bronze_ingest/
     ├─ bronze_ingest.py
     ├─ requirements.txt  # only for runtime image, CI uses uv
     ├─ Dockerfile
     ├─ pyproject.toml
     ├─ .pre-commit-config.yaml
     └─ tests/
        ├─ test_schema.py
        └─ test_pii.py
.github/workflows/
  └─ bronze-ci.yml
```

## Deliverables (This Part)
- `bronze_ingest.py`: read CSV/JSON → write partitioned `bronze/` and normalized `silver/`.
- `Great Expectations` suite: schema + nulls + uniqueness.
- `PII redaction`: Email/phone/SSN by regex baseline; pluggable Presidio.
- `Dockerfile`: multi-stage build, non-root user, uv cache.
- `pre-commit`: black, isort, ruff; CI gate.
- `bronze-ci.yml`: run formatting, unit tests, GE checks on PRs.

## Success Criteria
- Unit tests pass locally and in CI.
- GE validation passes on a 10k-row sample.
- End-to-end runtime < 2 minutes in CI.
- Storage layout documented and reproducible.

## Next
- Part 3 (Silver + RAG Store): choose vector DB (Qdrant vs PGVector) with decision matrix, build loaders.
