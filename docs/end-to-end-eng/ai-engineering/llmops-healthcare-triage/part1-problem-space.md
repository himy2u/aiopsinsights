# Redefining Healthcare Triage : An AI & Data Engineering Challenge

*Published: November 19, 2025*  
*Part 1 of 6 in the "Building a Production-Grade LLM Triage System" series*

While the business case for improving healthcare triage is clear, the core of the problem is a fascinating AI and data engineering challenge. This series will tackle it head-on.

## Quick answers

- **What is triage?** Sorting patient concerns to the right level of care and specialty with appropriate urgency.
- **What problem are we solving?** Converting messy, free‑text symptom descriptions into structured decisions: `{specialty, urgency, next best action}`—reliably, safely, and fast.
- **Why now?** Mature open models, affordable vector databases, and measurable ROI from reducing wait times and misrouting make automated triage both feasible and valuable.
- **How to solve (preferred)?** A hybrid approach: a small fine‑tuned classifier for `{specialty, urgency}` plus RAG for medical grounding and an agentic clarifier for ambiguous inputs—chosen for accuracy, latency, and cost control.
- **What does success look like?** Patients get the right care, quickly, with fewer ER misroutes. We’ll know we’ve succeeded when we hit our target KPIs (below) and sustain them in production.

### The Core Problem: From Unstructured Symptoms to Structured Decisions

The central task is to transform a patient's colloquial, often ambiguous, description of their symptoms into a structured, clinically relevant, and actionable decision. 

**Example:**
- **Patient Input**: `"I have a sharp pain in my chest when I breathe deep and my left arm feels numb."`
- **Desired AI Output**:
  ```json
  {
    "specialty": "Cardiology",
    "urgency": "High",
    "confidence_score": 0.92,
    "suggested_intake_questions": [
      "When did the symptoms start?",
      "Do you have a history of heart conditions?"
    ],
    "next_best_action": "call emergency services"
  }
  ```

This is a non-trivial task that requires a sophisticated, multi-layered solution. The system should also suggest clarifying follow-up questions and a safe, **next best action** (e.g., "call emergency services", "book primary care", "self-care guidance"). Our goal is to build this system, focusing on three key engineering pillars.

#### How symptoms turn into decisions
1. Parse the free-text complaint and extract key medical signals.
2. Classify `{specialty, urgency}` with a fine‑tuned model.
3. Use an agentic step to ask follow-up questions when confidence is low or information is missing.
4. Ground on clinical guidance via RAG to propose a safe "next best action" and output a structured decision.

#### Output schema
The triage service produces a compact, structured decision payload:
```json
{
  "specialty": "Cardiology",
  "urgency": "High",            
  "confidence_score": 0.0,
  "suggested_intake_questions": ["string"],
  "next_best_action": "emergency|book_primary_care|self_care"
}
```

### Big picture: Use cases
- **AI/ML**: Real-time triage decisions, model evaluation (agreement, safety), drift monitoring, and cost/tokens tracking per model version.
- **BI/Analytics**: Executive and operational dashboards on volume, latency SLOs, referral accuracy, ED diversion rate, patient experience, safety events.
- **Product Analytics**: Funnel from intake → questions → decision → outcome; clarifier effectiveness; channel performance; A/B tests of prompts/models.

--- 

### Pillar 1: The Data Engineering Challenge
**Goal**: Create a high-quality, reliable "Golden Dataset" to train and evaluate our AI.

There is no public, off-the-shelf dataset for this task. We must build it from the ground up.

1.  **Data Sourcing**: We'll ingest and combine public medical dialog datasets (e.g., from HuggingFace) with synthetically generated data for symptoms, and physician specialties to ensure comprehensive coverage.
2.  **Bronze-Silver-Gold Pipeline**: We will design a data pipeline to process this information.
    *   **Bronze Layer**: Raw, unstructured source data.
    *   **Silver Layer**: Data is cleaned, structured, and anonymized (PII redacted).
    *   **Gold Layer**: An expert-validated dataset that maps free-text symptoms to structured outputs. This becomes our ground truth for training and RAG.

#### Event schema (stream-first)
We model patient symptom submissions as immutable events.

```json
{
  "event_name": "symptom_reported",
  "event_version": 1,
  "event_time": "2025-11-19T12:34:56Z",
  "patient_id": "uuid",
  "payload": {
    "symptom_text": "string",
    "channel": "web|mobile|ivr",
    "locale": "en-US"
  },
  "trace_id": "uuid"
}
```

#### Schema evolution strategy
- **Backward-compatible changes** only (additive fields, defaults).
- Enforce via **registry** (e.g., Confluent Schema Registry) and CI checks.
- Version events with `event_version`; deprecate, never break.

#### Streaming backbone (Kafka/Pulsar)
- Use a topic like `triage.symptom_reported` for ingestion and `triage.decision_made` for outputs.
- Bronze persists the raw event; Silver materializes curated tables; Gold stores expert labels.

#### Data contracts
- Producer promises: required fields, PII handling, SLA on delivery.
- Consumer expectations: latency SLOs, retry semantics, idempotency keys.
- Contract as code: JSON schema + tests enforced in CI before deploys.

#### Data product & metadata
- Data product: `triage_decisions` (owner: AI Engineering).
- Interfaces: stream (`triage.decision_made`), table (`gold.triage_decisions`).
- Metadata: lineage, freshness, quality scores, owners, and SLOs exposed in catalog.

### Pillar 2: The AI/ML Challenge
**Goal**: Build a highly accurate, context-aware, and safe triage and routing engine.

Simply prompting a generic LLM is not enough; it's expensive, slow, and lacks the necessary real-time context and safety guardrails.

1.  **Retrieval-Augmented Generation (RAG)**: We'll build a RAG system to provide the LLM with two critical pieces of real-time information:
    *   A **Medical Knowledge Base** for grounding in facts.
    *   A **Physician Directory** with specialty.
2.  **Fine-Tuned Classification Model**: We will fine-tune an efficient, open-source LLM (like Llama3-8B or Mistral-7B) on our "Golden Dataset." This specialized model will perform the core classification task (`symptom text → {specialty, urgency}`) far more reliably and cost-effectively than a general model.
3.  **Agentic Workflow**: For ambiguous inputs, the model will act as an agent, generating clarifying questions to gather more information before making a final recommendation.

### Pillar 3: The Analytics & Measurement Challenge
**Goal**: Prove the system is effective, safe, and meets business objectives.

1.  **Accuracy Benchmarking**: We will continuously evaluate the model's accuracy against a human expert baseline, targeting **>90% agreement**.
2.  **Latency SLOs**: We will define and monitor a Service Level Objective for API response time, aiming for a **p99 latency of <2 seconds**.
3.  **Confidence Scoring & Monitoring**: Every prediction will have a confidence score. Low-confidence outputs will be automatically flagged for human review. We will also monitor for data and model drift to ensure performance doesn't degrade over time.

#### Business KPIs
In addition to technical metrics, we will track outcomes that demonstrate business value:
- **Average Time-to-Triage**: Reduce from minutes to seconds for first, safe guidance
- **Referral Accuracy**: Correct specialty/level-of-care routing rate (audit-based)
- **ED Diversion Rate**: Appropriate routing of non-emergent cases away from ER
- **Patient Safety Events**: Near-miss/adverse-event rate (target: zero)
- **Patient Experience (CSAT/NPS)**: Post-triage satisfaction score

## What's Next

In Part 2, we will dive into the next stage of our lifecycle: **Data Profiling** and **Bronze Ingestion** fundamentals.

- Continue to: [Part 2 — Data Profiling](part2-data-profiling.md)
- Or jump ahead to: [Part 3 — Bronze Ingestion](part3-bronze-ingestion.md)

---
*This series is based on real-world implementations but uses synthetic data and anonymized case studies. Always consult healthcare professionals for medical advice.*

*Himanshu Pandey is a Data Leader with expertise in AI/ML, data engineering, and cloud infrastructure. Connect with me on [Twitter](https://x.com/himanshuptech) or [LinkedIn](https://www.linkedin.com/in/hrnp).*
