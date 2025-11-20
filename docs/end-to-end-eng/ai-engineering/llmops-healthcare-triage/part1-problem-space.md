# Redefining Healthcare Triage : An AI & Data Engineering Challenge

*Published: November 19, 2025*  
*Part 1 of 6 in the "Building a Production-Grade LLM Triage System" series*

While the business case for improving healthcare triage is clear, the core of the problem is a fascinating AI and data engineering challenge. This series will tackle it head-on.

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
    ]
  }
  ```

This is a non-trivial task that requires a sophisticated, multi-layered solution. Our goal is to build this system, focusing on three key engineering pillars.

--- 

### Pillar 1: The Data Engineering Challenge
**Goal**: Create a high-quality, reliable "Golden Dataset" to train and evaluate our AI.

There is no public, off-the-shelf dataset for this task. We must build it from the ground up.

1.  **Data Sourcing**: We'll ingest and combine public medical dialog datasets (e.g., from HuggingFace) with synthetically generated data for symptoms, locations, and physician specialties to ensure comprehensive coverage.
2.  **Bronze-Silver-Gold Pipeline**: We will design a data pipeline to process this information.
    *   **Bronze Layer**: Raw, unstructured source data.
    *   **Silver Layer**: Data is cleaned, structured, and anonymized (PII redacted).
    *   **Gold Layer**: An expert-validated dataset that maps free-text symptoms to structured outputs. This becomes our ground truth for training and RAG.

### Pillar 2: The AI/ML Challenge
**Goal**: Build a highly accurate, context-aware, and safe triage and routing engine.

Simply prompting a generic LLM is not enough; it's expensive, slow, and lacks the necessary real-time context and safety guardrails.

1.  **Retrieval-Augmented Generation (RAG)**: We'll build a RAG system to provide the LLM with two critical pieces of real-time information:
    *   A **Medical Knowledge Base** for grounding in facts.
    *   A **Physician Directory** with specialty, location, and availability.
2.  **Fine-Tuned Classification Model**: We will fine-tune an efficient, open-source LLM (like Llama3-8B or Mistral-7B) on our "Golden Dataset." This specialized model will perform the core classification task (`symptom text → {specialty, urgency}`) far more reliably and cost-effectively than a general model.
3.  **Agentic Workflow**: For ambiguous inputs, the model will act as an agent, generating clarifying questions to gather more information before making a final recommendation.

### Pillar 3: The Analytics & Measurement Challenge
**Goal**: Prove the system is effective, safe, and meets business objectives.

1.  **Accuracy Benchmarking**: We will continuously evaluate the model's accuracy against a human expert baseline, targeting **>90% agreement**.
2.  **Latency SLOs**: We will define and monitor a Service Level Objective for API response time, aiming for a **p99 latency of <2 seconds**.
3.  **Confidence Scoring & Monitoring**: Every prediction will have a confidence score. Low-confidence outputs will be automatically flagged for human review. We will also monitor for data and model drift to ensure performance doesn't degrade over time.

#### Business KPIs
In addition to technical metrics, we will track outcomes that demonstrate business value:
- **Average Time-to-Triage**: Target reduction (e.g., from 12 mins to < 2 mins)
- **ED Diversion Rate**: Percent of non-emergent cases routed away from ER appropriately
- **Referral Accuracy**: Correct specialty routing rate measured via retrospective audits
- **Care Access Uplift**: Increase in patients successfully routed to in-network providers
- **Cost per Triage**: All-in unit cost reduction vs. human-only workflow
- **Safety Events**: Near-miss and adverse-event rate; zero-tolerance thresholds
- **Patient Satisfaction (CSAT/NPS)**: Experience scores post-triage
- **Clinician Load Reduction**: Reduction in manual reviews per 1,000 triages

## What's Next

In Part 2, we will dive into the next stage of our lifecycle: **Data Profiling** and **Bronze Ingestion** fundamentals.

- Continue to: [Part 2 — Data Profiling](part2-data-profiling.md)
- Or jump ahead to: [Part 3 — Bronze Ingestion](part3-bronze-ingestion.md)

---
*This series is based on real-world implementations but uses synthetic data and anonymized case studies. Always consult healthcare professionals for medical advice.*

*Himanshu Pandey is a Data Leader with expertise in AI/ML, data engineering, and cloud infrastructure. Connect with me on [Twitter](https://x.com/himanshuptech) or [LinkedIn](https://www.linkedin.com/in/hrnp).*
