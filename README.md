# Bachelor Thesis Staffa: Automated Generation of Unique Marketing Texts Using LLMs

This GitHub repository contains all code used in my Bachelor's Thesis.
The project develops an automated pipeline for generating marketing texts from structured PIM (Product Information Management) data and semi-structured IBF (Insight-Benefit-Feature) descriptors using large language models.

All scripts were written by me, Stefano Staffa, as part of my Bachelor's thesis in Computational Linguistics.

---

## Repository Structure

| File | Description |
|------|-------------|
| `benchmark_pipeline.py` | Main entry point — runs the complete pipeline |
| `benchmark_core.py` | Core utilities: rate limiting, configuration, JSON parsing, quality scoring, rejection analysis |
| `benchmark_data.py` | Data loading and preprocessing for PIM data, IBF descriptors, and gold standards |
| `benchmark_llm_calls.py` | Async LLM API calls for generation, judging, hallucination detection, and prompt optimization |
| `benchmark_evaluation.py` | Evaluation framework: semantic similarity, feature coverage, attribute alignment |
| `prompts.py` | All prompt templates (generation, judge, hallucination, optimizer) |
| `editable_section.py` | Editable prompt sections for the generation prompt used during optimization iterations |
| `plots.ipynb` | A jupyter notebook to plot the graphs for visualisation |

---

## Data

| File | Description | Source |
|------|-------------|--------|
| `product_data_with_reference.xlsx` | Product data export with PIM attributes, these already have human written marketing text in gold_standards | Geberit AG (internal) |
| `product_data_without_reference.xlsx` | Product data export with PIM attributes, these have no marketing texts | Geberit AG (internal) |
| `ibf_data.xlsx` | IBF descriptors (Insight-Benefit-Feature) | Geberit AG (internal) |
| `gold_standards.xlsx` | Human-written reference texts for evaluation (optional) | Geberit AG (internal) |

> **Note:** `ibf_data.xlsx` and `gold_standards.xlsx` are not included in the repository as these are confidential company data.
The pipeline runs without these files, but results may be worse as the model won't have access to curated Insight-Benefit-Feature messaging or reference texts for evaluation.
---

## How to Run

### Environment Setup

Requires Python 3.13+
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API keys in a seperate .env file
OPENAI_API_KEY="your-key-here"
GEMINI_API_KEY="your-key-here"
```

### Running the Pipeline

The entire pipeline is executed with a single command:

```bash
python benchmark_pipeline.py
```

This script automatically:

1. **Loads and preprocesses data** — Extracts PIM attributes, merges IBF descriptors by product type/brand
2. **Generates marketing texts** — Uses GPT-4o-mini and/or Gemini 2.0 Flash with configurable prompts
3. **Evaluates outputs** using a hybrid framework:
   - **Semantic similarity** (SBERT embeddings)
   - **Rule-based validation** (character limits: 450–600 chars)
   - **LLM-as-judge scoring** (relevance, generalization, overall quality)
   - **Hallucination detection** (factual grounding check)
   - **Feature coverage & attribute alignment** (semantic matching)
4. **Computes composite scores** — Weighted combination: 50% judge + 25% attribute alignment + 10% feature coverage + 15% hallucination
5. **Optimizes prompts** — Iterative batch-based refinement targeting detected issues
6. **Analyzes rejections** — Categorizes failure patterns (too short, hallucination, off-topic, etc.)

---

## Configuration

Key parameters can be modified directly in `benchmark_pipeline.py`:

```python
CONFIG["USE_OPENAI"] = True
CONFIG["USE_GEMINI"] = True

CONFIG["OPENAI_MODEL"] = "gpt-4o-mini"
CONFIG["GEMINI_MODEL"] = "gemini-2.0-flash"

CONFIG["MIN_CHARS"] = 450
CONFIG["MAX_CHARS"] = 600

CONFIG["OPTIMIZE"] = True
CONFIG["OPT_ITERATIONS"] = 4
CONFIG["OPT_THRESHOLD"] = 8.5

CONFIG["RAW_EXPORT_PATH"] = "pipeline/export_test.xlsx"
CONFIG["IBF_PATH"] = "pipeline/ibf_data.xlsx"
CONFIG["GOLD_PATH"] = ""  # Optional

```

Additional settings in `benchmark_core.py`:

```python
CONFIG = {
    "OPENAI_RPM": 60,              # Rate limit (requests per minute)
    "GEMINI_RPM": 60,
    "MAX_CONCURRENT_OPENAI": 10,   # Concurrency control
    "MAX_CONCURRENT_GEMINI": 10,
    "RETRY_ATTEMPTS": 3,
    ...
}
```

---

## Output

Results are saved to timestamped Excel files:

| Output | Description |
|--------|-------------|
| `results_YYYY-MM-DD_HH-MM-SS.xlsx` | Full results with all metrics per product |
| `*.quality_summary.json` | Rejection analysis and top issues to fix |
| `prompt_logs/` | Prompt evolution across optimization iterations |

The Excel output contains two sheets:
- **results** — Individual product scores (judge, relevance, hallucination, composite, etc.)
- **aggregate** — Aggregated metrics by iteration, model, and prompt

---

## Visualization

The `plots.ipynb` notebook generates the figures used in the thesis:

- Score distributions by model (GPT-4o-mini vs. Gemini)
- Length compliance rates across optimization iterations
- Composite score breakdown by evaluation dimension
- Rejection pattern analysis

**Requires:** A results file generated by `benchmark_pipeline.py`

---

## Evaluation Metrics

| Metric | Description | Weight |
|--------|-------------|--------|
| `judge_score` | LLM-as-judge overall quality (1–10) | 50% |
| `relevance_score` | Product-specific relevance (1–5) | - |
| `generalization_score` | Appropriate generalization level (1–5) | - |
| `hallucination_score` | Factual accuracy (1–10, higher = better) | 15% |
| `semantic_attribute_alignment` | Embedding similarity to product attributes | 25% |
| `semantic_feature_coverage` | Coverage of input features in output | 10% |
| `in_range` | Character length within 450–600 | - |

---

## Key Results

| Model | Length Compliance | Stakeholder Acceptance | Zero Hallucinations |
|-------|-------------------|------------------------|---------------------|
| GPT-4o-mini | 76–82% | 80% | ✓ |
| Gemini 2.0 Flash | 12–48% | - | ✓ |

---

## Dependencies

Core libraries:
- `openai` — OpenAI API client
- `google-generativeai` — Gemini API client
- `sentence-transformers` — SBERT embeddings
- `pandas` — Data manipulation
- `tenacity` — Retry logic
- `python-dotenv` — Environment variables
