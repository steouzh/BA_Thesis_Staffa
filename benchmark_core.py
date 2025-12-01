# benchmark_core.py
from dotenv import load_dotenv
load_dotenv()

import os
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_CPP_PLUGIN_LOGGING"] = "0"

import re
import json
import asyncio
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import List, Dict, Optional, Any
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
import google.generativeai as genai
from sentence_transformers import util
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from datetime import datetime

from benchmark_evaluation import (
    build_attribute_text,
    compute_attribute_alignment,
    compute_feature_coverage_batch,
    check_uniqueness as optimized_check
)


# -------------------------------------------------------------------
# GLOBALS: thread pool, rate limiters, fake datetime, CONFIG, dataclass
# -------------------------------------------------------------------

threadpool = ThreadPoolExecutor(max_workers=4)

class RateLimiter:
    """Rate limiter with FIXED timing."""
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                await asyncio.sleep(sleep_time)
            self.last_request_time = time.time()

openai_rate_limiter = RateLimiter(requests_per_minute=60)
gemini_rate_limiter = RateLimiter(requests_per_minute=60)

from datetime import datetime as _datetime, timedelta

# ===================== CONFIG =====================

CONFIG: Dict[str, Any] = {
    # Input/Output
    "RAW_EXPORT_PATH": "export_test.xlsx",
    "IBF_PATH": "ibf_data.xlsx",
    "STANDARDIZED_INPUT_PATH": None,
    "GOLD_PATH": "gold_standards/gold_candidates.xlsx",
    "OUTPUT_PATH": "results.xlsx",
    "PROMPT_LOG_DIR": "prompt_logs",

    # Providers
    "USE_OPENAI": True,
    "USE_GEMINI": True,
    "SBERT_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    
    # Judges
    "PRIMARY_JUDGE": "openai",
    "SECONDARY_JUDGE": False,
    "SECONDARY_JUDGE_PROVIDER": "gemini",

    # Models
    "SYSTEM_PROMPT": "You are a marketing specialist. Write precise, factual marketing copy.",
    "OPENAI_MODEL": "gpt-4o-mini",
    "GEMINI_MODEL": "gemini-2.0-flash",
    "GEN_CONCURRENCY": 6, 

    # Text constraints
    "MIN_CHARS": 450,
    "MAX_CHARS": 600,

    # Optimization 
    "OPTIMIZE": True,
    "OPT_ITERATIONS": 3,
    "OPT_THRESHOLD": 8.5,  
    
    # Rate limiting
    "OPENAI_RPM": 60,
    "GEMINI_RPM": 60,
    "MAX_CONCURRENT_OPENAI": 10,
    "MAX_CONCURRENT_GEMINI": 10,
    "RETRY_ATTEMPTS": 3,
    "RETRY_MIN_WAIT": 1,
    "RETRY_MAX_WAIT": 10,
    
    # Testing
    "DRY_RUN": False,  # Set True for fast free testing
}


# ===================== DATA STRUCTURES =====================

@dataclass
class PromptLogEntry:
    timestamp: str
    iteration: int
    prompt_name: str
    original_prompt: str
    improved_prompt: str
    rationale: str
    runtime_seconds: float
    trigger_data: Dict[str, Any]

# ===================== HELPER FUNCTIONS =====================


def safe_json_parse(text):
    """
    Attempts to parse model output as JSON, even if malformed.
    Fixes:
    - trailing text after JSON
    - leading explanations before JSON
    - single quotes → double quotes
    - missing closing brackets
    - incomplete JSON objects
    """

    if not text or not isinstance(text, str):
        return None

    # 1. Extract the first {...} block from the text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        candidate = match.group(0)
    else:
        candidate = text.strip()

    # 2. Replace single quotes with double quotes if needed
    if "'" in candidate and '"' not in candidate:
        candidate = candidate.replace("'", '"')

    # 3. Remove trailing commas before }
    candidate = re.sub(r',\s*}', '}', candidate)

    # 4. Attempt JSON parse
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # 5. Try to close JSON if missing ending }
    if candidate.count("{") > candidate.count("}"):
        candidate = candidate + "}"

    try:
        return json.loads(candidate)
    except Exception:
        return None

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def create_retry_decorator(max_attempts: int = 3, min_wait: int = 1, max_wait: int = 10):
    """Create retry decorator."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )



# ===================== API WRAPPERS =====================

async def call_openai_with_retry(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict],
    temperature: float = 0.7,
    max_tokens: int = 1000,
    semaphore: Optional[asyncio.Semaphore] = None,
    max_attempts: int = 3,
    context: str = "unknown"  # NEW: Better error messages
) -> Optional[str]:
    """Call OpenAI API with retry and context."""
    
    @create_retry_decorator(max_attempts=max_attempts)
    async def _call():
        await openai_rate_limiter.acquire()
        
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=60.0
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"⚠️ OpenAI error in {context}: {type(e).__name__}")
            raise
    
    try:
        if semaphore:
            async with semaphore:
                return await _call()
        else:
            return await _call()
    except RetryError:
        print(f"❌ OpenAI failed after {max_attempts} attempts in {context}")
        return None
    except Exception as e:
        print(f"❌ OpenAI error in {context}: {e}")
        return None

async def call_gemini_with_retry(
    model_name: str,
    prompt: str,
    temperature: float = 0.7,
    max_output_tokens: int = 200,
    semaphore: Optional[asyncio.Semaphore] = None,
    max_attempts: int = 3,
    context: str = "unknown"
) -> Optional[str]:
    """Call Gemini API with retry."""
    
    async def _call():
        await gemini_rate_limiter.acquire()
        
        loop = asyncio.get_event_loop()
        model = genai.GenerativeModel(model_name)
        
        def _sync_call():
            resp = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens
                )
            )
            return resp.text
        
        return await loop.run_in_executor(threadpool, _sync_call)
    
    for attempt in range(max_attempts):
        try:
            if semaphore:
                async with semaphore:
                    return await _call()
            else:
                return await _call()
        except Exception as e:
            if attempt < max_attempts - 1:
                wait_time = min(2 ** attempt, 10)
                print(f"⚠️ Gemini error in {context} (attempt {attempt + 1}/{max_attempts}), retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ Gemini failed after {max_attempts} attempts in {context}")
                return None

# ===================== GOLD + SBERT HELPERS =====================

def compute_sbert_similarity(cand_text: str, gold_df: pd.DataFrame, sbert_model) -> float:
    """Compute SBERT similarity to gold."""
    if gold_df is None or gold_df.empty:
        return 0.0
    
    try:
        cand_emb = sbert_model.encode(cand_text, convert_to_tensor=True, show_progress_bar=False)
        gold_embeddings = [e for e in gold_df["gold_emb"] if e is not None]
        
        if not gold_embeddings:
            return 0.0
        
        sims = [float(util.cos_sim(cand_emb, g_emb)) for g_emb in gold_embeddings]
        return max(sims) if sims else 0.0
    except Exception as e:
        print(f"⚠️ SBERT error: {e}")
        return 0.0


def check_uniqueness(df: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    """Check for duplicates - uses optimized version from evaluation_optimized."""
    return optimized_check(df, threshold)


# ===================== METRICS =====================

def normalize_feature_coverage(feat_cov: float, num_features: int) -> float:
    """Normalize feature coverage."""
    if num_features == 0:
        return 0.0
    
    features_mentioned = feat_cov * num_features
    
    if features_mentioned < 2:
        return 0.75 * (features_mentioned / 2)
    elif features_mentioned < 3:
        return 0.75 + (features_mentioned - 2) * 0.15
    elif features_mentioned < 4:
        return 0.90 + (features_mentioned - 3) * 0.05
    else:
        return min(1.0, 0.95 + (features_mentioned - 4) * 0.0125)

def compute_metrics(row, min_chars, max_chars):
    """Compute metrics for single row."""
    text = str(row.get("marketing_text", ""))
    
    # Build grounding source
    attribute_text = build_attribute_text(row)
    
    # Parse features
    features = row.get("features", [])
    if isinstance(features, str):
        try:
            import ast
            features = ast.literal_eval(features)
        except:
            features = [features]
    
    # Compute coverage
    feat_cov_raw = compute_feature_coverage_batch(text, features, threshold=0.4)
    num_features = len(features) if isinstance(features, list) else 0
    feat_cov_normalized = normalize_feature_coverage(feat_cov_raw, num_features)
    
    return {
        "char_len": len(text),
        "in_range": min_chars <= len(text) <= max_chars,
        "semantic_attribute_alignment": compute_attribute_alignment(text, attribute_text),
        "semantic_feature_coverage": feat_cov_raw,
        "semantic_feature_coverage_normalized": feat_cov_normalized
    }

def compute_overall_score(row):
    """Compute composite quality score with robust None handling."""
    
    # Helper function to safely get numeric values
    def safe_get(key, default):
        val = row.get(key, default)
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default
    
    # Get scores with safe defaults
    hallucination_score = safe_get('hallucination_score', 10.0)
    semantic_attribute_alignment = safe_get('semantic_attribute_alignment', 0.0)
    judge_score = safe_get('judge_score', 0.0)
    feat_cov_norm = safe_get('semantic_feature_coverage_normalized', 0.0)
    
    # Critical failures
    if hallucination_score < 7.0:
        return 3.0
    if semantic_attribute_alignment < 0.25:
        return 4.0
    
    # Normalize scores
    judge_norm = judge_score / 10.0
    attr_align = semantic_attribute_alignment
    hall_norm = hallucination_score / 10.0
    
    # Weighted average
    score = (
        0.5 * judge_norm +
        0.25 * attr_align +
        0.1 * feat_cov_norm +
        0.15 * hall_norm
    )
    
    return score * 10.0

def passes_quality_gates(row):
    """
    Multi-dimensional quality thresholds prevent accepting outputs 
    that score well overall but fail critically in one dimension.
    
    Based on paper's Table 8 approach - separate thresholds per criterion.
    
    Returns:
        (bool, str): (passed, reason)
    """
    
    # Helper to safely get values
    def safe_get(key, default):
        val = row.get(key, default)
        if val is None:
            return default
        try:
            return float(val) if not isinstance(default, bool) else val
        except (TypeError, ValueError):
            return default
    
    # Define thresholds for each dimension
    gates = {
        "judge_score": (safe_get("judge_score", 0), 7.0, "Judge score too low"),
        "relevance": (safe_get("relevance_score", 0), 3.5, "Relevance too low"),
        "generalization": (safe_get("generalization_score", 0), 3.0, "Generalization too low"),
        "hallucination": (safe_get("hallucination_score", 0), 8.0, "Hallucination detected"),
        "attribute_alignment": (safe_get("semantic_attribute_alignment", 0), 0.35, "Poor attribute alignment"),
        "in_range": (safe_get("in_range", False), True, "Length out of range"),
    }
    
    # Check each gate
    for criterion, (value, threshold, message) in gates.items():
        if isinstance(threshold, bool):
            if value != threshold:
                return False, f"{message} ({criterion}: {value})"
        else:
            if value < threshold:
                return False, f"{message} ({criterion}: {value:.2f} < {threshold})"
    
    # If all gates pass, check overall score
    overall = safe_get("overall_score", 0)
    if overall >= 7.0:
        return True, "Passed all gates"
    else:
        return False, f"Overall score below threshold ({overall:.2f} < 7.0)"

def categorize_rejection(row):
    """
    Categorize WHY an output was rejected.
    Based on paper's Table 11 rejection taxonomy.
    
    Returns:
        list of str: Rejection reason categories
    """
    reasons = []
    
    # Helper to safely get values
    def safe_get(key, default):
        val = row.get(key, default)
        if val is None:
            return default
        try:
            return float(val) if not isinstance(default, bool) else val
        except (TypeError, ValueError):
            return default
    
    # Check factual accuracy issues
    judge_relevance = safe_get("relevance_score", 5)
    judge_generalization = safe_get("generalization_score", 5)
    hallucination_score = safe_get("hallucination_score", 10)
    
    if hallucination_score < 8:
        reasons.append("hallucination")
    elif judge_relevance < 3.5:
        reasons.append("not_relevant")
    
    # Check generalization
    if judge_generalization < 3.0:
        # Could be too specific or too vague
        justification = str(row.get("judge_reason", "")).lower()
        if any(word in justification for word in ["specific", "narrow", "limited"]):
            reasons.append("too_specific")
        elif any(word in justification for word in ["generic", "vague", "non-specific"]):
            reasons.append("too_generic")
        else:
            reasons.append("poor_generalization")
    
    # Check length issues
    if not safe_get("in_range", True):
        char_len = len(str(row.get("marketing_text", "")))
        if char_len < 450:
            reasons.append("too_short")
        else:
            reasons.append("too_long")
    
    # Check judge feedback for other patterns
    justification = str(row.get("judge_reason", "")).lower()
    
    if any(word in justification for word in ["unclear", "confusing", "hard to understand"]):
        reasons.append("unclear")
    
    if any(word in justification for word in ["claim", "unsupported", "unverified"]):
        reasons.append("overclaim")
    
    if "bias" in justification or "stereotyp" in justification:
        reasons.append("bias")
    
    # Check feature coverage
    feat_cov = safe_get("semantic_feature_coverage_normalized", 1.0)
    if feat_cov < 0.3:
        reasons.append("missing_features")
    
    # Check attribute alignment
    attr_align = safe_get("semantic_attribute_alignment", 1.0)
    if attr_align < 0.35:
        reasons.append("off_topic")
    
    # If no specific reasons found, mark as general low quality
    if not reasons:
        reasons.append("low_overall_quality")
    
    return reasons

def analyze_rejection_patterns(results_df, threshold=7.0):
    """
    Generate rejection analysis report like paper's Table 1(b).
    Shows which issues are most common across all rejected outputs.
    
    Args:
        results_df: DataFrame with all results
        threshold: Score below which output is considered rejected
    
    Returns:
        dict: Rejection reason counts
    """
    from collections import Counter
    
    # Filter to rejected outputs
    rejected = results_df[results_df["overall_score"] < threshold].copy()
    
    if rejected.empty:
        print(f"\n{'='*70}")
        print(f"✅ NO REJECTIONS - All {len(results_df)} outputs passed!")
        print(f"{'='*70}\n")
        return {}
    
    # Collect all rejection reasons
    all_reasons = []
    for idx, row in rejected.iterrows():
        reasons = categorize_rejection(row)
        all_reasons.extend(reasons)
    
    reason_counts = Counter(all_reasons)
    total_rejected = len(rejected)
    
    # Print formatted report
    print(f"\n{'='*70}")
    print(f"REJECTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Total outputs: {len(results_df)}")
    print(f"Rejected (score < {threshold}): {total_rejected} ({total_rejected/len(results_df)*100:.1f}%)")
    print(f"Approved (score ≥ {threshold}): {len(results_df) - total_rejected} ({(len(results_df) - total_rejected)/len(results_df)*100:.1f}%)")
    print(f"\n{'Rejection Reason':<25s}  {'Count':>5s}  {'% of Rejected':>12s}")
    print(f"{'-'*70}")
    
    for reason, count in reason_counts.most_common():
        pct = (count / total_rejected * 100) if total_rejected > 0 else 0
        reason_display = reason.replace("_", " ").title()
        print(f"{reason_display:<25s}  {count:>5d}  {pct:>11.1f}%")
    
    print(f"{'='*70}\n")
    
    # Show worst example
    if not rejected.empty:
        worst_idx = rejected["overall_score"].idxmin()
        worst = rejected.loc[worst_idx]
        
        print(f"WORST EXAMPLE:")
        print(f"  Product: {worst.get('product_id', 'N/A')}")
        print(f"  Overall Score: {worst.get('overall_score', 0):.2f}/10")
        print(f"  Judge Score: {worst.get('judge_score', 0):.2f}/10")
        print(f"  Relevance: {worst.get('relevance_score', 0):.2f}/5")
        print(f"  Hallucination: {worst.get('hallucination_score', 0):.2f}/10")
        print(f"  Reasons: {', '.join(categorize_rejection(worst))}")
        print(f"  Judge Feedback: {worst.get('judge_reason', 'N/A')[:150]}...")
        print(f"\n")
    
    return reason_counts

# ===================== PROMPT LOGGING =====================

def write_prompts_snapshot(prompts: Dict[str, str], log_dir: str, iteration: int):
    """Write prompt snapshot."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(log_dir) / f"prompts_iter{iteration}.json"
    
    with open(snapshot_path, "w", encoding="utf-8") as f:
        safe_prompts = make_json_safe(prompts)
        json.dump(safe_prompts, f, indent=2, ensure_ascii=False)
    
    print(f"📸 Prompt snapshot saved: {snapshot_path}")

def write_prompt_log(entries: List[PromptLogEntry], log_dir: str) -> str:
    """Write prompt log with int64 handling."""
    import numpy as np
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"prompt_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    data = [asdict(e) for e in entries]
    
    # Custom JSON encoder to handle numpy/pandas types
    def json_default(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return round(float(obj), 3) 
        elif isinstance(obj, float):
            return round(obj, 3)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(log_path, "w", encoding="utf-8") as f:
        safe_data = make_json_safe(data)
        json.dump(safe_data, f, indent=2, ensure_ascii=False, default=json_default)
    
    return str(log_path)
