# benchmark_llm_calls.py
from typing import Dict, Any, Optional, List

from openai import AsyncOpenAI

import asyncio
from benchmark_core import (
    call_openai_with_retry,
    call_gemini_with_retry,
    safe_json_parse,
)
from prompts import JUDGE_PROMPT, HALLUCINATION_PROMPT, OPTIMIZER_PROMPT

# ===================== GENERATION =====================

async def generate_openai_async(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    semaphore: Optional[asyncio.Semaphore] = None,
    max_attempts: int = 3,
    context: str = "generation"
) -> Optional[Dict]:
    """Generate marketing text using OpenAI."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    raw = await call_openai_with_retry(
        client, model, messages, temperature, 
        max_tokens=1000, semaphore=semaphore, 
        max_attempts=max_attempts, context=context
    )
    
    if not raw:
        return None
    
    parsed = safe_json_parse(raw)
    if parsed and all(k in parsed for k in ["header", "subheader", "marketing_text"]):
        return parsed
    
    print(f"⚠️ Invalid generation response in {context}")
    return None

async def generate_gemini_async(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    semaphore: Optional[asyncio.Semaphore] = None,
    max_attempts: int = 3,
    context: str = "generation"
) -> Optional[Dict]:
    """Generate marketing text using Gemini."""
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    raw = await call_gemini_with_retry(
        model_name, full_prompt, temperature,
        max_output_tokens=1000, semaphore=semaphore, 
        max_attempts=max_attempts, context=context
    )
    
    if not raw:
        return None
    
    parsed = safe_json_parse(raw)
    if parsed and all(k in parsed for k in ["header", "subheader", "marketing_text"]):
        return parsed
    
    print(f"⚠️ Invalid generation response in {context}")
    return None


# ===================== JUDGING =====================

async def judge_openai_async(
    client: AsyncOpenAI,
    model: str,
    candidate: str,
    min_chars: int,
    max_chars: int,
    char_len: int,
    in_range: bool,
    semantic_attribute_alignment: float = 0.0,
    semantic_feature_coverage_normalized: float = 0.0,
    semaphore: Optional[asyncio.Semaphore] = None,
    max_attempts: int = 3,
    context: str = "judging"
) -> Dict[str, Any]:
    """Judge candidate text using OpenAI."""
    prompt = JUDGE_PROMPT.format(
        candidate=candidate,
        min_chars=min_chars,
        max_chars=max_chars,
        char_len=char_len,
        in_range=in_range,
        semantic_attribute_alignment=semantic_attribute_alignment,
        semantic_feature_coverage_normalized=semantic_feature_coverage_normalized
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    raw = await call_openai_with_retry(
        client, model, messages, temperature=0.3,
        max_tokens=500, semaphore=semaphore, 
        max_attempts=max_attempts, context=context
    )
    
    if not raw:
        return {"overall_score": None, "relevance_score": None, "generalization_score": None, "reason": "API error"}
    
    parsed = safe_json_parse(raw)
    if parsed:
        return {
            "overall_score": parsed.get("overall_score"),
            "relevance_score": parsed.get("relevance_score"),
            "generalization_score": parsed.get("generalization_score"),
            "reason": parsed.get("reason", "")
        }
    
    return {"overall_score": None, "relevance_score": None, "generalization_score": None, "reason": "Parse error"}

async def judge_gemini_async(
    model_name: str,
    candidate: str,
    min_chars: int,
    max_chars: int,
    semaphore: Optional[asyncio.Semaphore] = None,
    max_attempts: int = 3,
    context: str = "judging"
) -> Dict[str, Any]:
    """Judge candidate text using Gemini."""
    prompt = JUDGE_PROMPT.format(
        candidate=candidate,
        min_chars=min_chars,
        max_chars=max_chars
    )
    
    raw = await call_gemini_with_retry(
        model_name, prompt, temperature=0.3,
        max_output_tokens=500, semaphore=semaphore, 
        max_attempts=max_attempts, context=context
    )
    
    if not raw:
        return {"overall_score": None, "relevance_score": None, "generalization_score": None, "reason": "API error"}
    
    parsed = safe_json_parse(raw)
    if parsed:
        return {
            "overall_score": parsed.get("overall_score"),
            "relevance_score": parsed.get("relevance_score"),
            "generalization_score": parsed.get("generalization_score"),
            "reason": parsed.get("reason", "")
        }
    
    return {"overall_score": None, "relevance_score": None, "generalization_score": None, "reason": "Parse error"}


# ===================== HALLUCINATION DETECTION =====================

async def detect_hallucinations_openai_async(
    client: AsyncOpenAI,
    model: str,
    candidate: str,
    name: str,
    series: str,
    features: str,
    specs: str,
    semaphore: Optional[asyncio.Semaphore] = None,
    max_attempts: int = 3,
    context: str = "hallucination"
) -> Dict[str, Any]:
    """Detect hallucinations."""
    prompt = HALLUCINATION_PROMPT.format(
        candidate=candidate,
        name=name,
        series=series,
        features=features,
        specs=specs
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    raw = await call_openai_with_retry(
        client, model, messages, temperature=0.3,
        max_tokens=500, semaphore=semaphore, 
        max_attempts=max_attempts, context=context
    )
    
    if not raw:
        return {"has_hallucination": False, "hallucination_score": 10.0, "hallucinated_claims": [], "explanation": "API error"}
    
    parsed = safe_json_parse(raw)
    if parsed:
        return {
            "has_hallucination": parsed.get("has_hallucination"),
            "hallucination_score": parsed.get("hallucination_score", 10.0),
            "hallucinated_claims": parsed.get("hallucinated_claims", []),
            "explanation": parsed.get("explanation", "")
        }
    
    return {"has_hallucination": False, "hallucination_score": 10.0, "hallucinated_claims": [], "explanation": "Parse error"}

# ===================== PROMPT OPTIMIZATION =====================

async def propose_improved_prompt_openai_async(
    client: AsyncOpenAI,
    model: str,
    current_prompt: str,
    issues: str,
    min_chars: int,
    max_chars: int,
    avg_score: float = 0.0,
    avg_rel: float = 0.0,
    avg_gen: float = 0.0,
    avg_len: float = 0.0,
    avg_feats: float = 0.0,
    avg_hall: float = 0.0,
    avg_attr_align: float = 0.0,
    avg_feat_cov: float = 0.0,
    avg_overall_composite: float = 0.0,
    sample_size: int = 0,
    best_example: Optional[Dict] = None,
    target_score: float = 8.5
) -> Dict[str, str]:
    """Propose improved prompt (Optimized to handle JSON parsing safely)."""
    
    best_example = best_example or {}
    
    # 1. Extract only the editable text to send to the LLM (Reduces token load/confusion)
    try:
        # Split roughly where the editable section starts
        if "<EDITABLE_SECTION>" in current_prompt:
            parts = current_prompt.split("<EDITABLE_SECTION>")
            protected_part = parts[0] + "<EDITABLE_SECTION>\n"
            # Get the content inside the tags
            existing_editable_content = parts[1].split("</EDITABLE_SECTION>")[0].strip()
        else:
            # Fallback if tags missing
            protected_part = ""
            existing_editable_content = current_prompt
    except Exception as e:
        print(f"⚠️ Prompt splitting error: {e}")
        existing_editable_content = current_prompt[-500:] # Fallback
        protected_part = ""

    # 2. Format the new optimizer prompt
    prompt = OPTIMIZER_PROMPT.format(
        current_prompt="(Full prompt hidden to save tokens)", 
        current_editable_text=existing_editable_content, # NEW FIELD in prompt
        avg_score=avg_score,
        avg_rel=avg_rel,
        avg_gen=avg_gen,
        avg_len=avg_len,
        avg_feats=avg_feats,
        avg_hall=avg_hall,
        avg_attr_align=avg_attr_align,
        avg_feat_cov=avg_feat_cov,
        avg_overall_composite=avg_overall_composite,
        sample_size=sample_size,
        min_chars=min_chars,
        max_chars=max_chars,
        target_score=target_score,
        issues=issues,
        best_prod=best_example.get("product_id", "N/A"),
        best_score=best_example.get("score", 0.0),
        best_len=best_example.get("length", 0),
        best_model=best_example.get("model", "N/A")
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    raw = await call_openai_with_retry(
        client, model, messages, temperature=0.7,
        max_tokens=2000, semaphore=None, max_attempts=3,
        context="optimization"
    )
    
    raw_text = raw.strip()

    # If response does not start with "{", wrap it so regex can find JSON inside
    if not raw_text.startswith("{"):
        wrapped = "{ " + raw_text + " }"
    else:
        wrapped = raw_text

    # Pass wrapped text into parser
    parsed = safe_json_parse(wrapped)

    
    # 3. Reconstruct the full prompt
    if parsed:
        rationale = parsed.get("rationale", "No rationale provided.")
        
        # Scenario A: New optimized format (Recommended)
        if "editable_content_only" in parsed:
            new_content = parsed["editable_content_only"]
            # Stitch it back together
            if protected_part:
                full_improved_prompt = f"{protected_part}{new_content}\n</EDITABLE_SECTION>\n"
                return {"improved_prompt": full_improved_prompt, "rationale": rationale}
            else:
                return {"improved_prompt": current_prompt, "rationale": "Could not reconstruct prompt structure."}

        # Scenario B: Legacy format (The LLM returned 'improved_prompt' anyway)
        elif "improved_prompt" in parsed:
            return parsed
            
    return {"improved_prompt": current_prompt, "rationale": "JSON Parse error"}

