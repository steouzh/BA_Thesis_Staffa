# benchmark_pipeline.py
import asyncio
import os
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from openai import AsyncOpenAI
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime

from benchmark_core import (
    CONFIG,
    datetime,
    RateLimiter,
    PromptLogEntry,
    make_json_safe,
    compute_overall_score,
    passes_quality_gates,
    categorize_rejection,
    analyze_rejection_patterns,
    write_prompts_snapshot,
    write_prompt_log,
    check_uniqueness,
    normalize_feature_coverage,
    compute_sbert_similarity,
    openai_rate_limiter,
    gemini_rate_limiter,
)

from benchmark_data import (
    inspect_columns,
    load_data,
    load_gold_if_exists,
)

from benchmark_llm_calls import (
    generate_openai_async,
    generate_gemini_async,
    judge_openai_async,
    judge_gemini_async,
    detect_hallucinations_openai_async,
    propose_improved_prompt_openai_async,
)

from prompts import GENERATION_PROMPT
from editable_section import INITIAL_EDITABLE_SECTION

from benchmark_evaluation import evaluate_batch_optimized

current_editable_section = INITIAL_EDITABLE_SECTION

# ===================== MAIN PIPELINE =====================

async def run_pipeline_async(cfg: Dict[str, Any]):
    """Main async pipeline - FULLY OPTIMIZED."""
    
    print("="*60)
    print("🚀 Starting OPTIMIZED Benchmark Pipeline")
    print("="*60)
    
    pipeline_start_time = datetime.now()
    
    # VALIDATE CONFIGURATION
    required_files = [("RAW_EXPORT_PATH", cfg.get("RAW_EXPORT_PATH"))]
    
    for name, path in required_files:
        if not path:
            raise ValueError(f"Missing required config: {name}")
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    
    # Validate API keys
    if cfg["USE_OPENAI"] and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY required but not set")
    
    if cfg["USE_GEMINI"] and not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY required but not set")
    
    # Initialize rate limiters
    global openai_rate_limiter, gemini_rate_limiter
    openai_rate_limiter = RateLimiter(requests_per_minute=cfg.get("OPENAI_RPM", 60))
    gemini_rate_limiter = RateLimiter(requests_per_minute=cfg.get("GEMINI_RPM", 60))
    
    # Create semaphores
    openai_semaphore = asyncio.Semaphore(cfg.get("MAX_CONCURRENT_OPENAI", 10))
    gemini_semaphore = asyncio.Semaphore(cfg.get("MAX_CONCURRENT_GEMINI", 10))
    
    # Load data
    if cfg.get("STANDARDIZED_INPUT_PATH") and os.path.exists(cfg["STANDARDIZED_INPUT_PATH"]):
        products_df = pd.read_excel(cfg["STANDARDIZED_INPUT_PATH"])
        print(f"✅ Loaded {len(products_df)} products from standardized input")
    else:
        products_df = load_data(cfg["RAW_EXPORT_PATH"], ibf_path=cfg.get("IBF_PATH"))
        print(f"✅ Loaded {len(products_df)} products")
    
    # Initialize models
    print("\n🤖 Initializing models...")
    sbert_model = SentenceTransformer("all-mpnet-base-v2")
    gold_df = load_gold_if_exists(cfg.get("GOLD_PATH"), sbert_model)
    
    # Initialize API clients
    openai_client = None
    if cfg.get("USE_OPENAI") or cfg.get("PRIMARY_JUDGE") == "openai":
        openai_client = AsyncOpenAI()
        print(f"✅ OpenAI initialized (model: {cfg['OPENAI_MODEL']})")
    
    if cfg.get("USE_GEMINI") or (cfg.get("SECONDARY_JUDGE") and cfg.get("SECONDARY_JUDGE_PROVIDER") == "gemini"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        print(f"✅ Gemini initialized (model: {cfg['GEMINI_MODEL']})")
    
    # Initialize prompts
    current_prompts = {pname: ptext for pname, ptext in cfg["PROMPTS"]}
    current_editable_section = INITIAL_EDITABLE_SECTION
    prompt_log_entries = []
    
    # Storage
    all_iters = []
    
    # Main iteration loop
    opt_iterations = cfg["OPT_ITERATIONS"] if cfg.get("OPTIMIZE") else 1
    
    for it in range(opt_iterations):
        iteration_start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"📍 Iteration {it+1} / {opt_iterations}")
        print(f"{'='*60}")

        full_assembled_prompt = GENERATION_PROMPT.replace(
            "{EDITABLE_SECTION}", 
            current_editable_section
        )
        
        # Update the prompts dictionary for this iteration
        current_prompts["prompt"] = full_assembled_prompt
        
        iter_results = []
        
        # GENERATION
        print(f"\n🎨 Generating marketing texts...")
        generation_tasks = []
        
        for idx, row in products_df.iterrows():
            prod_id = row["product_id"]
            
            
            for pname, ptext in current_prompts.items():
                
                pname_with_iter = f"{pname}_iter{it}"
                
                # Format prompt
                user_prompt = ptext
                user_prompt = user_prompt.replace('{name}', str(row.get("name", "")))
                user_prompt = user_prompt.replace('{series}', str(row.get("series", "")))
                user_prompt = user_prompt.replace('{features}', str(row.get("features", "")))
                user_prompt = user_prompt.replace('{specs}', str(row.get("specs", "")))
                user_prompt = user_prompt.replace('{ibf_data}', str(row.get("ibf_data", "")))
                
                # OpenAI generation
                if cfg.get("USE_OPENAI") and openai_client:
                    task = generate_openai_async(
                        openai_client,
                        cfg["OPENAI_MODEL"],
                        cfg["SYSTEM_PROMPT"],
                        user_prompt,
                        temperature=0.3,
                        semaphore=openai_semaphore,
                        max_attempts=cfg.get("RETRY_ATTEMPTS", 3),
                        context=f"generation_openai_{prod_id}"
                    )
                    generation_tasks.append(("openai", prod_id, pname_with_iter, row, task))
                
                # Gemini generation
                if cfg.get("USE_GEMINI"):
                    task = generate_gemini_async(
                        cfg["GEMINI_MODEL"],
                        cfg["SYSTEM_PROMPT"],
                        user_prompt,
                        temperature=0.3,
                        semaphore=gemini_semaphore,
                        max_attempts=cfg.get("RETRY_ATTEMPTS", 3),
                        context=f"generation_gemini_{prod_id}"
                    )
                    generation_tasks.append(("gemini", prod_id, pname_with_iter, row, task))
        
        # Execute generation
        if generation_tasks:
            print(f"⚙️ Executing {len(generation_tasks)} generation tasks...")
            generation_results = await asyncio.gather(*[task for _, _, _, _, task in generation_tasks])
            
            # Process results
            candidates = []
            for (model_name, prod_id, pname_with_iter, row, _), result in zip(generation_tasks, generation_results):
                if result:
                    candidates.append({
                        "model": model_name,
                        "product_id": prod_id,
                        "prompt_name": pname_with_iter,
                        "row": row,
                        "result": result
                    })
            
            print(f"✅ Generated {len(candidates)} candidates (success rate: {len(candidates)/len(generation_tasks)*100:.1f}%)")
            
            if not candidates:
                print("⚠️ No candidates generated, skipping iteration")
                continue
            
            # ============================================
            # BATCH METRICS COMPUTATION (THE FAST PART!)
            # ============================================
            
            print(f"\n📊 Computing metrics (BATCH MODE)...")
            import time
            start_eval = time.time()
            
            # Step 1: Build DataFrame with ALL candidates
            candidates_data = []
            for cand in candidates:
                result = cand["result"]
                row = cand["row"]
                
                candidates_data.append({
                    "marketing_text": result.get("marketing_text", ""),
                    "product_id": cand["product_id"],
                    "features": row.get("features", []),
                    "technical_data": row.get("technical_data", {}),
                    "IBF": row.get("ibf_data", ""),
                    "material": row.get("material", ""),
                    "colour": row.get("colour", ""),
                })
            
            candidates_df = pd.DataFrame(candidates_data)
            
            # Step 2: BATCH EVALUATE (10-50x faster!)
            evaluated_df = evaluate_batch_optimized(candidates_df)
            
            # Step 3: Map results back to candidates
            for i, cand in enumerate(candidates):
                row = cand["row"]
                
                # Build combined feature list (same list used for semantic_feature_coverage)
                combined = []

                # 1) Marketing features (Characteristics + Application purposes)
                features = row.get("features", [])
                if isinstance(features, list):
                    for f in features:
                        f_clean = str(f).strip()
                        if f_clean:
                            combined.append(f_clean)

                # 2) Technical specs (keys only)
                tech_data = row.get("technical_data", {})
                if isinstance(tech_data, dict):
                    for key in tech_data.keys():
                        key_clean = str(key).strip()
                        if key_clean:
                            combined.append(key_clean)

                # 3) IBF data (recommended)
                ibf_data = row.get("ibf_data", "")
                if isinstance(ibf_data, str):
                    for line in ibf_data.split("\n"):
                        line = line.strip()
                        if line:
                            combined.append(line)

                # FINAL FEATURE COUNT
                num_features = len(combined)
                

                
                cand["char_len"] = len(cand["result"]["marketing_text"])
                cand["in_range"] = cfg["MIN_CHARS"] <= cand["char_len"] <= cfg["MAX_CHARS"]
                cand["semantic_attribute_alignment"] = float(evaluated_df.iloc[i]["semantic_attribute_alignment"])
                cand["semantic_feature_coverage"] = float(evaluated_df.iloc[i]["semantic_feature_coverage"])
                
                # Normalize coverage
                feat_cov_raw = cand["semantic_feature_coverage"]

                cand["semantic_feature_coverage_normalized"] = normalize_feature_coverage(feat_cov_raw, num_features)
            
            eval_time = time.time() - start_eval
            print(f"✅ Metrics computed in {eval_time:.1f}s ({eval_time/len(candidates)*1000:.0f}ms per product)")
            
            if eval_time/len(candidates) > 1.0:
                print("⚠️ WARNING: Slower than expected - check evaluation_optimized import")
            
            # ============================================
            # JUDGING
            # ============================================
            
            print(f"\n⚖️  Judging candidates...")
            judging_tasks = []
            
            for cand in candidates:
                marketing_text = cand["result"]["marketing_text"]
                
                # Primary judge (OpenAI)
                if cfg.get("PRIMARY_JUDGE") == "openai" and openai_client:
                    task = judge_openai_async(
                        openai_client,
                        cfg["OPENAI_MODEL"],
                        marketing_text,
                        cfg["MIN_CHARS"],
                        cfg["MAX_CHARS"],
                        cand["char_len"],
                        cand["in_range"],
                        cand["semantic_attribute_alignment"],
                        cand["semantic_feature_coverage_normalized"],
                        semaphore=openai_semaphore,
                        max_attempts=cfg.get("RETRY_ATTEMPTS", 3),
                        context=f"judge_{cand['product_id']}"
                    )
                    judging_tasks.append(("primary", cand, task))
            
            # Execute judging
            if judging_tasks:
                judging_results = await asyncio.gather(*[task for _, _, task in judging_tasks])
                
                for (judge_type, cand, _), result in zip(judging_tasks, judging_results):
                    if judge_type == "primary" and result:
                        cand["judge_score"] = result.get("overall_score")
                        cand["relevance_score"] = result.get("relevance_score")
                        cand["generalization_score"] = result.get("generalization_score")
                        cand["judge_reason"] = result.get("reason", "")
            
            # ============================================
            # HALLUCINATION DETECTION
            # ============================================
            
            print(f"\n🔍 Detecting hallucinations...")
            hall_tasks = []
            
            for cand in candidates:
                row = cand["row"]
                marketing_text = cand["result"]["marketing_text"]
                
                if openai_client:
                    task = detect_hallucinations_openai_async(
                        openai_client,
                        cfg["OPENAI_MODEL"],
                        marketing_text,
                        row.get("name", ""),
                        row.get("series", ""),
                        str(row.get("features", [])),
                        row.get("specs", ""),
                        semaphore=openai_semaphore,
                        max_attempts=cfg.get("RETRY_ATTEMPTS", 3),
                        context=f"hallucination_{cand['product_id']}"
                    )
                    hall_tasks.append((cand, task))
            
            # Execute hallucination detection
            if hall_tasks:
                hall_results = await asyncio.gather(*[task for _, task in hall_tasks])
                
                for (cand, _), result in zip(hall_tasks, hall_results):
                    if result:
                        cand["has_hallucination"] = result.get("has_hallucination")
                        cand["hallucination_score"] = result.get("hallucination_score", 10.0)
                        cand["hallucinated_claims"] = result.get("hallucinated_claims", [])
            
            # ============================================
            # COMPUTE SBERT SIMILARITY TO GOLD
            # ============================================
            
            if gold_df is not None:
                print(f"\n📊 Computing SBERT similarity to gold...")
                for cand in candidates:
                    marketing_text = cand["result"]["marketing_text"]
                    cand["sbert"] = compute_sbert_similarity(marketing_text, gold_df, sbert_model)
            else:
                for cand in candidates:
                    cand["sbert"] = 0.0
            
            # ============================================
            # BUILD RESULTS
            # ============================================
            
            for cand in candidates:
                result_row = {
                    "iteration": it,
                    "model": cand["model"],
                    "product_id": cand["product_id"],
                    "prompt_name": cand["prompt_name"],
                    "header": cand["result"].get("header", ""),
                    "subheader": cand["result"].get("subheader", ""),
                    "marketing_text": cand["result"].get("marketing_text", ""),
                    "char_len": cand["char_len"],
                    "in_range": cand["in_range"],
                    "sbert": cand.get("sbert", 0.0),
                    "judge_score": cand.get("judge_score"),
                    "relevance_score": cand.get("relevance_score"),
                    "generalization_score": cand.get("generalization_score"),
                    "judge_reason": cand.get("judge_reason", ""),
                    "judge_score_secondary": None,
                    "relevance_score_secondary": None,
                    "generalization_score_secondary": None,
                    "has_hallucination": cand.get("has_hallucination"),
                    "hallucination_score": cand.get("hallucination_score", 10.0),
                    "hallucinated_claims": str(cand.get("hallucinated_claims", [])),
                    "semantic_feature_coverage": cand["semantic_feature_coverage"],
                    "semantic_feature_coverage_normalized": cand["semantic_feature_coverage_normalized"],
                    "semantic_attribute_alignment": cand["semantic_attribute_alignment"],
                }
                
                # Compute overall score
                result_row["overall_score"] = compute_overall_score(result_row)
                
                # ===== NEW: Add quality gate checking =====
                passed_gates, gate_reason = passes_quality_gates(result_row)
                result_row["passed_quality_gates"] = passed_gates
                result_row["gate_failure_reason"] = gate_reason if not passed_gates else ""
                
                # ===== NEW: Add rejection categorization =====
                if not passed_gates:
                    result_row["rejection_reasons"] = ", ".join(categorize_rejection(result_row))
                else:
                    result_row["rejection_reasons"] = ""
                # ===== END NEW =====
                
                iter_results.append(result_row)
            
        # Convert to DataFrame
        iter_df = pd.DataFrame(iter_results) if iter_results else pd.DataFrame()
        
        if not iter_df.empty:
            all_iters.append(iter_df)
            
            # CHECKPOINT
            checkpoint_path = Path(cfg["OUTPUT_PATH"]).with_suffix(f'.iter{it}.checkpoint.xlsx')
            iter_df.to_excel(checkpoint_path, index=False)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

            print(f"\n📊 Quality Analysis for Iteration {it}:")
            rejection_patterns = analyze_rejection_patterns(iter_df, threshold=7.0)
        
        # OPTIMIZATION (with EARLY EXIT FIX!)
        if cfg.get("OPTIMIZE") and it < opt_iterations - 1 and not iter_df.empty:
            print(f"\n🔧 Running prompt optimization...")
            
            current_patterns = analyze_rejection_patterns(iter_df, threshold=7.5)
            
            if current_patterns:
                top_3_issues = current_patterns.most_common(3)
                issues_summary = "\n".join([
                    f"  • {issue}: {count} occurrences ({count/len(iter_df)*100:.1f}%)"
                    for issue, count in top_3_issues
                ])
                print(f"\n🎯 TOP ISSUES THIS ITERATION:\n{issues_summary}\n")
            else:
                issues_summary = "  • No significant issues detected"

            for pname in current_prompts.keys():
                sub = iter_df[iter_df["prompt_name"].str.startswith(pname)].copy()
                
                if sub.empty:
                    continue
                
                # Compute metrics
                avg_overall = float(sub["judge_score"].dropna().mean()) if not sub["judge_score"].dropna().empty else 0.0
                avg_rel = float(sub["relevance_score"].dropna().mean()) if not sub["relevance_score"].dropna().empty else 0.0
                avg_gen = float(sub["generalization_score"].dropna().mean()) if not sub["generalization_score"].dropna().empty else 0.0
                avg_len = float(sub["char_len"].mean()) if not sub["char_len"].empty else 0.0
                avg_feats = 0.0
                avg_hall = float(sub["hallucination_score"].dropna().mean()) if not sub["hallucination_score"].dropna().empty else 0.0
                avg_attr_align = float(sub["semantic_attribute_alignment"].dropna().mean()) if not sub["semantic_attribute_alignment"].dropna().empty else 0.0
                avg_feat_cov = float(sub["semantic_feature_coverage_normalized"].dropna().mean()) if not sub["semantic_feature_coverage_normalized"].dropna().empty else 0.0
                avg_overall_composite = float(sub["overall_score"].dropna().mean()) if not sub["overall_score"].dropna().empty else 0.0
                sample_size = len(sub)
                
                # Find best example
                best_idx = sub["overall_score"].idxmax()
                best_example = {
                    "product_id": sub.loc[best_idx, "product_id"] if not pd.isna(best_idx) else "N/A",
                    "score": sub.loc[best_idx, "overall_score"] if not pd.isna(best_idx) else 0.0,
                    "length": sub.loc[best_idx, "char_len"] if not pd.isna(best_idx) else 0,
                    "model": sub.loc[best_idx, "model"] if not pd.isna(best_idx) else "N/A"
                }
                
                # Identify issues
                issues = []
                if avg_overall < cfg.get("OPT_THRESHOLD", 8.5):
                    issues.append(f"Average score {avg_overall:.1f} below threshold {cfg.get('OPT_THRESHOLD', 8.5)}")
                if avg_len < cfg["MIN_CHARS"]:
                    issues.append(f"Texts too short (avg {avg_len:.0f} chars)")
                if avg_len > cfg["MAX_CHARS"]:
                    issues.append(f"Texts too long (avg {avg_len:.0f} chars)")
                if avg_feats < 3:
                    issues.append(f"Not enough features (avg {avg_feats:.1f})")
                if avg_hall < 8.0:
                    issues.append(f"High hallucination rate ({avg_hall:.1f}/10)")
                if avg_attr_align < 0.50:
                    issues.append(f"Low attribute alignment ({avg_attr_align:.3f})")
                if avg_feat_cov < 0.70:
                    issues.append(f"Low feature coverage ({avg_feat_cov:.3f})")

                from collections import Counter
                
                sub_rejected = sub[sub["overall_score"] < 7.5].copy()
                
                if not sub_rejected.empty:
                    all_rejection_reasons = []
                    for idx, row in sub_rejected.iterrows():
                        reasons = categorize_rejection(row)
                        all_rejection_reasons.extend(reasons)
                    
                    if all_rejection_reasons:
                        reason_counts = Counter(all_rejection_reasons)
                        total_rejected = len(sub_rejected)
                        
                        issues.append(f"\n🎯 REJECTION PATTERNS FOR {pname} ({total_rejected}/{len(sub)} outputs failed):")
                        
                        for reason, count in reason_counts.most_common(3):
                            pct = count / total_rejected * 100
                            reason_display = reason.replace("_", " ").title()
                            issues.append(f"  • {reason_display}: {count} ({pct:.0f}%)")
                        
                        top_reason = reason_counts.most_common(1)[0][0]
                        
                        recommendations = {
                            "too_generic": "⚠️ CRITICAL: Require specific technical features with measurements",
                            "missing_features": "⚠️ CRITICAL: Require 3-5 concrete product features",
                            "too_short": "⚠️ CRITICAL: Make 450-600 char requirement MORE prominent",
                            "too_long": "⚠️ CRITICAL: Prioritize top 3 features only",
                            "not_relevant": "⚠️ CRITICAL: Every claim MUST reference product data",
                            "hallucination": "⚠️ CRITICAL: ZERO invented specifications allowed",
                            "unclear": "⚠️ CRITICAL: Write for non-technical buyers",
                            "overclaim": "⚠️ CRITICAL: Ban superlatives unless in product data",
                            "off_topic": "⚠️ CRITICAL: Strengthen connection to product attributes",
                        }
                        
                        if top_reason in recommendations:
                            issues.append(f"\n{recommendations[top_reason]}")
                # ===== END NEW BLOCK =====
                
                # EARLY EXIT: Skip if all targets met
                if not issues:
                    print(f"✅ {pname}: All targets met (score={avg_overall:.1f}) - no optimization needed")
                    continue
                
                # Run optimizer
                print(f"⚠️ {pname}: {len(issues)} issues detected - optimizing...")
                issues_text = "\n".join(f"- {issue}" for issue in issues)
                
                trigger_data = {
                    "avg_score": round(avg_overall, 3),
                    "avg_relevance": round(avg_rel, 3),
                    "avg_generalization": round(avg_gen, 3),
                    "avg_length": round(avg_len, 3),
                    "avg_hallucination_score": round(avg_hall, 3),
                    "avg_attribute_alignment": round(avg_attr_align, 3),
                    "avg_feature_coverage": round(avg_feat_cov, 3),
                    "avg_overall_composite": round(avg_overall_composite, 3),
                    "sample_size": sample_size,
                    "best_example": best_example,
                }
                
                opt_start = time.time()
                suggestion = await propose_improved_prompt_openai_async(
                    openai_client,
                    cfg["OPENAI_MODEL"],
                    current_prompts[pname],
                    issues_text,
                    cfg["MIN_CHARS"],
                    cfg["MAX_CHARS"],
                    avg_overall,
                    avg_rel,
                    avg_gen,
                    avg_len,
                    avg_feats,
                    avg_hall,
                    avg_attr_align,
                    avg_feat_cov,
                    avg_overall_composite,
                    sample_size,
                    best_example,
                    target_score=cfg.get("OPT_THRESHOLD", 8.5)
                )
                
                improved = suggestion.get("improved_prompt", current_editable_section)
                
                if improved and improved != current_editable_section:
                    opt_runtime = time.time() - opt_start
                    pname_versioned = f"{pname}_iter{it+1}"

                    # Round best_example score for logging
                    trigger_data_logged = trigger_data.copy()
                    if "best_example" in trigger_data_logged:
                        best_ex = trigger_data_logged["best_example"].copy()
                        if "score" in best_ex:
                            best_ex["score"] = round(best_ex["score"], 3)
                        trigger_data_logged["best_example"] = best_ex
                    
                    prompt_log_entries.append(PromptLogEntry(
                        timestamp=datetime.now().isoformat(timespec="seconds"),
                        iteration=it+1,
                        prompt_name=pname_versioned,
                        original_prompt=current_editable_section,
                        improved_prompt=improved,
                        rationale=suggestion.get("rationale", ""),
                        runtime_seconds=round(opt_runtime, 2),
                        trigger_data=trigger_data_logged,
                    ))
                    
                    current_editable_section = improved
                    print(f"✅ {pname} optimized in {opt_runtime:.1f}s")
                else:
                    print(f"ℹ️ {pname} unchanged")
            
            write_prompts_snapshot(current_prompts, cfg["PROMPT_LOG_DIR"], iteration=it+1)
        
        iteration_runtime = (datetime.now() - iteration_start_time).total_seconds()
        print(f"\n⏱️ Iteration {it} completed in {iteration_runtime:.1f}s ({iteration_runtime/60:.1f} minutes)")
    
    # Write prompt log
    if prompt_log_entries:
        log_path = write_prompt_log(prompt_log_entries, cfg["PROMPT_LOG_DIR"])
        print(f"\n📝 Prompt log written to: {log_path}")
    
    # Combine results
    print(f"\n📊 Finalizing results...")
    full_df = pd.concat(all_iters, ignore_index=True) if all_iters else pd.DataFrame()
    
    # Check uniqueness
    if not full_df.empty:
        print(f"\n🧩 Checking uniqueness...")
        full_df["is_unique"] = False
        full_df["duplicate_of"] = None
        
        for it_val in full_df["iteration"].unique():
            sub = full_df[full_df["iteration"] == it_val].copy()
            sub_checked = check_uniqueness(sub, threshold=0.95)
            full_df.loc[sub_checked.index, ["is_unique", "duplicate_of"]] = sub_checked[["is_unique", "duplicate_of"]]
            dup_rate = 1 - sub_checked["is_unique"].mean()
            print(f"  Iteration {it_val}: {dup_rate:.1%} duplicate rate")
    
    if not full_df.empty:
        print(f"\n{'='*70}")
        print(f"FINAL QUALITY SUMMARY (All Iterations)")
        print(f"{'='*70}")
        
        # ===== CAPTURE the patterns instead of just printing =====
        rejection_patterns = analyze_rejection_patterns(full_df, threshold=7.0)
        
        # ===== NOW USE IT: Save to file for tracking over time =====
        if rejection_patterns:
            top_issue = rejection_patterns.most_common(1)[0]
            top_issue_name = top_issue[0]
            top_issue_count = top_issue[1]
            
            # Create actionable summary
            summary = {
                "timestamp": datetime.now().isoformat(),
                "total_outputs": len(full_df),
                "rejected_count": (full_df["overall_score"] < 7.0).sum(),
                "rejection_rate": float((full_df["overall_score"] < 7.0).mean()),
                "top_issue": top_issue_name,
                "top_issue_count": int(top_issue_count),
                "all_patterns": {k: int(v) for k, v in rejection_patterns.items()},
            }
            
            # Save summary for tracking
            summary_path = Path(cfg["OUTPUT_PATH"]).with_suffix('.quality_summary.json')
            with open(summary_path, 'w') as f:
                safe_summary = make_json_safe(summary)
                json.dump(safe_summary, f, indent=2)

            
            print(f"\n💾 Quality summary saved: {summary_path}")
            print(f"🎯 TOP ISSUE TO FIX: '{top_issue_name}' ({top_issue_count} occurrences)")
            print(f"   → Your next optimization should target this pattern!\n")
        
    # Show improvement over iterations if multiple iterations
    if len(full_df["iteration"].unique()) > 1:
        print(f"\n📈 QUALITY IMPROVEMENT OVER ITERATIONS:")
        print(f"{'Iteration':<12s}  {'Mean Score':>11s}  {'Pass Rate':>10s}")
        print(f"{'-'*70}")
        for it_val in sorted(full_df["iteration"].unique()):
            it_sub = full_df[full_df["iteration"] == it_val]
            mean_score = it_sub["overall_score"].mean()
            pass_rate = (it_sub["overall_score"] >= 8.0).mean()
            print(f"{it_val:<12d}  {mean_score:>11.2f}  {pass_rate:>9.1%}")
        print()

    # Write results
    print(f"\n💾 Writing results to {cfg['OUTPUT_PATH']}...")
    
    with pd.ExcelWriter(cfg["OUTPUT_PATH"], engine="xlsxwriter") as xw:
        # FIRST: Convert booleans (but work on copies to avoid changing original)
        full_df_export = full_df.copy()
        full_df_export = full_df_export.replace({True: 'TRUE', False: 'FALSE'}).infer_objects(copy=False)

        
        # THEN: Write to Excel
        full_df_export.to_excel(xw, index=False, sheet_name="results")
        
        if not full_df.empty:
            # Create aggregate from ORIGINAL data (numeric booleans work for mean)
            agg = (
                full_df.groupby(["iteration", "model", "prompt_name"])
                .agg(
                    overall_score=("overall_score", "mean"),
                    char_len=("char_len", "mean"),
                    sbert=("sbert", "mean"),
                    judge_score=("judge_score", "mean"),
                    relevance_score=("relevance_score", "mean"),
                    generalization_score=("generalization_score", "mean"),
                    judge_score_secondary=("judge_score_secondary", "mean"),
                    relevance_score_secondary=("relevance_score_secondary", "mean"),
                    generalization_score_secondary=("generalization_score_secondary", "mean"),
                    hallucination_score=("hallucination_score", "mean"),
                    hallucination_rate=("has_hallucination", "mean"),
                    semantic_feature_coverage=("semantic_feature_coverage", "mean"),
                    semantic_feature_coverage_normalized=("semantic_feature_coverage_normalized", "mean"),
                    semantic_attribute_alignment=("semantic_attribute_alignment", "mean"),
                    
                )
                .reset_index()
            )
            
            # Convert aggregate booleans too
            agg = agg.replace({True: 'TRUE', False: 'FALSE'}).infer_objects(copy=False)
            
            # Write aggregate
            agg.to_excel(xw, index=False, sheet_name="aggregate")

        workbook = xw.book
        
        # Define formats
        fmt_1 = workbook.add_format({'num_format': '0.0', 'align': 'center'})
        fmt_3 = workbook.add_format({'num_format': '0.000', 'align': 'center'})
        fmt_text = workbook.add_format({'align': 'center'}) 
        header_fmt = workbook.add_format({
            'bold': True,
            'bg_color': '#D7E4BD',
            'align': 'center',
            'valign': 'vcenter',
            'text_wrap': True,
            'border': 1
        })
        
        for sheet_name in xw.sheets:
            ws = xw.sheets[sheet_name]
            df = full_df_export if sheet_name == "results" else agg 
            
            # Column groups
            decimal_1_cols = ['judge_score', 'relevance_score', 'generalization_score', 
                            'hallucination_score', 'judge_score_secondary', 
                            'relevance_score_secondary', 'generalization_score_secondary', 'char_len']
            decimal_3_cols = ['overall_score', 'sbert', 'semantic_feature_coverage',
                            'semantic_feature_coverage_normalized', 'semantic_attribute_alignment',
                            'hallucination_rate']
            text_cols = ['marketing_text', 'header', 'subheader', 'product_id', 'prompt_name']
            
            for col_idx, col_name in enumerate(df.columns):
                # Calculate width
                if col_name in text_cols:
                    if col_name == 'marketing_text':
                        width = 60
                    elif col_name in ['header', 'subheader']:
                        width = 40
                    else:
                        width = 20
                else:
                    # Auto-calculate based on content
                    max_len = len(str(col_name))
                    if len(df) > 0:
                        max_len = max(max_len, df[col_name].astype(str).str.len().max())
                    width = min(max_len + 2, 30)
                
                # Apply formatting
                if col_name in decimal_1_cols:
                    ws.set_column(col_idx, col_idx, width, fmt_1)
                elif col_name in decimal_3_cols:
                    ws.set_column(col_idx, col_idx, width, fmt_3)
                else:
                    ws.set_column(col_idx, col_idx, width, fmt_text)
                
                # Apply header format
                ws.write(0, col_idx, col_name, header_fmt)
            
            # Freeze header row
            ws.freeze_panes(1, 0)

    print(f"✅ Formatted Excel saved to: {cfg['OUTPUT_PATH']}")
    
    pipeline_runtime = (datetime.now() - pipeline_start_time).total_seconds()
    print(f"\n{'='*60}")
    print(f"✅ Pipeline complete!")
    print(f"⏱️ Total runtime: {pipeline_runtime:.1f}s ({pipeline_runtime/60:.1f} minutes)")
    print(f"📁 Results saved to: {cfg['OUTPUT_PATH']}")
    print(f"{'='*60}\n")


# ===================== MAIN =====================

if __name__ == "__main__":
    INSPECT_MODE = False

    if INSPECT_MODE:
        print("🔍 INSPECTION MODE")
        inspect_columns(CONFIG["RAW_EXPORT_PATH"])
        raise SystemExit(0)

    CONFIG["USE_OPENAI"] = True
    CONFIG["USE_GEMINI"] = True
    CONFIG["PROMPTS"] = [("prompt", GENERATION_PROMPT)]

    CONFIG["PRIMARY_JUDGE"] = "openai"
    CONFIG["SECONDARY_JUDGE"] = False

    CONFIG["MIN_CHARS"] = 450
    CONFIG["MAX_CHARS"] = 600

    CONFIG["OPTIMIZE"] = True
    CONFIG["OPT_ITERATIONS"] = 2
    CONFIG["OPT_THRESHOLD"] = 8.5

    CONFIG["OPENAI_MODEL"] = "gpt-4o-mini"
    CONFIG["GEMINI_MODEL"] = "gemini-2.0-flash"

    CONFIG["RAW_EXPORT_PATH"] = "pipeline/export_test4.xlsx"
    CONFIG["GOLD_PATH"] = ""
    CONFIG["OUTPUT_PATH"] = f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
    CONFIG["PROMPT_LOG_DIR"] = "prompt_logs"
    CONFIG["IBF_PATH"] = "pipeline/ibf_data2.xlsx"

    asyncio.run(run_pipeline_async(CONFIG))
