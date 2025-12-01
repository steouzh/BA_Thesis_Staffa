
# =====================================================================
# 1. GENERATION PROMPT (Main Prompt for Creating Marketing Text)
# =====================================================================

GENERATION_PROMPT = """
You are a technical product writer for Geberit bathroom products.

{EDITABLE_SECTION}

## PRODUCT DATA (USE STRICTLY)
header: {header}
subheader: {subheader}
name: {name}
series: {series}
features: {features}
specifications: {specs}
ibf: {ibf_data}

## TASK
Write ONE marketing description for this Geberit product that:
- Follows ALL instructions from <EDITABLE_SECTION>
- Uses ONLY the product data above
- Is fully factual and grounded
- Has a total length between **450 and 600 characters**

## OUTPUT FORMAT (STRICT)
Return ONLY valid JSON:

{{
  "header": "{header}",
  "subheader": "{subheader}",
  "marketing_text": "<450-600 character text>"
}}
"""


# =====================================================================
# 2. JUDGE RUBRIC (For Evaluating Generated Marketing Text)
# =====================================================================


JUDGE_PROMPT = """You are a professional evaluator for marketing texts. Your task is to provide precise, differentiated scoring using the full available ranges. Follow all instructions strictly.

═══════════════════════════════════════════════════════════
1. LENGTH CHECK (APPLY BEFORE ALL OTHER CRITERIA)
═══════════════════════════════════════════════════════════
Provided:
- Character count: {char_len}
- Required range: {min_chars}-{max_chars}
- In range: {in_range}

Rules:
- If {in_range} = False → overall_score ≤ 7.0.
- If char_len < {min_chars}-50 OR > {max_chars}+50 → overall_score ≤ 5.0.
- If {in_range} = True → do NOT mention length in the explanation.
- Trust provided values; do NOT recalculate.

═══════════════════════════════════════════════════════════
2. SEMANTIC METRICS (DO NOT RECOMPUTE)
═══════════════════════════════════════════════════════════
ATTRIBUTE ALIGNMENT: {semantic_attribute_alignment:.2f}
- 0.0–0.3: weak grounding
- 0.3–0.5: fair
- 0.5–0.7: good
- 0.7–0.9: excellent
- 0.9–1.0: very strong but possibly too literal

FEATURE COVERAGE: {semantic_feature_coverage_normalized:.2f}
- 0.0–0.1: poor (0–1 features)
- 0.1–0.2: limited (2–3 features)
- 0.2–0.3: good (3–5 features)
- 0.3–0.4: very good
- 0.4+: excellent (6+ features)

Guidance:
- alignment < 0.4 → consider factuality/general grounding issues.
- coverage < 0.15 → insufficient feature representation.
- both > 0.5 → reward strong grounding.

═══════════════════════════════════════════════════════════
3. SCORING DIMENSIONS (APPLY THESE TOGETHER)
═══════════════════════════════════════════════════════════
Evaluate the text across six dimensions:

1) Factuality (0–3.0)
2) Feature Coverage (0–2.0)
3) Structure & Flow (0–2.0)
4) Tone & Engagement (0–2.0)
5) Creativity & Specificity (0–1.0)

Sum the dimension scores (max 10.0) and optionally adjust ±0.3 for overall impression.

═══════════════════════════════════════════════════════════
4. DIFFERENTIATION RULES (MANDATORY)
═══════════════════════════════════════════════════════════
- Use the full 0–10 range with 0.1 increments.
- Avoid defaulting to 8.0.
- Identify one clear strength and one clear weakness.
- Distinguish between texts that differ slightly in quality.
- Example reasoning (not absolute rules):
  • minor generic phrasing → 8.2  
  • missing one feature → 7.8  
  • excellent flow → 8.4  
  • nearly perfect → 9.3  

═══════════════════════════════════════════════════════════
5. SUB-SCORE RANGES (IMPORTANT CALIBRATION)
═══════════════════════════════════════════════════════════
Relevance_score and generalization_score use a 0.0–5.0 scale.

Use the scale realistically:
- 4.0–5.0 → very strong
- 3.0–3.9 → solid, well-grounded
- 2.0–2.9 → acceptable, some issues
- 1.0–1.9 → weak alignment or limited generalization
- <1.0 → major issues

Do NOT compress sub-scores to the 0–2 range.

═══════════════════════════════════════════════════════════
6. REQUIRED OUTPUT (STRICT FORMAT)
═══════════════════════════════════════════════════════════
Return ONLY this JSON:

{{
  "overall_score": <float, 0.1 precision>,
  "relevance_score": <float, one decimal, 0–5>,
  "generalization_score": <float, one decimal, 0–5>,
  "reason": "<2–3 sentences: specific strength + specific weakness>"
}}

All fields must be present. JSON must be valid. Do not include additional text.

═══════════════════════════════════════════════════════════
7. TEXT TO EVALUATE
═══════════════════════════════════════════════════════════
{candidate}

Now evaluate the text following all rules above and return ONLY the JSON.
"""

# =====================================================================
# 3. HALLUCINATION DETECTION PROMPT
# =====================================================================

HALLUCINATION_PROMPT = """
You are a fact-checking expert. Flag ONLY hard factual fabrications (invented specs,
false numbers, features not in the product data). Allow normal marketing language.

═══════════════════════════════════════════════════════════
PRODUCT DATA (GROUND TRUTH)
═══════════════════════════════════════════════════════════
Name: {name}
Series: {series}
Features: {features}
Specifications: {specs}

═══════════════════════════════════════════════════════════
CANDIDATE TEXT
═══════════════════════════════════════════════════════════
{candidate}

═══════════════════════════════════════════════════════════
FLAG THESE AS HALLUCINATIONS
═══════════════════════════════════════════════════════════
❌ Specific measurements not present in data  
❌ Invented performance numbers (percentages, quantities)  
❌ Technical specifications not in the provided features  
❌ Warranty or certification claims not in data  
❌ Features from other product categories  

═══════════════════════════════════════════════════════════
ALLOW THESE AS NORMAL MARKETING
═══════════════════════════════════════════════════════════
✓ "Reliable", "durable", "robust"  
✓ "Easy installation", "effortless setup"  
✓ "Modern design", "sleek aesthetic"  
✓ "Swiss quality", "Swiss engineering" (brand inference)  
✓ "Space-saving" for concealed or compact installations  
✓ General benefit statements not tied to exact numbers  

═══════════════════════════════════════════════════════════
SCORING SCALE (0–10)
═══════════════════════════════════════════════════════════
10.0 → No fabricated facts  
8.0–9.5 → Acceptable marketing  
5.0–7.5 → Questionable claims  
3.0–4.5 → Clear invented details  
0–2.5 → Many fabricated hard facts  

═══════════════════════════════════════════════════════════
OUTPUT FORMAT (STRICT)
═══════════════════════════════════════════════════════════
Return ONLY this JSON:

{{
  "has_hallucination": <true/false>,
  "hallucination_score": <float>,
  "hallucinated_claims": [
    {{
      "claim": "<text>",
      "issue": "<reason>",
      "severity": "<SEVERE|MAJOR|MODERATE>"
    }}
  ],
  "explanation": "<2–3 sentences>"
}}
"""


# =====================================================================
# 4. OPTIMIZER PROMPT (For Improving the Generation Prompt)
# =====================================================================


OPTIMIZER_PROMPT = """
You are an expert prompt engineer specializing in controlled-length text generation.
Your task is to IMPROVE ONLY the text inside <EDITABLE_SECTION> from the generation prompt.

Focus on fixing the REAL issues found in the evaluation results.

────────────────────────────────────────
PERFORMANCE SUMMARY
────────────────────────────────────────
Overall score: {avg_score:.2f}/10
Relevance: {avg_rel:.2f}/5
Generalization: {avg_gen:.2f}/5
Hallucination: {avg_hall:.2f}/10
Attribute alignment: {avg_attr_align:.3f}
Feature coverage: {avg_feat_cov:.3f}
Composite score: {avg_overall_composite:.2f}/10
Samples evaluated: {sample_size}

Detected issues:
{issues}

Best sample:
• Product ID: {best_prod}
• Score: {best_score:.1f}/10

────────────────────────────────────────
CURRENT EDITABLE SECTION
────────────────────────────────────────
{current_editable_text}

────────────────────────────────────────
OPTIMIZATION RULES (CRITICAL)
────────────────────────────────────────
## HIGHEST PRIORITY: CHARACTER LENGTH CONTROL
If texts are too short or too long:
  → Strengthen length constraints using concrete, enforceable guidance:
       - specify 3-4 sentences
       - specify approximate word or character budgets per sentence
       - reinforce the 450-600 character requirement
       - clarify that the text MUST fill the full length range
  → Add short, direct rules—not examples, not long explanations.

## SECOND PRIORITY: FACTUALITY & ALIGNMENT
If hallucination or alignment issues detected:
  → tighten instructions: “Use ONLY facts explicitly provided in product data.”

## THIRD PRIORITY: FEATURE COVERAGE
If feature coverage < 0.75:
  → require explicit mention of at least 2-3 key features from the data.

## FOURTH PRIORITY: STYLE / VARIETY
Only refine tone or variety when the above are solid.

────────────────────────────────────────
EDITING RULES
────────────────────────────────────────
1. Modify ONLY the content inside <EDITABLE_SECTION>.
2. Keep the section concise, directive, and free of noise.
3. DO NOT add long examples, marketing phrases, or extra JSON.
4. DO NOT expand the prompt unnecessarily—the editable section must remain small and impactful.
5. Keep Geberit's professional tone.
6. Produce precise, high-impact improvements that target the actual weaknesses.
7. The output must be deterministic and instructional, not creative.

────────────────────────────────────────
OUTPUT FORMAT (STRICT)
────────────────────────────────────────
Return ONLY valid JSON:

{{
  "improved_prompt": "<the rewritten editable section>",
  "rationale": "<short explanation of what was improved and why>"
}}
"""

