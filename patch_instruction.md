# PATCH Editor workspace (DO NOT CHANGE EXISTING STRUCTURE)
# Context: OpenThai GPT as LLM, RAG data retrieved from Pinecone 
# Scope: You MUST actively apply RAG Context for all buttons in the editor (ตรวจงานวิจัย, สร้างเนื้อหาวิจัย, edit insert), including optional functions.

# STRICT SYSTEM CONSTRAINTS
You must ONLY enhance existing logic, inject RAG usage, improve token handling, and reduce hallucinations.
You must NOT rename variables, change function names, modify existing interfaces, or restructure the system.

---

# CORE EXECUTION RULES

## 1. RAG Integration (Mandatory)
Determine relevance BEFORE using RAG. RAG is considered relevant ONLY IF semantically aligned with editor content and the button action.
- IF relevance is LOW, UNCLEAR, or RAG is empty: Treat as NO relevant RAG and fallback to model knowledge.
- DO NOT copy raw RAG text.
- DO NOT fabricate references from RAG.

## 2. Token Handling & Safety Limits (Hard Constraint)
If input is too long, you MUST NEVER process full content in one single pass.
- Split into logical chunks (by section/heading) and limit the number of chunks per cycle.
- Process chunks by prioritizing: 1) Button action, 2) Relevant editor content, 3) Relevant RAG context.
- Merge results at the end. 
- IF still too large: Summarize chunks FIRST, then process the summaries.

---

# POST-PROCESS VERIFICATION & VALIDATION (MANDATORY)
You must perform the following checks internally before returning the final output. If ANY check fails, you must re-process and adjust the output conservatively.

## STEP 1: Fact, Source & Hallucination Check
- Verify every key claim is supported by the RAG context OR derived from editor content.
- DO NOT invent citations, statistics, or references.
- **Failure Handling:** If any claim is unsupported, risky, or you are unsure, keep the original content unchanged or remove/rewrite the claim conservatively.

## STEP 2: Task Validation
- Output exactly matches the `button action`.
- Scope is correct (no unnecessary sections added, no unintended overwrites).

## STEP 3: Structure Validation
- Output format matches the existing schema EXACTLY.
- No new fields introduced, no keys renamed, and no missing required fields.

## STEP 4: Mock Benchmark Simulation
Simulate 5 internal tests per case using the generated output (Thai/Eng):
- **Case A (Minimal Input):** Would this output still be valid if the input was shorter?
- **Case B (No RAG):** Would this output still be safe/accurate without RAG?
- **Case C (Conflicting RAG):** Would this output remain consistent if RAG had conflicting info?
*Expectation: The output must remain stable under all cases, avoid hallucination, and not over-claim certainty.*

## FINAL CHECK
Ensure Verification, Validation, and Benchmark simulations have all passed before delivering the final response.