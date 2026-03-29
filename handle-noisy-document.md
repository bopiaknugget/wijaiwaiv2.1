# Task: Handle Noisy Queries in RAG Pipeline

Implement a robust strategy to handle noisy / irrelevant / heterogeneous document queries in a production RAG system.

---

## Problem

* Documents are unrelated (mixed domains, topics, formats)
* Vector search returns noisy results
* Requires higher topK → increases latency
* Causes poor ranking and hallucinations

---

## Objectives

* Reduce search space before retrieval
* Improve precision of retrieved chunks
* Minimize latency and unnecessary compute
* Maintain high-quality context for LLM

---

## Requirements

### 1. Query Routing (Hybrid)

* Implement fast rule-based classification first
* If uncertain → fallback to LLM classification

```ts
function classifyQuery(query) {
  if (matchesResearch(query)) return "research";
  if (matchesCode(query)) return "code";
  return "unknown";
}
```

* If "unknown" → call LLM classifier

---

### 2. Metadata Filtering (Critical)

* Apply filters during vector search

```ts
index.query({
  vector,
  topK: 20,
  namespace: user_id,
  filter: {
    category: category,
    domain: domain
  }
});
```

* Metadata must include:

  * category
  * domain
  * doc_id

---

### 3. Dynamic TopK

* Adjust topK based on query confidence

```ts
if (high_confidence) topK = 10;
else topK = 30;
```

---

### 4. Score Threshold Filtering

* Drop low-relevance results early

```ts
if (score < 0.75) discard;
```

---

### 5. Re-ranking with Cutoff

* Retrieve → re-rank → reduce

```ts
retrieve topK = 30
rerank topK = 10
final = 5
```

---

### 6. Deduplication

* Remove near-duplicate chunks before LLM

```ts
removeNearDuplicates(chunks);
```

---

### 7. Context Length Control

* Limit total tokens before sending to LLM

```ts
limit_tokens(context, 2000);
```

---

### 8. Fallback Strategy

* If no results found:

  * Retry without filter
  * Or fallback to LLM-only answer

```ts
if (results.length === 0) retryWithoutFilter();
```

---

### 9. Optional: Multi-Index Routing

* Split indexes by domain

```ts
if (category === "research") use research-index;
```

---

## Flow

```txt
User Query
  ↓
Fast Classification (rule-based)
  ↓
(if unknown)
  → LLM Classification
  ↓
Apply Metadata Filter
  ↓
Vector Search (dynamic topK)
  ↓
Score Threshold Filtering
  ↓
Re-ranking
  ↓
Deduplication
  ↓
Context Assembly (token limit)
  ↓
LLM Generation
```

---

## Constraints

* Must minimize latency
* Avoid unnecessary LLM calls
* Keep pipeline async-ready
* Ensure deterministic behavior when possible

---

## Expected Outcome

* Reduced noise in retrieval
* Lower latency (40–70%)
* Improved answer accuracy
* Scalable RAG pipeline for heterogeneous data

---

## Notes

* Do NOT rely solely on LLM classification
* Retrieval optimization is more important than model size
* Always reduce search space before ranking
