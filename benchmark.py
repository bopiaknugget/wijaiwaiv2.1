"""
Benchmark suite for Research Workbench core modules.
Tests database.py and document_loader.py without requiring the embedding model.
"""

import sys
import os
import time
import timeit
import uuid
import tempfile
import shutil

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(__file__))

# Use a temporary DB for benchmarks so we don't pollute the real DB
_tmp_dir = tempfile.mkdtemp()
_tmp_db = os.path.join(_tmp_dir, "benchmark_notes.db")

# Patch database module to use temp DB before importing
import database as _db_module_patcher
_db_module_patcher.DB_PATH = _tmp_db  # type: ignore

import database
database.DB_PATH = _tmp_db  # type: ignore
database.initialize_database()

from document_loader import (
    _detect_section,
    get_adaptive_chunk_params,
    enrich_metadata,
    create_parent_child_chunks,
    chunk_documents,
)
from langchain_core.documents import Document

# ── Synthetic data helpers ────────────────────────────────────────────────────

SAMPLE_TEXT_SHORT = "This is an abstract. " * 20
SAMPLE_TEXT_MEDIUM = "Introduction to research methodology. " * 200
SAMPLE_TEXT_LONG = "Results show significant improvements. " * 800

SECTION_SAMPLES = [
    "Abstract: This paper presents...",
    "Introduction\nThis study aims to...",
    "Methodology\nWe conducted experiments using...",
    "Results\nThe model achieved 95% accuracy...",
    "Conclusion\nIn summary, our approach demonstrates...",
    "References\n[1] Smith et al. 2023...",
    "บทคัดย่อ: งานวิจัยนี้นำเสนอ...",
    "บทนำ การศึกษานี้มุ่งเน้น...",
    "สรุป ผลการวิจัยแสดงให้เห็น...",
    "Random body text with no section header here.",
]

def make_documents(n: int, text_size: str = "medium") -> list:
    """Create n synthetic Document objects."""
    text = {"short": SAMPLE_TEXT_SHORT, "medium": SAMPLE_TEXT_MEDIUM, "long": SAMPLE_TEXT_LONG}[text_size]
    return [
        Document(
            page_content=text,
            metadata={"page": i, "source": "benchmark.pdf"}
        )
        for i in range(n)
    ]


# ── Benchmark runners ─────────────────────────────────────────────────────────

def bench(label: str, fn, number: int = 1) -> float:
    t = timeit.timeit(fn, number=number)
    per = t / number * 1000
    print(f"  {label:<55} {per:8.3f} ms/call  (n={number})")
    return per


def run_benchmarks():
    results = {}
    print("\n" + "=" * 75)
    print("BENCHMARK: Research Workbench Core Modules")
    print("=" * 75)

    # ── 1. _detect_section ────────────────────────────────────────────────────
    print("\n[1] document_loader._detect_section (per call)")
    timings = []
    for sample in SECTION_SAMPLES:
        t = timeit.timeit(lambda s=sample: _detect_section(s), number=10000)
        timings.append(t / 10000 * 1000)
    avg = sum(timings) / len(timings)
    worst = max(timings)
    print(f"  Average per call:  {avg:.4f} ms")
    print(f"  Worst case:        {worst:.4f} ms")
    results["_detect_section (avg)"] = avg

    # ── 2. get_adaptive_chunk_params ──────────────────────────────────────────
    print("\n[2] document_loader.get_adaptive_chunk_params")
    t = bench("get_adaptive_chunk_params (100k calls)", lambda: get_adaptive_chunk_params(5000), number=100000)
    results["get_adaptive_chunk_params"] = t

    # ── 3. chunk_documents ────────────────────────────────────────────────────
    print("\n[3] document_loader.chunk_documents")
    docs_5 = make_documents(5, "medium")
    t = bench("chunk_documents (5 medium docs)", lambda: chunk_documents(docs_5), number=50)
    results["chunk_documents (5 docs)"] = t

    docs_20 = make_documents(20, "long")
    t = bench("chunk_documents (20 long docs)", lambda: chunk_documents(docs_20), number=20)
    results["chunk_documents (20 long docs)"] = t

    # ── 4. enrich_metadata ───────────────────────────────────────────────────
    print("\n[4] document_loader.enrich_metadata")
    docs_10 = make_documents(10, "medium")
    t = bench("enrich_metadata (10 docs)", lambda: enrich_metadata(make_documents(10, "medium"), "paper.pdf"), number=100)
    results["enrich_metadata (10 docs)"] = t

    # ── 5. create_parent_child_chunks ─────────────────────────────────────────
    print("\n[5] document_loader.create_parent_child_chunks")
    t = bench("create_parent_child_chunks (10 medium docs)",
              lambda: create_parent_child_chunks(make_documents(10, "medium"), "bench.pdf"), number=50)
    results["create_parent_child_chunks (10 docs)"] = t

    t = bench("create_parent_child_chunks (50 long docs)",
              lambda: create_parent_child_chunks(make_documents(50, "long"), "bench.pdf"), number=10)
    results["create_parent_child_chunks (50 long docs)"] = t

    # ── 6. database.save_note ─────────────────────────────────────────────────
    print("\n[6] database.save_note (individual inserts)")
    t = bench("save_note (100 calls)", lambda: database.save_note("Title", "Content " * 50), number=100)
    results["database.save_note"] = t

    # ── 7. database.load_all_notes ────────────────────────────────────────────
    print("\n[7] database.load_all_notes")
    t = bench("load_all_notes (after 100 rows)", lambda: database.load_all_notes(), number=200)
    results["database.load_all_notes"] = t

    # ── 8. database.save_parent_chunk (individual, N separate connections) ───
    print("\n[8] database.save_parent_chunk (individual calls — N separate connections)")
    N = 100
    ids = [f"parent_{uuid.uuid4().hex[:12]}" for _ in range(N)]
    content = SAMPLE_TEXT_MEDIUM

    start = time.perf_counter()
    for pid in ids:
        database.save_parent_chunk(pid, content, "bench.pdf", 0, "body")
    elapsed = (time.perf_counter() - start) * 1000
    per = elapsed / N
    print(f"  save_parent_chunk × {N} (sequential):            {elapsed:8.1f} ms total  ({per:.3f} ms/call)")
    results["database.save_parent_chunk (per call)"] = per

    # ── 8b. database.save_parent_chunks_batch ──────────────────────────────
    print("\n[8b] database.save_parent_chunks_batch (single transaction)")
    batch_ids = [f"batch_{uuid.uuid4().hex[:12]}" for _ in range(N)]
    batch_records = [{'id': pid, 'content': content, 'source_file': 'bench.pdf',
                      'page_number': 0, 'section': 'body'} for pid in batch_ids]
    t = bench(f"save_parent_chunks_batch ({N} records)", lambda: database.save_parent_chunks_batch(batch_records), number=50)
    results["database.save_parent_chunks_batch"] = t

    # ── 9. database.get_parent_chunks_batch ───────────────────────────────────
    print("\n[9] database.get_parent_chunks_batch")
    t = bench(f"get_parent_chunks_batch ({N} ids)", lambda: database.get_parent_chunks_batch(ids), number=200)
    results["database.get_parent_chunks_batch"] = t

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("SUMMARY — sorted by ms/call (worst first):")
    print("=" * 75)
    for label, ms in sorted(results.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * min(int(ms * 2), 50)
        print(f"  {label:<55} {ms:8.3f} ms  {bar}")

    worst_label = max(results, key=results.__getitem__)
    print(f"\n  → Worst performer: '{worst_label}' at {results[worst_label]:.3f} ms/call")
    return worst_label, results


if __name__ == "__main__":
    try:
        worst, results = run_benchmarks()
    finally:
        shutil.rmtree(_tmp_dir, ignore_errors=True)
