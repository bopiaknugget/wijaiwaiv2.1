# Agent Memory Index

- [Project Architecture](project_architecture.md) — Pinecone+OpenThaiGPT RAG stack, key module roles, SQLite schema
- [Performance Improvements](project_perf_improvements.md) — All fix_emedding.md items implemented, which files changed, design decisions
- [Benchmark Results](benchmark_results.md) — 9/9 pass with real data (ingest+retrieve+parent-child+hybrid), latency baseline, 1 cp874 bug found (2026-03-29)
- [User Isolation Security Fix](project_user_isolation.md) — How per-user data isolation works in SQLite (user_id columns, migration pattern, editor_documents table)
