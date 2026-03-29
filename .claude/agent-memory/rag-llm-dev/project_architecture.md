---
name: Project Architecture
description: Core stack, module roles, and key implementation details for WijaiWai RAG platform
type: project
---

WijaiWai is a Thai/English AI research workbench using Pinecone as the vector store (NOT ChromaDB — migrated). Key facts:

- Vector DB: Pinecone serverless, namespace = Google user ID, index name from PINECONE_INDEX_NAME env var
- Embeddings: Pinecone Inference API, model `multilingual-e5-large` (dim=1024, Thai-capable)
- LLM: OpenThaiGPT at `http://thaillm.or.th/api/openthaigpt/v1/chat/completions`, auth header `apikey:`, model `/model`
- Frontend: Streamlit 3-panel (sidebar, center editor, right chat)
- Persistent storage: SQLite at `./Database/research_notes.db` (tables: research_notes, documents, parent_chunks, web_pages)

Module roles:
- `vector_store.py`: Pinecone client singleton (@st.cache_resource), upsert, retrieve_unified, delete
- `generator.py`: _call_api, generate_answer, generate_answer_stream (streaming), is_small_talk (routing)
- `app.py`: Streamlit UI, session state, 3-panel layout, chat rendering
- `document_loader.py`: PDF/TXT/DOCX loading, parent-child chunking, rich metadata
- `database.py`: SQLite CRUD
- `rag_pipeline.py`: LEGACY — do not modify or use

**Why:** Do not confuse with ChromaDB — previous task descriptions mentioned ChromaDB but the actual codebase uses Pinecone throughout.

**How to apply:** When touching vector store code, always use Pinecone patterns (namespace, top_k, filter dict with $eq/$and). Never reference ChromaDB APIs.
