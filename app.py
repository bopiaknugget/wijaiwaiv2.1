"""
Research Workbench — AI-Powered RAG with Text Notes
3-panel layout: Sidebar (Docs + Notes) | Center (Research Workbench) | Right (Assistant chat)

Key improvements in this version:
- Unified single ChromaDB collection separated by metadata (source_type)
- Advanced RAG: rich metadata, parent-child chunking, summary embeddings
- Adaptive chunk sizing based on content length
- MMR-based retrieval with parent expansion for full-context answers
- @st.cache_resource on embeddings; @st.cache_data on note loading
"""

import gc
import json
import os
import re
import shutil
import tempfile
import uuid
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import database
from document_loader import (
    load_document, chunk_documents,
    enrich_metadata, create_parent_child_chunks, create_summary_documents,
)
from generator import generate_answer, generate_selection_edit
from vector_store import (
    initialize_embeddings,
    get_or_create_vector_store,
    ingest_documents,
    ingest_note,
    retrieve_unified,
    retrieve_from_both_stores,
    UNIFIED_DB_PATH,
    UNIFIED_COLLECTION,
)


_THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)
WORK_DIR = os.path.join(os.path.dirname(__file__), "your_work")


# ── File helpers ───────────────────────────────────────────────────────────────

def _ensure_work_dir():
    os.makedirs(WORK_DIR, exist_ok=True)


def save_work_to_file(title: str, content: str) -> str:
    return save_work_to_file_with_name(title, title, content)


def save_work_to_file_with_name(name: str, title: str, content: str) -> str:
    _ensure_work_dir()
    from datetime import datetime
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", name.strip())[:60]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{timestamp}.txt"
    filepath = os.path.join(WORK_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"TITLE: {title}\n---\n{content}")
    return filepath


def load_work_from_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    if raw.startswith("TITLE: ") and "\n---\n" in raw:
        header, _, content = raw.partition("\n---\n")
        title = header[len("TITLE: "):]
    else:
        title = os.path.splitext(os.path.basename(filepath))[0]
        content = raw
    return title, content


def overwrite_work_file(filepath: str, title: str, content: str):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"TITLE: {title}\n---\n{content}")


def list_work_files():
    _ensure_work_dir()
    files = [f for f in os.listdir(WORK_DIR) if f.endswith(".txt")]
    files.sort(reverse=True)
    return [(f, os.path.join(WORK_DIR, f)) for f in files]


# ── Think-tag helpers ──────────────────────────────────────────────────────────

def parse_think_content(text: str):
    thinks = _THINK_PATTERN.findall(text)
    answer = _THINK_PATTERN.sub('', text).strip()
    think_text = '\n\n'.join(t.strip() for t in thinks) if thinks else ''
    return think_text, answer


def display_assistant_message(content: str):
    think_text, answer = parse_think_content(content)
    if think_text:
        with st.expander("💭 ความคิด (Thinking)", expanded=False):
            st.markdown(f'<div class="think-block">{think_text}</div>',
                        unsafe_allow_html=True)
    st.write(answer)


# ── Unified vector-store helpers ──────────────────────────────────────────────

def _load_unified_vector_store(embeddings):
    """Load (or initialise) the persistent unified ChromaDB."""
    return get_or_create_vector_store(
        db_path=UNIFIED_DB_PATH,
        chunked_documents=None,
        embeddings=embeddings,
        collection_name=UNIFIED_COLLECTION,
    )


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="🔬 Research Workbench",
        page_icon="📚",
        layout="wide"
    )

    # ── Session state defaults ─────────────────────────────────────────────────
    defaults = {
        "unified_vector_store": None,   # Single unified ChromaDB for docs + notes
        "processed_docs": [],           # [{"name": str, "chunks": int}]
        "messages": [],
        "total_tokens": 0,
        "total_cost_thb": 0.0,
        "note_title_val": "",
        "note_content_val": "",
        "work_title_val": "",
        "work_content_val": "",
        "work_current_file": None,
        "work_load_select": None,
        "work_save_dialog": None,
        "work_save_dialog_name": "",
        "work_import_open": False,
        "_unified_store_initialised": False,
        "_app_initialized": False,
        "_research_mode": False,
        "ai_edit_undo_stack": [],   # list of editor content snapshots (before AI edits)
        "ai_edit_redo_stack": [],   # list of editor content snapshots (after undone AI edits)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Loading screen (first run only) ───────────────────────────────────────
    if not st.session_state._app_initialized:
        # ── Splash screen: Logo + App name ────────────────────────────────
        splash = st.empty()
        with splash.container():
            st.markdown("""
            <style>
            @keyframes pulseGlow {
                0%, 100% { opacity: 1; transform: scale(1); filter: drop-shadow(0 0 0px rgba(102,126,234,0)); }
                50% { opacity: 0.6; transform: scale(1.08); filter: drop-shadow(0 0 18px rgba(102,126,234,0.5)); }
            }
            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(24px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .splash-screen {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 92vh;
                text-align: center;
                background: linear-gradient(160deg, #f8faff 0%, #f0f4ff 40%, #f5f0ff 100%);
                border-radius: 16px;
            }
            .splash-logo {
                font-size: 6rem;
                animation: pulseGlow 2s ease-in-out infinite;
                margin-bottom: 1.2rem;
            }
            .splash-title {
                font-size: 2.4rem;
                font-weight: 800;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: fadeInUp 1s ease-out 0.3s both;
            }
            </style>
            <div class="splash-screen">
                <div class="splash-logo">🔬</div>
                <div class="splash-title">Research Workbench</div>
            </div>
            """, unsafe_allow_html=True)

        import time
        time.sleep(2.5)
        splash.empty()

        # ── Loading progress screen ──────────────────────────────────────
        loading = st.empty()
        with loading.container():
            st.markdown("""
            <style>
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.4; }
            }
            .loading-screen {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 80vh;
                text-align: center;
            }
            .loading-icon {
                font-size: 4rem;
                animation: pulse 2s ease-in-out infinite;
                margin-bottom: 1rem;
            }
            .loading-title {
                font-size: 2rem;
                font-weight: 700;
                color: #1f2937;
                margin-bottom: 0.5rem;
            }
            .loading-subtitle {
                font-size: 1rem;
                color: #6b7280;
                margin-bottom: 2rem;
            }
            .loading-step {
                font-size: 0.95rem;
                color: #374151;
                margin: 0.3rem 0;
            }
            .loading-step .done { color: #10b981; }
            .loading-step .working { color: #3b82f6; animation: pulse 1.5s ease-in-out infinite; }
            </style>
            <div class="loading-screen">
                <div class="loading-icon">🔬</div>
                <div class="loading-title">Research Workbench</div>
                <div class="loading-subtitle">กำลังเตรียมระบบ...</div>
            </div>
            """, unsafe_allow_html=True)

            progress = st.progress(0, text="เริ่มต้นระบบ...")

            # Step 1: Load embeddings model (~400MB)
            progress.progress(15, text="📦 กำลังโหลด Embedding Model...")
            embeddings = initialize_embeddings()
            progress.progress(60, text="✅ โหลด Embedding Model สำเร็จ")

            # Step 2: Load unified vector store
            progress.progress(70, text="📝 กำลังโหลดฐานข้อมูล Vector Store...")
            if not st.session_state._unified_store_initialised:
                if os.path.exists(UNIFIED_DB_PATH):
                    try:
                        st.session_state.unified_vector_store = _load_unified_vector_store(embeddings)
                    except Exception:
                        st.session_state.unified_vector_store = None
                st.session_state._unified_store_initialised = True
            progress.progress(90, text="✅ โหลดฐานข้อมูล Vector Store สำเร็จ")

            # Step 3: Finalize
            progress.progress(100, text="✅ พร้อมใช้งาน!")

            import time
            time.sleep(0.8)

        # Clear loading screen and mark as initialized
        loading.empty()
        st.session_state._app_initialized = True
        st.session_state._cached_embeddings = embeddings
        st.rerun()

    # ── App already initialized — retrieve cached embeddings ──────────────────
    embeddings = st.session_state.get("_cached_embeddings")
    if embeddings is None:
        embeddings = initialize_embeddings()
        st.session_state._cached_embeddings = embeddings

    # Apply pending editor content BEFORE any widget is rendered
    for widget_key, pending_key in [
        ("work_title_input", "_pending_work_title"),
        ("work_content_input", "_pending_work_content"),
    ]:
        if pending_key in st.session_state:
            st.session_state[widget_key] = st.session_state.pop(pending_key)

    # ── Styles ────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .think-block {
        background-color: #f0f4f8;
        border-left: 4px solid #90a4ae;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 8px;
        color: #546e7a;
        font-style: italic;
        font-size: 0.9em;
        white-space: pre-wrap;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1f2937;
    }

    /* ── Sidebar: push content down to align with main panel ── */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        margin-top: 0;
        margin-bottom: 0.4rem;
    }

    /* ── Compact buttons: reduce gap between button rows ── */
    div[data-testid="stColumns"] + div[data-testid="stColumns"] {
        margin-top: -0.5rem;
    }

    /* ── Editor toolbar buttons: smaller, uniform look ── */
    .stButton > button {
        font-size: 0.85rem;
        padding-top: 0.35rem;
        padding-bottom: 0.35rem;
    }
    .stDownloadButton > button {
        font-size: 0.85rem;
        padding-top: 0.35rem;
        padding-bottom: 0.35rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # ============================================================================
    # SIDEBAR — Documents + Notes tabs
    # ============================================================================
    with st.sidebar:
        sidebar_tab_docs, sidebar_tab_notes = st.tabs(["📄 Documents", "📝 Notes"])

        # ── Tab 1: Documents ──────────────────────────────────────────────────
        with sidebar_tab_docs:
            st.markdown("**Upload Documents**")
            uploaded_files = st.file_uploader(
                "Select files (PDF, TXT, DOCX, DOC)",
                type=["pdf", "txt", "docx", "doc"],
                accept_multiple_files=True,
                key="file_uploader_sidebar"
            )

            if uploaded_files:
                st.caption(f"{len(uploaded_files)} file(s) selected")
                if st.button("🔄 Process Documents", type="primary",
                             key="process_doc_btn", use_container_width=True):
                    all_child_chunks = []
                    all_parent_records = []
                    all_summary_docs = []
                    new_doc_entries = []

                    with st.spinner(f"Processing {len(uploaded_files)} file(s) with Advanced RAG..."):
                        for uploaded_file in uploaded_files:
                            try:
                                ext = os.path.splitext(uploaded_file.name)[1].lower()
                                with tempfile.NamedTemporaryFile(
                                    delete=False, suffix=ext
                                ) as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    tmp_path = tmp_file.name

                                # Load and enrich with rich metadata
                                documents = load_document(tmp_path)
                                documents = enrich_metadata(
                                    documents, uploaded_file.name,
                                    source_type="document"
                                )

                                # Parent-Child Chunking (adaptive sizing)
                                child_chunks, parent_records = create_parent_child_chunks(
                                    documents, uploaded_file.name,
                                    source_type="document"
                                )

                                # Summary Embedding (extractive fallback)
                                summary_docs = create_summary_documents(
                                    documents, uploaded_file.name
                                )

                                all_child_chunks.extend(child_chunks)
                                all_parent_records.extend(parent_records)
                                all_summary_docs.extend(summary_docs)
                                new_doc_entries.append({
                                    "name": uploaded_file.name,
                                    "chunks": len(child_chunks)
                                })
                                os.unlink(tmp_path)
                            except Exception as e:
                                st.error(f"❌ {uploaded_file.name}: {str(e)}")

                        if all_child_chunks:
                            try:
                                # Ensure unified store exists
                                if st.session_state.unified_vector_store is None:
                                    st.session_state.unified_vector_store = get_or_create_vector_store(
                                        db_path=UNIFIED_DB_PATH,
                                        embeddings=embeddings,
                                        collection_name=UNIFIED_COLLECTION,
                                    )

                                # Ingest everything into unified collection
                                ingest_documents(
                                    st.session_state.unified_vector_store,
                                    all_child_chunks,
                                    all_parent_records,
                                    all_summary_docs,
                                )

                                st.session_state.processed_docs.extend(new_doc_entries)
                                st.session_state.messages = []
                                st.session_state.total_tokens = 0
                                st.session_state.total_cost_thb = 0.0
                                st.success(
                                    f"✅ {len(new_doc_entries)}/{len(uploaded_files)} "
                                    "file(s) ready! (Advanced RAG)"
                                )
                            except Exception as e:
                                st.error(f"❌ Vector store error: {str(e)}")

            # ── Processed documents list with Delete buttons ───────────────
            if st.session_state.processed_docs:
                st.divider()
                st.caption(f"{len(st.session_state.processed_docs)} document(s) loaded")
                for doc_entry in list(st.session_state.processed_docs):
                    col_info, col_del = st.columns([5, 1])
                    with col_info:
                        st.markdown(f"📄 **{doc_entry['name']}**")
                        st.caption(f"{doc_entry['chunks']} chunks")
                    with col_del:
                        if st.button("🗑️", key=f"del_doc_{doc_entry['name']}",
                                     help="Delete this document"):
                            # Remove chunks from unified ChromaDB by doc_name metadata
                            if st.session_state.unified_vector_store is not None:
                                try:
                                    st.session_state.unified_vector_store \
                                        ._collection.delete(
                                            where={"doc_name": doc_entry["name"]}
                                        )
                                except Exception as e:
                                    st.error(f"VectorDB delete error: {e}")
                            # Remove parent chunks from SQLite
                            database.delete_parent_chunks_by_source(doc_entry["name"])
                            # Remove from tracking list immediately
                            st.session_state.processed_docs = [
                                d for d in st.session_state.processed_docs
                                if d["name"] != doc_entry["name"]
                            ]
                            st.rerun()

        # ── Tab 2: Notes ──────────────────────────────────────────────────────
        with sidebar_tab_notes:
            note_title_input = st.text_input(
                "Title",
                value=st.session_state.note_title_val,
                placeholder="Note title...",
                key="note_title_input_sidebar"
            )
            note_content_input = st.text_area(
                "Content",
                value=st.session_state.note_content_val,
                placeholder="Write your research notes here...",
                height=160,
                key="note_content_input_sidebar"
            )
            save_note_clicked = st.button(
                "💾 Save Note", type="primary",
                key="save_note_btn_sidebar", use_container_width=True
            )

            if save_note_clicked:
                if note_title_input.strip() and note_content_input.strip():
                    with st.spinner("Saving note..."):
                        note_id = database.save_note(
                            note_title_input, note_content_input
                        )
                        # Embed into the unified collection with source_type='note'
                        st.session_state.unified_vector_store = ingest_note(
                            st.session_state.unified_vector_store,
                            note_id, note_title_input, note_content_input,
                            embeddings,
                        )
                    st.success(f"✅ Saved '{note_title_input}' (ID: {note_id})")
                    st.session_state.note_title_val = ""
                    st.session_state.note_content_val = ""
                    st.rerun()
                else:
                    st.warning("⚠️ Please enter both title and content.")

            st.divider()

            # ── Saved notes list with Delete buttons (same style as docs) ────
            notes = database.load_all_notes()
            if notes:
                st.caption(f"{len(notes)} note(s) saved")
                for note in notes:
                    col_info, col_del = st.columns([5, 1])
                    with col_info:
                        st.markdown(f"📝 **{note['title']}**")
                        st.caption(note['timestamp'])
                    with col_del:
                        if st.button("🗑️", key=f"del_note_{note['id']}",
                                     help="Delete this note"):
                            # 1. Remove from SQLite
                            database.delete_note_by_id(note['id'])
                            # 2. Remove from unified ChromaDB
                            if st.session_state.unified_vector_store is not None:
                                try:
                                    st.session_state.unified_vector_store \
                                        ._collection.delete(
                                            where={"note_id": note['id']}
                                        )
                                except Exception as e:
                                    st.error(f"VectorDB delete error: {e}")
                            st.rerun()
                    # Content preview in a collapsed expander below the row
                    with st.expander("ดูเนื้อหา", expanded=False):
                        st.text_area(
                            "",
                            value=note['content'],
                            height=90,
                            disabled=True,
                            key=f"note_view_{note['id']}"
                        )
            else:
                st.info("No notes saved yet.")


    # ============================================================================
    # MAIN CONTENT: Center (Research Workbench) | Right (Assistant)
    # ============================================================================
    col_center, col_right = st.columns([3, 2], gap="large")

    # ── Center: Research Workbench ─────────────────────────────────────────────────────
    with col_center:
        st.markdown("""
        <div style="font-size:1.35rem;font-weight:700;color:#1f2937;padding:0.25rem 0 0.4rem 0;">
            🔬 Research Workbench
        </div>""", unsafe_allow_html=True)

        work_title = st.text_input(
            "Title",
            value=st.session_state.work_title_val,
            placeholder="Enter a title for your research...",
            key="work_title_input"
        )
        work_content = st.text_area(
            "Content",
            value=st.session_state.work_content_val,
            placeholder="Start writing your research here...",
            height=400,
            key="work_content_input"
        )

        current_file = st.session_state.get("work_current_file")
        if current_file:
            st.caption(f"📄 `{os.path.basename(current_file)}`")

        # ── Row 1: File actions ────────────────────────────────────────────
        c_save, c_saveas, c_load, c_export, c_import = st.columns(5)
        with c_save:
            save_clicked = st.button("💾 Save", type="primary",
                                     key="save_work_btn", use_container_width=True)
        with c_saveas:
            save_as_clicked = st.button("📑 Save As",
                                        key="save_as_work_btn", use_container_width=True)
        with c_load:
            load_work_clicked = st.button("📂 Load",
                                          key="load_work_btn", use_container_width=True)
        with c_export:
            export_data = (
                f"TITLE: {work_title}\n---\n{work_content}"
                if (work_title.strip() and work_content.strip()) else ""
            )
            export_fname = (
                re.sub(r'[\\/*?:"<>|]', "_", work_title.strip())[:60] or "work"
            ) + ".txt"
            st.download_button(
                "📤 Export",
                data=export_data.encode("utf-8"),
                file_name=export_fname,
                mime="text/plain",
                key="export_work_btn",
                use_container_width=True,
                disabled=not export_data,
            )
        with c_import:
            import_clicked = st.button("📥 Import",
                                       key="import_work_btn", use_container_width=True)

        # ── Row 2: Edit actions ───────────────────────────────────────────
        c_undo, c_redo, c_clear = st.columns(3)
        with c_undo:
            undo_clicked = st.button(
                "↩️ Undo",
                key="ai_undo_btn",
                use_container_width=True,
                disabled=not st.session_state.ai_edit_undo_stack,
                help="Undo the last AI edit",
            )
        with c_redo:
            redo_clicked = st.button(
                "↪️ Redo",
                key="ai_redo_btn",
                use_container_width=True,
                disabled=not st.session_state.ai_edit_redo_stack,
                help="Redo the last undone AI edit",
            )
        with c_clear:
            clear_editor_clicked = st.button("🗑️ Clear",
                                             key="clear_editor_btn", use_container_width=True)

        # ── Button logic ───────────────────────────────────────────────────────
        if save_clicked:
            if not work_title.strip() or not work_content.strip():
                st.warning("⚠️ Please enter both a title and content.")
            elif current_file and os.path.exists(current_file):
                overwrite_work_file(current_file, work_title, work_content)
                st.success(f"✅ Saved → `{os.path.basename(current_file)}`")
            else:
                st.session_state.work_save_dialog = "save"
                st.session_state.work_save_dialog_name = work_title
                st.session_state.work_load_select = False
                st.session_state.work_import_open = False
                st.rerun()

        if save_as_clicked:
            if not work_title.strip() or not work_content.strip():
                st.warning("⚠️ Please enter both a title and content.")
            else:
                st.session_state.work_save_dialog = "save_as"
                st.session_state.work_save_dialog_name = work_title
                st.session_state.work_load_select = False
                st.session_state.work_import_open = False
                st.rerun()

        if load_work_clicked:
            st.session_state.work_load_select = True
            st.session_state.work_save_dialog = None
            st.session_state.work_import_open = False

        if import_clicked:
            st.session_state.work_import_open = True
            st.session_state.work_load_select = False
            st.session_state.work_save_dialog = None

        if clear_editor_clicked:
            st.session_state["_pending_work_title"] = ""
            st.session_state["_pending_work_content"] = ""
            st.session_state.work_title_val = ""
            st.session_state.work_content_val = ""
            st.session_state.work_current_file = None
            st.session_state.ai_edit_undo_stack = []
            st.session_state.ai_edit_redo_stack = []
            st.rerun()

        if undo_clicked and st.session_state.ai_edit_undo_stack:
            st.session_state.ai_edit_redo_stack.append(work_content)
            restored = st.session_state.ai_edit_undo_stack.pop()
            st.session_state["_pending_work_content"] = restored
            st.session_state.work_content_val = restored
            st.rerun()

        if redo_clicked and st.session_state.ai_edit_redo_stack:
            st.session_state.ai_edit_undo_stack.append(work_content)
            restored = st.session_state.ai_edit_redo_stack.pop()
            st.session_state["_pending_work_content"] = restored
            st.session_state.work_content_val = restored
            st.rerun()

        # ── Save dialog ────────────────────────────────────────────────────────
        if st.session_state.get("work_save_dialog"):
            st.markdown("**💾 ตั้งชื่อไฟล์**")
            dialog_name = st.text_input(
                "ชื่อไฟล์",
                value=st.session_state.work_save_dialog_name,
                placeholder="ระบุชื่อไฟล์...",
                key="save_dialog_name_input"
            )
            col_confirm_s, col_cancel_s = st.columns([1, 1])
            with col_confirm_s:
                confirm_save = st.button("✅ บันทึก",
                                         key="confirm_save_dialog_btn",
                                         use_container_width=True)
            with col_cancel_s:
                cancel_save = st.button("❌ ยกเลิก",
                                        key="cancel_save_dialog_btn",
                                        use_container_width=True)
            if confirm_save:
                if dialog_name.strip():
                    filepath = save_work_to_file_with_name(
                        dialog_name, work_title, work_content
                    )
                    st.session_state.work_current_file = filepath
                    st.session_state.work_save_dialog = None
                    st.success(f"✅ Saved → `{os.path.basename(filepath)}`")
                    st.rerun()
                else:
                    st.warning("⚠️ กรุณาระบุชื่อไฟล์")
            if cancel_save:
                st.session_state.work_save_dialog = None
                st.rerun()

        # ── Load panel ─────────────────────────────────────────────────────────
        if st.session_state.get("work_load_select"):
            work_files = list_work_files()
            if work_files:
                display_names = [name for name, _ in work_files]
                selected = st.selectbox("Select a file to load",
                                        options=display_names,
                                        key="work_file_selectbox")
                if st.button("✅ Load into editor", key="confirm_load_btn"):
                    selected_path = dict(work_files)[selected]
                    loaded_title, loaded_content = load_work_from_file(selected_path)
                    st.session_state["_pending_work_title"] = loaded_title
                    st.session_state["_pending_work_content"] = loaded_content
                    st.session_state.work_current_file = selected_path
                    st.session_state.work_load_select = False
                    st.rerun()
            else:
                st.info("No saved work files found.")

        # ── Import panel ───────────────────────────────────────────────────────
        if st.session_state.get("work_import_open"):
            import_file = st.file_uploader(
                "เลือกไฟล์ที่ต้องการ import",
                type=["txt", "docx", "doc"],
                key="import_file_uploader"
            )
            if import_file is not None:
                try:
                    ext = os.path.splitext(import_file.name)[1].lower()
                    if ext == ".txt":
                        imported_content = import_file.read().decode("utf-8")
                    elif ext in (".docx", ".doc"):
                        import docx2txt, io
                        imported_content = docx2txt.process(
                            io.BytesIO(import_file.read())
                        )
                    else:
                        imported_content = import_file.read().decode(
                            "utf-8", errors="replace"
                        )
                    st.session_state["_pending_work_title"] = (
                        os.path.splitext(import_file.name)[0]
                    )
                    st.session_state["_pending_work_content"] = imported_content
                    st.session_state.work_current_file = None
                    st.session_state.work_import_open = False
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ ไม่สามารถ import ไฟล์ได้: {str(e)}")
            if st.button("❌ ยกเลิก import", key="cancel_import_btn"):
                st.session_state.work_import_open = False
                st.rerun()

        st.divider()

        # ── Session usage stats ────────────────────────────────────────────────
        with st.expander("📈 Session Usage", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
            with c2:
                st.metric("Total Cost", f"฿{st.session_state.total_cost_thb:.4f}")
            with c3:
                avg = (
                    (st.session_state.total_cost_thb
                     / st.session_state.total_tokens * 1_000_000)
                    if st.session_state.total_tokens > 0 else 0
                )
                st.metric("Per 1M Tokens", f"฿{avg:.4f}")

    # ── Right: Assistant Chat ─────────────────────────────────────────────────
    with col_right:
        st.markdown("""
        <div style="font-size:1.35rem;font-weight:700;color:#1f2937;padding:0.25rem 0 0.4rem 0;">
            🤖 Assistant
        </div>""", unsafe_allow_html=True)

        unified_store = st.session_state.unified_vector_store
        has_context = unified_store is not None

        # ── Research mode indicator ───────────────────────────────────────
        if st.session_state._research_mode:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
                border: 2px solid #3b82f6;
                border-radius: 10px;
                padding: 10px 16px;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            ">
                <div>
                    <span style="font-size: 1.1rem; font-weight: 700; color: #1e40af;">
                        🔬 Research Mode
                    </span>
                    <span style="font-size: 0.85rem; color: #6b7280; margin-left: 8px;">
                        คำตอบจะละเอียดขึ้นและแสดงใน Research Workbench
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("✕ ออกจาก Research Mode", key="exit_research_mode_btn",
                         type="secondary", use_container_width=True):
                st.session_state._research_mode = False
                st.rerun()

        chat_container = st.container(height=480)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant":
                        if message.get("action") == "research":
                            st.caption("🔬 Research — ผลลัพธ์อยู่ใน Research Workbench")
                        elif message.get("action") == "edit":
                            st.caption("✏️ แก้ไขเอกสารแล้ว")
                        display_assistant_message(message["content"])
                        if "tokens" in message:
                            st.caption(
                                f"⏱️ {message['tokens']} tokens "
                                f"| ฿{message['cost_thb']:.4f}"
                            )
                        if "sources" in message:
                            with st.expander("📚 Sources"):
                                for i, doc in enumerate(message["sources"], 1):
                                    src_type = doc.metadata.get("source", "doc")
                                    label = (
                                        "📝 Note" if src_type == "research_note"
                                        else f"📄 Doc {i}"
                                    )
                                    st.markdown(f"**{label}:**")
                                    preview = doc.page_content
                                    st.text(
                                        preview[:300] + "..."
                                        if len(preview) > 300 else preview
                                    )
                    else:
                        # Show research badge on user message if it was a /research query
                        if message.get("research"):
                            st.caption("🔬 Research Mode")
                        st.write(message["content"])
            if not st.session_state.messages and not has_context:
                st.info(
                    "📄 Upload documents or save a note for RAG-based answers.\n\n"
                    "💡 พิมพ์ `/research` ก่อนคำถามเพื่อเข้าสู่โหมดค้นคว้าเชิงลึก"
                )

        # ── Tab-autocomplete JS: /r → /research ──────────────────────────
        import streamlit.components.v1 as components
        components.html("""
        <script>
        // ── Hide Streamlit widget default-value warnings ──
        const _hideWidgetWarnings = () => {
            const alerts = parent.document.querySelectorAll('[data-testid="stAlert"], .stAlert');
            alerts.forEach(el => {
                if (el.textContent.includes('was created with a default value')) {
                    el.style.display = 'none';
                }
            });
        };
        setInterval(_hideWidgetWarnings, 300);

        // ── Autocomplete: /r + Tab → /research ──
        const _setupAutocomplete = () => {
            const textarea = parent.document.querySelector(
                'textarea[data-testid="stChatInputTextArea"]'
            );
            if (!textarea || textarea._rcAutoComplete) return;
            textarea._rcAutoComplete = true;
            textarea.addEventListener('keydown', function(e) {
                if (e.key !== 'Tab') return;
                const val = this.value;
                if (val.match(/^\\/r(?!esearch)/) ) {
                    e.preventDefault();
                    const rest = val.replace(/^\\/r\\s*/, '');
                    const nset = Object.getOwnPropertyDescriptor(
                        window.HTMLTextAreaElement.prototype, 'value'
                    ).set;
                    nset.call(this, '/research ' + rest);
                    this.dispatchEvent(new Event('input', {bubbles: true}));
                }
            });
        };
        const _iv = setInterval(() => {
            const ta = parent.document.querySelector(
                'textarea[data-testid="stChatInputTextArea"]'
            );
            if (ta) { clearInterval(_iv); _setupAutocomplete(); }
        }, 300);
        setTimeout(() => clearInterval(_iv), 10000);

        // ── Content Edit: right-click on selected text → floating chatbox ──
        const _removeEditOverlay = () => {
            const el = parent.document.getElementById('__ceOverlay');
            if (el) el.remove();
        };

        const _showEditOverlay = (x, y, selectedText) => {
            _removeEditOverlay();
            const overlay = parent.document.createElement('div');
            overlay.id = '__ceOverlay';
            overlay.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:999999;background:rgba(0,0,0,0.12);';

            const boxW = 350;
            let bx = Math.min(x + 4, parent.innerWidth - boxW - 16);
            let by = Math.min(y + 4, parent.innerHeight - 260);
            bx = Math.max(8, bx); by = Math.max(8, by);

            const preview = selectedText.length > 120
                ? selectedText.substring(0, 120).replace(/</g, '&lt;') + '...'
                : selectedText.replace(/</g, '&lt;');

            overlay.innerHTML = `
            <div style="position:fixed;left:${bx}px;top:${by}px;width:${boxW}px;background:#fff;border-radius:12px;box-shadow:0 8px 32px rgba(0,0,0,0.2);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;overflow:hidden;border:1px solid #e0e0e0;">
                <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:10px 16px;font-weight:600;font-size:14px;display:flex;justify-content:space-between;align-items:center;">
                    <span>&#9999;&#65039; Edit</span>
                    <span id="__ceClose" style="cursor:pointer;font-size:18px;opacity:0.8;">&#10005;</span>
                </div>
                <div style="padding:12px 16px;">
                    <div style="background:#f8f9fa;border-radius:6px;padding:8px 10px;margin-bottom:10px;font-size:12px;color:#555;max-height:60px;overflow-y:auto;border:1px solid #eee;white-space:pre-wrap;word-break:break-word;">${preview}</div>
                    <input type="text" id="__ceInput" placeholder="อธิบายว่าต้องการแก้ไขอย่างไร..."
                        style="width:100%;box-sizing:border-box;padding:9px 12px;border:1.5px solid #ddd;border-radius:8px;font-size:13px;outline:none;"
                    />
                    <div style="display:flex;gap:8px;margin-top:10px;">
                        <button id="__ceSubmit" style="flex:1;padding:9px;border:none;border-radius:8px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;font-weight:600;cursor:pointer;font-size:13px;">แก้ไข</button>
                        <button id="__ceCancel" style="flex:1;padding:9px;border:1.5px solid #ddd;border-radius:8px;background:#fff;color:#555;cursor:pointer;font-size:13px;">ยกเลิก</button>
                    </div>
                </div>
            </div>`;

            parent.document.body.appendChild(overlay);
            setTimeout(() => {
                const inp = parent.document.getElementById('__ceInput');
                if (inp) inp.focus();
            }, 50);

            // Submit handler
            parent.document.getElementById('__ceSubmit').addEventListener('click', () => {
                const instruction = parent.document.getElementById('__ceInput').value.trim();
                if (!instruction) return;
                const cmd = '__EDIT__' + JSON.stringify({s: selectedText, i: instruction});
                const chatTA = parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
                if (chatTA) {
                    const nset = Object.getOwnPropertyDescriptor(
                        window.HTMLTextAreaElement.prototype, 'value'
                    ).set;
                    nset.call(chatTA, cmd);
                    chatTA.dispatchEvent(new Event('input', {bubbles: true}));
                    setTimeout(() => {
                        const btn = parent.document.querySelector('button[data-testid="stChatInputSubmitButton"]');
                        if (btn) btn.click();
                    }, 150);
                }
                _removeEditOverlay();
            });

            // Enter key submits
            parent.document.getElementById('__ceInput').addEventListener('keydown', (e) => {
                if (e.key === 'Enter') parent.document.getElementById('__ceSubmit').click();
            });

            // Cancel / close
            parent.document.getElementById('__ceCancel').addEventListener('click', _removeEditOverlay);
            parent.document.getElementById('__ceClose').addEventListener('click', _removeEditOverlay);
            overlay.addEventListener('click', (e) => { if (e.target === overlay) _removeEditOverlay(); });
        };

        const _setupContentEdit = () => {
            const textareas = parent.document.querySelectorAll('textarea');
            for (const ta of textareas) {
                if (ta.placeholder && ta.placeholder.includes('Start writing')) {
                    if (ta._contentEditAttached) return;
                    ta._contentEditAttached = true;
                    ta.addEventListener('contextmenu', function(e) {
                        const sel = this.value.substring(this.selectionStart, this.selectionEnd);
                        if (!sel.trim()) return;
                        e.preventDefault();
                        _showEditOverlay(e.clientX, e.clientY, sel);
                    });
                }
            }
        };

        const _ceIv = setInterval(() => {
            _setupContentEdit();
            const found = Array.from(parent.document.querySelectorAll('textarea')).some(
                ta => ta.placeholder && ta.placeholder.includes('Start writing') && ta._contentEditAttached
            );
            if (found) clearInterval(_ceIv);
        }, 500);
        setTimeout(() => clearInterval(_ceIv), 15000);
        </script>
        """, height=0)

        # Chat input
        prompt = st.chat_input(
            "พิมพ์ /r + Tab → /research | ถามคำถาม หรือ สั่งแก้ไขเอกสาร...",
            key="chat_input_main",
        )

        # ── Clear Chat button — below the chat input ───────────────────────
        if st.button("🗑️ Clear Chat History", type="secondary",
                     key="clear_chat_btn", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_tokens = 0
            st.session_state.total_cost_thb = 0.0
            st.rerun()

        if prompt:
            # ── Detect content edit command from right-click overlay ──────
            if prompt.startswith("__EDIT__"):
                try:
                    edit_data = json.loads(prompt[8:])
                    selected = edit_data["s"]
                    instruction = edit_data["i"]

                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"✏️ แก้ไขข้อความ: {instruction}",
                    })

                    with st.spinner("✏️ กำลังแก้ไขข้อความที่เลือก..."):
                        edited, ri, ro = generate_selection_edit(
                            selected, instruction
                        )
                        new_content = work_content.replace(selected, edited, 1)

                        st.session_state.ai_edit_undo_stack.append(work_content)
                        if len(st.session_state.ai_edit_undo_stack) > 20:
                            st.session_state.ai_edit_undo_stack.pop(0)
                        st.session_state.ai_edit_redo_stack = []
                        st.session_state["_pending_work_content"] = new_content
                        st.session_state.work_content_val = new_content

                        total_tokens_turn = ri + ro
                        cost_thb = (total_tokens_turn / 1_000_000) * 0.4 * 35
                        st.session_state.total_tokens += total_tokens_turn
                        st.session_state.total_cost_thb += cost_thb

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": (
                                f"✏️ แก้ไขข้อความที่เลือกเรียบร้อยแล้ว\n\n"
                                f"คำสั่ง: {instruction}"
                            ),
                            "tokens": total_tokens_turn,
                            "cost_thb": cost_thb,
                            "action": "edit",
                        })
                    st.rerun()
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ เกิดข้อผิดพลาดในการแก้ไข: {str(e)}"
                    })
                    st.rerun()

            # ── Detect /research prefix ───────────────────────────────────
            is_research = False
            actual_query = prompt
            if re.match(r'^/research\b\s*', prompt, re.IGNORECASE):
                is_research = True
                actual_query = re.sub(r'^/research\s*', '', prompt, flags=re.IGNORECASE).strip()
                st.session_state._research_mode = True
            elif st.session_state._research_mode:
                is_research = True

            # If /research was typed alone with no query, just toggle mode on
            if is_research and not actual_query:
                st.session_state._research_mode = True
                st.rerun()

            st.session_state.messages.append({
                "role": "user",
                "content": actual_query,
                "research": is_research,
            })

            spinner_text = "🔬 กำลังค้นคว้าเชิงลึก..." if is_research else "Analyzing..."
            with st.spinner(spinner_text):
                try:
                    chat_history = st.session_state.messages[:-1]

                    # Retrieve from unified collection (with parent expansion)
                    retrieval_k = 5 if is_research else 3
                    retrieved_docs = retrieve_unified(
                        unified_store, actual_query, k=retrieval_k,
                        expand_parents=True,
                    )

                    action, response_text, new_editor_content, input_tokens, output_tokens = (
                        generate_answer(
                            actual_query, retrieved_docs, chat_history,
                            editor_content=work_content,
                            research_mode=is_research,
                        )
                    )

                    total_tokens_turn = input_tokens + output_tokens
                    cost_thb = (total_tokens_turn / 1_000_000) * 0.4 * 35

                    st.session_state.total_tokens += total_tokens_turn
                    st.session_state.total_cost_thb += cost_thb

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": retrieved_docs,
                        "tokens": total_tokens_turn,
                        "cost_thb": cost_thb,
                        "action": action,
                    })

                    # Push new content into the editor for edit/research actions
                    if action in ("edit", "research") and new_editor_content:
                        # Snapshot current content for undo (cap stack at 20)
                        st.session_state.ai_edit_undo_stack.append(work_content)
                        if len(st.session_state.ai_edit_undo_stack) > 20:
                            st.session_state.ai_edit_undo_stack.pop(0)
                        st.session_state.ai_edit_redo_stack = []
                        st.session_state["_pending_work_content"] = new_editor_content
                        st.session_state.work_content_val = new_editor_content

                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ Error: {str(e)}"
                    })
            st.rerun()


if __name__ == "__main__":
    main()
