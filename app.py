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
import requests
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
from generator import generate_answer, generate_selection_edit, generate_section
from reviewer import review_research
from web_scraper import scrape_url, summarize_content, generate_title, prepare_web_chunks
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
WORK_DIR = os.path.join(os.path.dirname(__file__), "user_data")


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


# ── Advisor review renderer ──────────────────────────────────────────────────

_REVIEW_TAG_STYLES = {
    "ต้องแก้ไข": {
        "bg": "#fef2f2", "border": "#fca5a5", "color": "#991b1b",
        "icon": "🔴", "label": "ต้องแก้ไข",
    },
    "ดีแล้ว": {
        "bg": "#f0fdf4", "border": "#86efac", "color": "#166534",
        "icon": "🟢", "label": "ดีแล้ว",
    },
    "คำแนะนำ": {
        "bg": "#fffbeb", "border": "#fcd34d", "color": "#92400e",
        "icon": "🟡", "label": "คำแนะนำ",
    },
}

_REVIEW_TAG_RE = re.compile(
    r'\[(ต้องแก้ไข|ดีแล้ว|คำแนะนำ)\]',
)


def _render_review_result(review_text: str):
    """Render advisor review with color-coded blocks."""
    import html as _html
    lines = review_text.split('\n')
    current_tag = None
    current_lines = []

    def _flush():
        nonlocal current_tag, current_lines
        content = _html.escape('\n'.join(current_lines).strip()).replace('\n', '<br>')
        if not content:
            current_lines = []
            return
        if current_tag and current_tag in _REVIEW_TAG_STYLES:
            s = _REVIEW_TAG_STYLES[current_tag]
            st.markdown(
                f'<div style="background:{s["bg"]};border-left:4px solid {s["border"]};'
                f'border-radius:8px;padding:10px 14px;margin:6px 0;'
                f'color:{s["color"]};font-size:0.92rem;line-height:1.6;">'
                f'<strong>{s["icon"]} [{s["label"]}]</strong> '
                f'{content}</div>',
                unsafe_allow_html=True,
            )
        else:
            # General text (overview / summary)
            st.markdown(
                f'<div style="background:#f8fafc;border-radius:8px;'
                f'padding:10px 14px;margin:6px 0;color:#334155;'
                f'font-size:0.92rem;line-height:1.6;">{content}</div>',
                unsafe_allow_html=True,
            )
        current_lines = []

    for line in lines:
        m = _REVIEW_TAG_RE.search(line)
        if m:
            _flush()
            current_tag = m.group(1)
            # Remove the tag from the line text, keep the rest
            cleaned = _REVIEW_TAG_RE.sub('', line).strip(' -—:')
            if cleaned:
                current_lines.append(cleaned)
        else:
            current_lines.append(line)

    _flush()


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
# Web Edit Dialog — แก้ไขชื่อเว็บเพจ + อัปเดต ChromaDB metadata
# ============================================================================

@st.dialog("แก้ไขชื่อ")
def _show_web_edit_dialog(web_page_id: int):
    """Pop-up สำหรับแก้ไขชื่อเว็บเพจ"""
    wp = database.get_web_page_by_id(web_page_id)
    if wp is None:
        st.error("ไม่พบข้อมูลนี้")
        if st.button("ปิด", key="web_edit_close_err"):
            del st.session_state._web_edit_id
            st.rerun()
        return

    st.caption(f"🔗 {wp['url']}")

    edit_title = st.text_input(
        "ชื่อ",
        value=wp['title'],
        key="web_edit_dialog_title"
    )

    col_save, col_cancel = st.columns(2)
    with col_save:
        save_clicked = st.button(
            "บันทึก", type="primary",
            key="web_edit_dialog_save",
            use_container_width=True
        )
    with col_cancel:
        cancel_clicked = st.button(
            "ยกเลิก",
            key="web_edit_dialog_cancel",
            use_container_width=True
        )

    if cancel_clicked:
        del st.session_state._web_edit_id
        st.rerun()

    if save_clicked:
        if not edit_title.strip():
            st.warning("กรุณาระบุชื่อ")
            return

        with st.spinner("กำลังบันทึก..."):
            try:
                # อัปเดตชื่อใน SQLite
                database.update_web_page_title(web_page_id, edit_title.strip())

                # อัปเดต paper_title ใน ChromaDB metadata
                if st.session_state.unified_vector_store is not None:
                    collection = st.session_state.unified_vector_store._collection
                    try:
                        results = collection.get(
                            where={"web_page_id": web_page_id},
                            include=["metadatas"]
                        )
                    except Exception:
                        results = collection.get(
                            where={"doc_name": wp['url']},
                            include=["metadatas"]
                        )
                    if results and results['ids']:
                        updated = []
                        for meta in results['metadatas']:
                            meta['paper_title'] = edit_title.strip()
                            updated.append(meta)
                        collection.update(
                            ids=results['ids'],
                            metadatas=updated
                        )

                del st.session_state._web_edit_id
                st.rerun()

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {str(e)}")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    from PIL import Image as _PILImage
    _logo_img = _PILImage.open(
        os.path.join(os.path.dirname(__file__), "pic", "logo.jpeg")
    )
    st.set_page_config(
        page_title="WijaiWai",
        page_icon=_logo_img,
        layout="wide"
    )

    # ── Session state defaults ─────────────────────────────────────────────────
    defaults = {
        "unified_vector_store": None,   # Single unified ChromaDB for docs + notes
        "processed_docs": [              # Restored from SQLite on startup
            {"name": d["filename"], "chunks": d["chunk_count"], "doc_id": d["id"]}
            for d in database.load_all_documents()
        ],
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
        "review_result": None,      # latest advisor review output text
        "review_expanded": True,    # whether review result is expanded
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Banner image (used by splash, loading, and top banner) ──────────────
    import base64, pathlib
    _banner_path = pathlib.Path(__file__).parent / "pic" / "banner.jpeg"
    _b64 = base64.b64encode(_banner_path.read_bytes()).decode()

    # ── Loading screen (first run only) ───────────────────────────────────────
    if not st.session_state._app_initialized:

        # ── Splash screen: Logo + App name ────────────────────────────────
        splash = st.empty()
        with splash.container():
            st.markdown(f"""
            <style>
            @keyframes fadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}
            .splash-screen {{
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 92vh;
                background: #f8faff;
                border-radius: 16px;
                animation: fadeIn 0.8s ease-out both;
            }}
            .splash-banner {{
                max-width: 100%;
                max-height: 80vh;
                object-fit: contain;
                border-radius: 12px;
            }}
            </style>
            <div class="splash-screen">
                <img class="splash-banner" src="data:image/jpeg;base64,{_b64}" />
            </div>
            """, unsafe_allow_html=True)

        import time
        time.sleep(2.5)
        splash.empty()

        # ── Loading progress screen ──────────────────────────────────────
        loading = st.empty()
        with loading.container():
            st.markdown(f"""
            <style>
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.4; }}
            }}
            .loading-screen {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 80vh;
                text-align: center;
            }}
            .loading-subtitle {{
                font-size: 1rem;
                color: #6b7280;
                margin-bottom: 2rem;
            }}
            </style>
            <div class="loading-screen">
                <img src="data:image/jpeg;base64,{_b64}" style="max-width:320px; border-radius:12px; margin-bottom:1rem;" />
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
        st.markdown("""
        <div style="font-size:1.35rem;font-weight:700;color:#1f2937;padding:0.25rem 0 0.4rem 0;">
            📚 แหล่งข้อมูล
        </div>""", unsafe_allow_html=True)

        sidebar_tab_docs, sidebar_tab_notes, sidebar_tab_web = st.tabs(
            ["📄 Documents", "📝 Notes", "🌐 Web"]
        )

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

                                # Save document metadata to SQLite
                                doc_id = database.save_document_metadata(
                                    filename=uploaded_file.name,
                                    file_type=ext.lstrip('.'),
                                    chunk_count=len(child_chunks),
                                    db_path=UNIFIED_DB_PATH,
                                )
                                # Inject doc_id into child chunk metadata
                                for chunk in child_chunks:
                                    chunk.metadata['doc_id'] = doc_id
                                if summary_docs:
                                    for sdoc in summary_docs:
                                        sdoc.metadata['doc_id'] = doc_id

                                all_child_chunks.extend(child_chunks)
                                all_parent_records.extend(parent_records)
                                all_summary_docs.extend(summary_docs)
                                new_doc_entries.append({
                                    "name": uploaded_file.name,
                                    "chunks": len(child_chunks),
                                    "doc_id": doc_id,
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
                            # Remove document metadata from SQLite
                            if doc_entry.get("doc_id"):
                                database.delete_document_by_id(doc_entry["doc_id"])
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

        # ── Tab 3: Web ─────────────────────────────────────────────────────
        with sidebar_tab_web:
            st.markdown("**เพิ่มข้อมูลจากเว็บ**")
            web_url_input = st.text_input(
                "ลิงก์เว็บไซต์",
                placeholder="วาง URL ที่นี่ เช่น https://...",
                key="web_url_input",
                label_visibility="collapsed"
            )

            scrape_clicked = st.button(
                "🔍 ดึงข้อมูลจาก web เข้าฐานข้อมูล",
                type="primary",
                key="scrape_btn",
                use_container_width=True,
                disabled=not web_url_input.strip()
            )

            if scrape_clicked and web_url_input.strip():
                total_input_tokens = 0
                total_output_tokens = 0

                with st.status("กำลังดึงข้อมูล...", expanded=True) as status:
                    st.write("กำลังเปิดหน้าเว็บ...")
                    scrape_result = scrape_url(web_url_input.strip())

                    if not scrape_result['success']:
                        status.update(label="ไม่สำเร็จ", state="error")
                        st.error(scrape_result['error'])
                    else:
                        web_content = scrape_result['content']
                        st.write("กำลังอ่านและสรุปเนื้อหา...")
                        summary_result = summarize_content(web_content)

                        if not summary_result['success']:
                            status.update(label="ไม่สำเร็จ", state="error")
                            st.error(summary_result['error'])
                        else:
                            summary_text = summary_result['summary']
                            total_input_tokens += summary_result['input_tokens']
                            total_output_tokens += summary_result['output_tokens']

                            st.write("กำลังตั้งชื่อ...")
                            title_result = generate_title(web_content)
                            if title_result['success']:
                                auto_title = title_result['title']
                                total_input_tokens += title_result['input_tokens']
                                total_output_tokens += title_result['output_tokens']
                            else:
                                auto_title = web_content[:60].replace('\n', ' ').strip()

                            st.write("กำลังบันทึกลงฐานข้อมูล...")
                            try:
                                web_page_id = database.save_web_page(
                                    url=scrape_result['url'],
                                    title=auto_title,
                                    summary=summary_text,
                                    chunk_count=0,
                                )

                                child_chunks, parent_records = prepare_web_chunks(
                                    summary_text, auto_title,
                                    scrape_result['url'],
                                    web_page_id=web_page_id,
                                )

                                if st.session_state.unified_vector_store is None:
                                    st.session_state.unified_vector_store = get_or_create_vector_store(
                                        db_path=UNIFIED_DB_PATH,
                                        embeddings=embeddings,
                                        collection_name=UNIFIED_COLLECTION,
                                    )

                                ingest_documents(
                                    st.session_state.unified_vector_store,
                                    child_chunks,
                                    parent_records,
                                )

                                database.update_web_page(
                                    web_page_id, auto_title, summary_text,
                                    chunk_count=len(child_chunks),
                                )

                                total_tokens_turn = total_input_tokens + total_output_tokens
                                cost_thb = (total_tokens_turn / 1_000_000) * 0.4 * 35
                                st.session_state.total_tokens += total_tokens_turn
                                st.session_state.total_cost_thb += cost_thb

                                status.update(label="เสร็จสิ้น", state="complete")
                                st.success(f"บันทึกแล้ว — **{auto_title}**")
                            except Exception as e:
                                status.update(label="ไม่สำเร็จ", state="error")
                                st.error(f"เกิดข้อผิดพลาด: {str(e)}")

            # ── รายการเว็บที่บันทึกไว้ ──
            web_pages = database.load_all_web_pages()
            if web_pages:
                st.divider()
                st.caption(f"เว็บที่บันทึกไว้ ({len(web_pages)})")

                for wp in web_pages:
                    col_info, col_edit, col_del = st.columns([4, 1, 1])
                    with col_info:
                        st.markdown(f"🌐 **{wp['title']}**")
                        st.caption(wp['timestamp'])
                    with col_edit:
                        if st.button("✏️", key=f"edit_web_{wp['id']}",
                                     help="แก้ไขชื่อ"):
                            st.session_state._web_edit_id = wp['id']
                            st.rerun()
                    with col_del:
                        if st.button("🗑️", key=f"del_web_{wp['id']}",
                                     help="ลบออก"):
                            if st.session_state.unified_vector_store is not None:
                                try:
                                    st.session_state.unified_vector_store \
                                        ._collection.delete(
                                            where={"doc_name": wp['url']}
                                        )
                                except Exception:
                                    pass
                            database.delete_parent_chunks_by_source(wp['url'])
                            database.delete_web_page_by_id(wp['id'])
                            st.rerun()

            # ── Pop-up แก้ไขชื่อ ──
            _web_edit_id = st.session_state.get("_web_edit_id")
            if _web_edit_id is not None:
                _show_web_edit_dialog(_web_edit_id)


    # ============================================================================
    # MAIN CONTENT: Center (Research Workbench) | Right (Assistant)
    # ============================================================================
    col_center, col_right = st.columns([3, 2], gap="large")

    # ── Center: Research Workbench ─────────────────────────────────────────────────────
    with col_center:
        st.markdown("""
        <div style="font-size:1.35rem;font-weight:700;color:#1f2937;padding:0.25rem 0 0.4rem 0;">
            📝 Research Workbench
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

        # (Long-Form Generator moved to right panel)

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

        # ── Advisor Review Result (below editor) ─────────────────────────────
        if st.session_state.review_result:
            with st.expander("🎓 ผลการตรวจจากอาจารย์ที่ปรึกษา",
                             expanded=st.session_state.review_expanded):
                _render_review_result(st.session_state.review_result)
                if st.button("🗑️ ล้างผลตรวจ", key="clear_review_btn",
                             use_container_width=True):
                    st.session_state.review_result = None
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
        _logo_b64 = base64.b64encode(
            pathlib.Path(__file__).parent.joinpath("pic", "logo.jpeg").read_bytes()
        ).decode()
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:0.25rem 0 0.4rem 0;">
            <img src="data:image/jpeg;base64,{_logo_b64}"
                 style="height:28px;width:28px;border-radius:6px;object-fit:cover;" />
            <span style="font-size:1.35rem;font-weight:700;color:#1f2937;">Assistant</span>
        </div>""", unsafe_allow_html=True)

        unified_store = st.session_state.unified_vector_store
        has_context = unified_store is not None

        # ── Chat container (compact) — show only after first interaction ──
        if st.session_state.messages:
            chat_container = st.container(height=300)
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
                                with st.expander("📚 แหล่งข้อมูล"):
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
                            st.write(message["content"])

        # Placeholder for spinner — sits right below chat container, above chat input
        _chat_spinner_area = st.empty()

        # ── JS helpers: widget warnings + content edit overlay ─────────────
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

        // ── Event Delegation: ผูก listener เดียวที่ parent.document ──
        // ลบ listener เก่าออกก่อน (ป้องกันทับซ้อนเมื่อ st.rerun() สร้าง iframe ใหม่)
        if (parent.window._customContextMenuListener) {
            parent.document.removeEventListener('contextmenu', parent.window._customContextMenuListener);
        }

        parent.window._customContextMenuListener = function(e) {
            const ta = e.target;
            if (ta && ta.tagName === 'TEXTAREA' && ta.placeholder && ta.placeholder.includes('Start writing')) {
                const sel = ta.value.substring(ta.selectionStart, ta.selectionEnd);
                if (sel.trim().length > 0) {
                    e.preventDefault();
                    _showEditOverlay(e.clientX, e.clientY, sel);
                }
            }
        };

        parent.document.addEventListener('contextmenu', parent.window._customContextMenuListener);
        </script>
        """, height=0)

        _input_placeholder = (
            "📖 ถามคำถาม — AI จะตอบเชิงลึก..."
            if st.session_state._research_mode
            else "ถามคำถาม..."
        )
        prompt = st.chat_input(_input_placeholder, key="chat_input_main")

        # ── Toggle: deep/short ────────────────────────────────────────────
        _deep_mode = st.toggle(
            "📖 ตอบเชิงลึก",
            value=st.session_state._research_mode,
            key="_deep_toggle",
            help="เปิด: AI ตอบละเอียด วิเคราะห์เชิงลึก | ปิด: ตอบสั้นกระชับ",
        )
        st.session_state._research_mode = _deep_mode

        # ── Clear chat button ─────────────────────────────────────────────
        if st.button("🗑️ ล้างประวัติการสนทนา", key="clear_chat_btn", type="secondary",
                     use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_tokens = 0
            st.session_state.total_cost_thb = 0.0
            st.rerun()

        # ── Section-by-Section: สร้างเนื้อหาวิจัย ────────────────────────
        with st.expander("📝 สร้างเนื้อหาวิจัย", expanded=False):
            sec_topic = st.text_input(
                "หัวข้อเอกสาร",
                value=work_title if work_title.strip() else "",
                placeholder="เช่น ผลกระทบของ AI ต่อการศึกษา",
                key="sec_topic_input",
            )
            _input_mode = st.radio(
                "วิธีระบุส่วนที่ต้องการเขียน",
                ["เลือกจาก preset", "กำหนดเอง"],
                horizontal=True,
                key="sec_input_mode",
                label_visibility="collapsed",
            )

            if _input_mode == "เลือกจาก preset":
                sec_presets = [
                    "บทที่ 1: บทนำ — ที่มาและความสำคัญ วัตถุประสงค์ ขอบเขต",
                    "บทที่ 2: ทบทวนวรรณกรรม — ทฤษฎีและงานวิจัยที่เกี่ยวข้อง",
                    "บทที่ 3: วิธีดำเนินการวิจัย — ประชากร เครื่องมือ การเก็บข้อมูล",
                    "บทที่ 4: ผลการวิจัย — นำเสนอข้อมูลและการวิเคราะห์",
                    "บทที่ 5: สรุป อภิปราย และข้อเสนอแนะ",
                ]
                preset_choice = st.radio(
                    "เลือก preset",
                    sec_presets,
                    key="sec_preset_select",
                    label_visibility="collapsed",
                )
                sec_instruction = ""
            else:
                sec_instruction = st.text_area(
                    "ส่วนที่ต้องการเขียน",
                    placeholder=(
                        "เช่น บทนำ — ที่มาและความสำคัญของปัญหา\n"
                        "หรือ ทบทวนวรรณกรรม — ทฤษฎีและงานวิจัยที่เกี่ยวข้อง"
                    ),
                    height=80,
                    key="sec_instruction_input",
                )
                preset_choice = None

            sec_generate = st.button(
                "🚀 สร้างเนื้อหา",
                key="sec_generate_btn",
                type="primary",
                use_container_width=True,
            )

            if sec_generate:
                final_instruction = preset_choice if preset_choice else sec_instruction.strip()

                if not sec_topic.strip():
                    st.warning("⚠️ กรุณาระบุหัวข้อเอกสาร")
                elif not final_instruction:
                    st.warning("⚠️ กรุณาระบุส่วนที่ต้องการเขียน หรือเลือกจาก preset")
                else:
                    _sec_store = st.session_state.unified_vector_store
                    retrieved = []
                    if _sec_store is not None:
                        try:
                            retrieved = retrieve_unified(
                                _sec_store, f"{sec_topic} {final_instruction}", k=3
                            )
                        except Exception as e:
                            st.warning(f"⚠️ ไม่สามารถดึงบริบทจากเอกสารได้: {e}")

                    with st.spinner(f"✍️ กำลังเขียน: {final_instruction[:50]}..."):
                        try:
                            section_text, ri, ro = generate_section(
                                topic=sec_topic.strip(),
                                section_instruction=final_instruction,
                                retrieved_docs=retrieved,
                                existing_content=work_content,
                            )

                            separator = "\n\n" if work_content.strip() else ""
                            new_content = work_content + separator + section_text

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
                                "content": f"📝 เขียนส่วน \"{final_instruction[:60]}\" เสร็จแล้ว — ต่อท้ายใน editor",
                                "tokens": total_tokens_turn,
                                "cost_thb": cost_thb,
                                "action": "edit",
                            })
                            st.rerun()
                        except ValueError as e:
                            st.error(f"❌ API Error: {str(e)}")
                        except Exception as e:
                            st.error(f"❌ เกิดข้อผิดพลาดในการสร้างเนื้อหา: {str(e)}")

        # ── Advisor Review Section ────────────────────────────────────────
        st.markdown("""
        <div style="
            margin-top: 16px;
            padding: 14px 16px 10px 16px;
            background: #e0f0ff;
            border: 1.5px solid #001f5b;
            border-radius: 12px;
        ">
            <div style="font-size: 1.05rem; font-weight: 700; color: #001f5b; margin-bottom: 6px;">
                🎓 Advisor — ตรวจงานวิจัย
            </div>
            <div style="font-size: 0.8rem; color: #334155; margin-bottom: 4px;">
                อาจารย์ที่ปรึกษา AI จะ review งานใน Research Workbench
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)

        review_focus = ""
        with st.expander("💬 อยากตรวจอะไรเป็นพิเศษ? (optional)", expanded=False):
            review_focus = st.text_input(
                "ระบุสิ่งที่อยากให้เน้น review",
                value="",
                placeholder="เช่น ตรวจบทที่ 2, ดูการอ้างอิง, ตรวจระเบียบวิธี...",
                key="review_focus_input",
                label_visibility="collapsed",
            )

        st.markdown('<div style="margin-top:6px;"></div>', unsafe_allow_html=True)

        st.markdown("""
        <style>
        .st-key-advisor_review_btn button {
            background-color: #001f5b !important;
            color: white !important;
            border: none !important;
        }
        .st-key-advisor_review_btn button:hover {
            background-color: #002d7a !important;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("🎓 ส่งให้อาจารย์ตรวจ", type="primary",
                     key="advisor_review_btn", use_container_width=True):
            # Read from the actual widget value (work_content from col_center),
            # NOT from work_content_val which is only updated by AI edits.
            editor_text = st.session_state.get("work_content_input", "")
            if not editor_text or not editor_text.strip():
                st.warning("⚠️ ไม่มีเนื้อหาใน Research Workbench ให้ตรวจ")
            else:
                with st.spinner("🎓 อาจารย์กำลังตรวจงาน..."):
                    try:
                        review_text, ri, ro = review_research(
                            editor_text,
                            user_focus=review_focus,
                        )
                        total_tokens_turn = ri + ro
                        cost_thb = (total_tokens_turn / 1_000_000) * 0.4 * 35
                        st.session_state.total_tokens += total_tokens_turn
                        st.session_state.total_cost_thb += cost_thb
                        st.session_state.review_result = review_text
                        st.session_state.review_expanded = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")

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

                    with _chat_spinner_area, st.spinner("✏️ กำลังแก้ไขข้อความที่เลือก..."):
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

            # ── Detect mode: toggle or /research prefix ────────────────────
            is_research = st.session_state._research_mode
            actual_query = prompt
            # /research prefix still works as override
            if re.match(r'^/research\b\s*', prompt, re.IGNORECASE):
                is_research = True
                actual_query = re.sub(r'^/research\s*', '', prompt, flags=re.IGNORECASE).strip()
                st.session_state._research_mode = True

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
            with _chat_spinner_area, st.spinner(spinner_text):
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

                except ValueError as e:
                    # API errors (invalid key, quota, HTTP errors) from generator.py
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": str(e),
                    })
                except requests.exceptions.Timeout:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "❌ การเชื่อมต่อ API หมดเวลา กรุณาลองใหม่อีกครั้ง",
                    })
                except requests.exceptions.ConnectionError:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "❌ ไม่สามารถเชื่อมต่อ API ได้ กรุณาตรวจสอบการเชื่อมต่ออินเทอร์เน็ต",
                    })
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ เกิดข้อผิดพลาด: {str(e)}",
                    })
            st.rerun()


if __name__ == "__main__":
    main()
