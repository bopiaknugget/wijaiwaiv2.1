"""
Research Workbench — AI-Powered RAG with Text Notes
3-panel layout: Sidebar (Docs + Notes) | Center (Your Work editor) | Right (Assistant chat)

Key improvements in this version:
- Separate vector stores: doc_vector_store (session) and note_vector_store (persistent)
- Delete buttons for Notes (SQLite + ChromaDB removal)
- Smaller chunk size for notes (300/50) for precise retrieval
- MMR-based merged retrieval from both stores for accurate, diverse context
- @st.cache_resource on embeddings; @st.cache_data on note loading
"""

import gc
import os
import re
import shutil
import tempfile
import uuid
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import database
from document_loader import load_document, chunk_documents
from generator import generate_answer
from vector_store import (
    initialize_embeddings,
    get_or_create_vector_store,
    retrieve_from_both_stores,
)


_THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)
WORK_DIR = os.path.join(os.path.dirname(__file__), "your_work")
NOTES_DB_PATH = "./notes_chroma_db"


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


# ── Note vector-store helpers ──────────────────────────────────────────────────

def _load_note_vector_store(embeddings):
    """Load (or initialise) the persistent notes ChromaDB."""
    return get_or_create_vector_store(
        db_path=NOTES_DB_PATH,
        chunked_documents=None,
        embeddings=embeddings,
        collection_name="documents"
    )


def _embed_note(note_id: int, title: str, content: str,
                embeddings, note_store):
    """Chunk a note with small chunks and add to note_vector_store."""
    doc = Document(
        page_content=content,
        metadata={"title": title, "source": "research_note", "note_id": note_id}
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents([doc])

    # Ensure note_store exists
    if note_store is None:
        note_store = get_or_create_vector_store(
            db_path=NOTES_DB_PATH,
            chunked_documents=chunks,
            embeddings=embeddings,
            collection_name="documents"
        )
    else:
        note_store.add_documents(chunks)

    return note_store


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="🔬 Research Workbench",
        page_icon="📚",
        layout="wide"
    )

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
    div[data-testid="stVerticalBlock"] > div:has(> .your-work-panel) {
        border-right: 1px solid #e5e7eb;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Session state defaults ─────────────────────────────────────────────────
    defaults = {
        "doc_vector_store": None,       # Session-scoped: uploaded documents
        "note_vector_store": None,      # Persistent: research notes
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
        "temp_db_path": f"./temp_chroma_db_{uuid.uuid4().hex[:8]}",
        "_notes_store_initialised": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Apply pending editor content BEFORE any widget is rendered
    for widget_key, pending_key in [
        ("work_title_input", "_pending_work_title"),
        ("work_content_input", "_pending_work_content"),
    ]:
        if pending_key in st.session_state:
            st.session_state[widget_key] = st.session_state.pop(pending_key)

    # ── Load embeddings (cached, runs once per session) ───────────────────────
    embeddings = initialize_embeddings()

    # ── Load persistent notes store on first run ──────────────────────────────
    if not st.session_state._notes_store_initialised:
        if os.path.exists(NOTES_DB_PATH):
            try:
                st.session_state.note_vector_store = _load_note_vector_store(embeddings)
            except Exception:
                st.session_state.note_vector_store = None
        st.session_state._notes_store_initialised = True

    # ============================================================================
    # SIDEBAR — Documents + Notes tabs
    # ============================================================================
    with st.sidebar:
        st.markdown("### 🔬 Research Workbench")
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
                    all_chunks = []
                    new_doc_entries = []

                    with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                        for uploaded_file in uploaded_files:
                            try:
                                ext = os.path.splitext(uploaded_file.name)[1].lower()
                                with tempfile.NamedTemporaryFile(
                                    delete=False, suffix=ext
                                ) as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    tmp_path = tmp_file.name

                                documents = load_document(tmp_path)
                                chunks = chunk_documents(
                                    documents, chunk_size=1000, chunk_overlap=200
                                )
                                # Tag every chunk with the original filename
                                for chunk in chunks:
                                    chunk.metadata["doc_name"] = uploaded_file.name
                                all_chunks.extend(chunks)
                                new_doc_entries.append({
                                    "name": uploaded_file.name,
                                    "chunks": len(chunks)
                                })
                                os.unlink(tmp_path)
                            except Exception as e:
                                st.error(f"❌ {uploaded_file.name}: {str(e)}")

                        if all_chunks:
                            try:
                                # Clean up old temp DB
                                old_db_path = st.session_state.temp_db_path
                                st.session_state.doc_vector_store = None
                                gc.collect()
                                try:
                                    if os.path.exists(old_db_path):
                                        shutil.rmtree(old_db_path)
                                except OSError:
                                    pass

                                new_db_path = f"./temp_chroma_db_{uuid.uuid4().hex[:8]}"
                                st.session_state.temp_db_path = new_db_path

                                st.session_state.doc_vector_store = get_or_create_vector_store(
                                    db_path=new_db_path,
                                    chunked_documents=all_chunks,
                                    embeddings=embeddings
                                )
                                st.session_state.processed_docs = new_doc_entries
                                st.session_state.messages = []
                                st.session_state.total_tokens = 0
                                st.session_state.total_cost_thb = 0.0
                                st.success(
                                    f"✅ {len(new_doc_entries)}/{len(uploaded_files)} "
                                    "file(s) ready!"
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
                            # Remove chunks from ChromaDB by doc_name metadata
                            if st.session_state.doc_vector_store is not None:
                                try:
                                    st.session_state.doc_vector_store \
                                        ._collection.delete(
                                            where={"doc_name": doc_entry["name"]}
                                        )
                                except Exception as e:
                                    st.error(f"VectorDB delete error: {e}")
                            # Remove from tracking list immediately
                            st.session_state.processed_docs = [
                                d for d in st.session_state.processed_docs
                                if d["name"] != doc_entry["name"]
                            ]
                            # If no docs remain, clear the store entirely
                            if not st.session_state.processed_docs:
                                st.session_state.doc_vector_store = None
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
                        # Embed into the dedicated note store with small chunks
                        st.session_state.note_vector_store = _embed_note(
                            note_id, note_title_input, note_content_input,
                            embeddings, st.session_state.note_vector_store
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
                            # 2. Remove from ChromaDB note store
                            if st.session_state.note_vector_store is not None:
                                try:
                                    st.session_state.note_vector_store \
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
    # MAIN CONTENT: Center (Your Work) | Right (Assistant)
    # ============================================================================
    col_center, col_right = st.columns([3, 2], gap="large")

    # ── Center: Your Work ─────────────────────────────────────────────────────
    with col_center:
        st.markdown("## ✍️ Your Work")

        work_title = st.text_input(
            "Title",
            value=st.session_state.work_title_val,
            placeholder="Enter a title for your work...",
            key="work_title_input"
        )
        work_content = st.text_area(
            "Content",
            value=st.session_state.work_content_val,
            placeholder="Start writing your work here...",
            height=400,
            key="work_content_input"
        )

        current_file = st.session_state.get("work_current_file")
        if current_file:
            st.caption(f"📄 `{os.path.basename(current_file)}`")

        btn_save, btn_save_as, btn_load_open = st.columns([1, 1, 1])
        with btn_save:
            save_clicked = st.button("💾 Save", type="primary",
                                     key="save_work_btn", use_container_width=True)
        with btn_save_as:
            save_as_clicked = st.button("💾 Save As", type="secondary",
                                        key="save_as_work_btn", use_container_width=True)
        with btn_load_open:
            load_work_clicked = st.button("📂 Load", type="secondary",
                                          key="load_work_btn", use_container_width=True)

        btn_export, btn_import = st.columns([1, 1])
        with btn_export:
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
        with btn_import:
            import_clicked = st.button("📥 Import", type="secondary",
                                       key="import_work_btn", use_container_width=True)

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
        st.markdown("## 🤖 Assistant")

        doc_store = st.session_state.doc_vector_store
        note_store = st.session_state.note_vector_store
        has_context = (doc_store is not None) or (note_store is not None)

        chat_container = st.container(height=480)
        with chat_container:
            if has_context:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        if message["role"] == "assistant":
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
                            st.write(message["content"])
            else:
                st.info(
                    "📄 Upload and process documents **or** save a research note "
                    "in the sidebar to start chatting."
                )

        # Chat input
        prompt = st.chat_input(
            "Ask about your documents and notes...", key="chat_input_main"
        )

        # ── Clear Chat button — below the chat input ───────────────────────
        if st.button("🗑️ Clear Chat History", type="secondary",
                     key="clear_chat_btn", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_tokens = 0
            st.session_state.total_cost_thb = 0.0
            st.rerun()

        if prompt:
            if has_context:
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.spinner("Analyzing..."):
                    try:
                        chat_history = st.session_state.messages[:-1]

                        # Merge MMR results from both stores
                        retrieved_docs = retrieve_from_both_stores(
                            doc_store, note_store, prompt, k=3
                        )

                        answer, input_tokens, output_tokens = generate_answer(
                            prompt, retrieved_docs, chat_history
                        )

                        total_tokens_turn = input_tokens + output_tokens
                        cost_thb = (total_tokens_turn / 1_000_000) * 0.4 * 35

                        st.session_state.total_tokens += total_tokens_turn
                        st.session_state.total_cost_thb += cost_thb

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": retrieved_docs,
                            "tokens": total_tokens_turn,
                            "cost_thb": cost_thb
                        })
                    except Exception as e:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"❌ Error: {str(e)}"
                        })
                st.rerun()
            else:
                st.warning("📄 Please process documents or save a note first.")


if __name__ == "__main__":
    main()
