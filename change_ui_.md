Your task is to refactor an existing Streamlit application to replace a plain text editor (st.text_area) with a TipTap-based rich text editor.
Implement TipTap editor with table support while keeping system impact minimal.

## ⚠️ Strict Constraints (VERY IMPORTANT)
- DO NOT modify backend business logic
- DO NOT change database schema
- DO NOT change AI pipeline functions
- DO NOT refactor RAG / Pinecone / retrieval logic
- KEEP all existing function signatures unchanged
- Editor content must still be stored as a string (HTML)

## 🏗️ Current State
- Editor uses: st.text_area
- Content stored in: st.session_state.work_content_val (string)
- Saved to DB as string
- AI functions expect plain text input

## ✅ Required Changes

### 1. Replace Editor UI
- Remove st.text_area
- Implement a custom Streamlit component using TipTap (React)
- Support:
  - bold, italic, headings
  - bullet list
  - table (must work)

### 2. Frontend (React + TipTap)
- Use:
  - @tiptap/react
  - @tiptap/starter-kit
  - @tiptap/extension-table
- Editor must:
  - accept initial content as HTML string
  - return updated content as HTML string
- On every update:
  - send HTML back to Streamlit

### 3. Streamlit Integration
- Create wrapper:
  - function: st_tiptap(value: str) -> str
- Replace usage:
  - st.text_area → st_tiptap

### 4. AI Compatibility Layer
- Before sending content to LLM:
  - convert HTML → plain text
- Use Python (BeautifulSoup or similar)

### 5. Rendering
- When displaying content:
  - use st.markdown(..., unsafe_allow_html=True)

## 🚫 What NOT to Do
- DO NOT switch to JSON document model
- DO NOT implement patch system
- DO NOT redesign editor architecture
- DO NOT change how undo/redo works
- DO NOT break existing session_state logic

## 📦 Deliverables
1. React TipTap editor component
2. Streamlit wrapper (Python)
3. Updated editor integration code
4. HTML → text helper function
5. Minimal changes only (diff-style if possible)

## 🧪 Acceptance Criteria
- Existing documents load correctly
- Save / Load still works
- AI generation still works
- Editor supports tables
- No breaking changes to backend

## 🧠 Mindset
This is a UI upgrade, NOT a system rewrite.

Optimize for:
- stability
- low risk
- minimal diff