Context: 
This is "Research Workbench", a Streamlit-based application utilizing a RAG pipeline and OpenThaiGPT API. Currently, the LLM generates responses that are too short and concise, especially when users request long-form academic content (e.g., "Write a 2-page research paper").

Goal: 
Enhance the generation pipeline (primarily focusing on `generator.py` and minor UI additions in `app.py`) to force the LLM to produce long-form, highly detailed content (1,000+ words). You must implement advanced prompt engineering techniques without breaking any existing core functionalities.

Actionable Tasks:
1. Optimize API Parameters: Locate the API calling functions (e.g., in `generator.py`). Ensure `max_tokens` is set to its maximum safe limit (e.g., 4096 or 8192) and adjust the `temperature` to 0.6 - 0.7 specifically for the content generation persona to encourage detailed elaboration.
2. Upgrade the System Prompt (Prompt Engineering): 
   - Inject "Anti-Summary Directives": Explicitly instruct the AI not to summarize, but to deeply elaborate on theories, provide examples, and thoroughly analyze the retrieved context.
   - Change length metrics: Instruct the AI to measure length in "words" and "paragraphs" (e.g., "at least 1,000 words and 10 paragraphs") instead of ambiguous "pages".
   - Mandate Chain of Thought: Force the AI to plan its document structure inside `<think>...</think>` tags before writing the actual output. This forces the model to generate a longer, well-structured response.
3. Add a "Section-by-Section" mode (Safe Feature Addition): In `app.py`, safely introduce a UI option (e.g., a "Long-Form Generate" button or an explicit instruction in the chat) that encourages appending content section-by-section to `st.session_state.work_content_val`, rather than trying to generate a massive document in a single prompt.

Strict Constraints (CRITICAL - DO NOT BREAK):
- DO NOT modify `vector_store.py`, `database.py`, or `document_loader.py`. The RAG data pipeline, ChromaDB, and SQLite are stable and must remain untouched.
- DO NOT break the Streamlit session state variables in `app.py` (e.g., `ai_edit_undo_stack`, `messages`, `_pending_work_content`).
- Preserve the existing UI layout structure (Sidebar, Workbench, Assistant). Any new UI elements must be unobtrusive.
- Maintain the exact OpenThaiGPT API request structure (headers, payload format).
- Ensure backward compatibility: Standard short-form Q&A in the chat must still work normally when the user just asks simple questions.

Please analyze the current `generator.py` and `app.py` and propose the exact code modifications to achieve this.