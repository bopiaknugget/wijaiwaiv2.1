---
name: User Isolation Security Fix
description: How per-user data isolation is implemented across SQLite tables and editor documents
type: project
---

All user-owned data is isolated by `user_id` (the Google OAuth `id` field, format `google_XXXXXXXXXXXX`).

**Migration approach:** `_add_column_if_missing()` in `database.py` adds `user_id TEXT` to `documents`, `research_notes`, and `web_pages` via `ALTER TABLE` on every startup. Pre-existing rows with `user_id = NULL` are intentionally not shown to any user after the fix (they become orphaned legacy data).

**Editor documents:** Stored in a new `editor_documents` SQLite table with `UNIQUE(user_id, name)`. The `work_current_file` session state key now holds the doc **name** string (not a filesystem path). `save_work_to_db()` overwrites by name; `save_work_to_db_new()` appends a timestamp to create a new unique name. The `./user_data/` directory still exists for Import-from-disk only.

**Functions updated with `user_id` param:**
- `save_note`, `load_all_notes`, `delete_note_by_id`
- `save_document_metadata`, `load_all_documents`, `delete_document_by_id`
- `save_web_page`, `load_all_web_pages`, `delete_web_page_by_id`
- New: `save_editor_document`, `load_editor_document`, `list_editor_documents`, `delete_editor_document`

**Why:** Two users (`tanawatl.cs@gmail.com` and `richmantanawat@gmail.com`) could see each other's documents — both the SQLite sidebar list and the editor Load panel used global queries with no user filter. Pinecone was already isolated by namespace (`user_id`); only SQLite was missing isolation.

**How to apply:** When adding new user-owned tables in future, always include `user_id TEXT` in the schema and filter all read/write/delete operations by it. Use `_add_column_if_missing()` for migrations.
