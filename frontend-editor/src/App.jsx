import React, { useEffect, useRef } from 'react'
import { TextSelection } from '@tiptap/pm/state'
import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Table from '@tiptap/extension-table'
import TableRow from '@tiptap/extension-table-row'
import TableCell from '@tiptap/extension-table-cell'
import TableHeader from '@tiptap/extension-table-header'
import {
  Streamlit,
  withStreamlitConnection,
} from 'streamlit-component-lib'
import './App.css'

const IconBold = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M6 4h8a4 4 0 0 1 4 4 4 4 0 0 1-4 4H6z"/>
    <path d="M6 12h9a4 4 0 0 1 4 4 4 4 0 0 1-4 4H6z"/>
  </svg>
)

const IconItalic = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="19" y1="4" x2="10" y2="4"/>
    <line x1="14" y1="20" x2="5" y2="20"/>
    <line x1="15" y1="4" x2="9" y2="20"/>
  </svg>
)

const IconH1 = () => (
  <svg width="18" height="16" viewBox="0 0 28 24" fill="currentColor">
    <text x="0" y="18" fontFamily="serif" fontWeight="700" fontSize="18">H</text>
    <text x="13" y="18" fontFamily="serif" fontWeight="700" fontSize="18">1</text>
  </svg>
)

const IconH2 = () => (
  <svg width="18" height="16" viewBox="0 0 28 24" fill="currentColor">
    <text x="0" y="18" fontFamily="serif" fontWeight="700" fontSize="18">H</text>
    <text x="13" y="18" fontFamily="serif" fontWeight="700" fontSize="18">2</text>
  </svg>
)

const IconBulletList = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="9" y1="6" x2="20" y2="6"/>
    <line x1="9" y1="12" x2="20" y2="12"/>
    <line x1="9" y1="18" x2="20" y2="18"/>
    <circle cx="4" cy="6" r="1.5" fill="currentColor" stroke="none"/>
    <circle cx="4" cy="12" r="1.5" fill="currentColor" stroke="none"/>
    <circle cx="4" cy="18" r="1.5" fill="currentColor" stroke="none"/>
  </svg>
)

const IconOrderedList = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="10" y1="6" x2="21" y2="6"/>
    <line x1="10" y1="12" x2="21" y2="12"/>
    <line x1="10" y1="18" x2="21" y2="18"/>
    <text x="2" y="8" fontSize="7" fill="currentColor" stroke="none" fontWeight="700">1.</text>
    <text x="2" y="14" fontSize="7" fill="currentColor" stroke="none" fontWeight="700">2.</text>
    <text x="2" y="20" fontSize="7" fill="currentColor" stroke="none" fontWeight="700">3.</text>
  </svg>
)

const IconTable = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2"/>
    <line x1="3" y1="9" x2="21" y2="9"/>
    <line x1="3" y1="15" x2="21" y2="15"/>
    <line x1="9" y1="3" x2="9" y2="21"/>
    <line x1="15" y1="3" x2="15" y2="21"/>
  </svg>
)

const IconAddCol = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="12" height="18" rx="2"/>
    <line x1="3" y1="9" x2="15" y2="9"/>
    <line x1="3" y1="15" x2="15" y2="15"/>
    <line x1="9" y1="3" x2="9" y2="21"/>
    <line x1="19" y1="8" x2="19" y2="16"/>
    <line x1="15" y1="12" x2="23" y2="12"/>
  </svg>
)

const IconAddRow = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="12" rx="2"/>
    <line x1="3" y1="9" x2="21" y2="9"/>
    <line x1="9" y1="3" x2="9" y2="15"/>
    <line x1="15" y1="3" x2="15" y2="15"/>
    <line x1="12" y1="19" x2="12" y2="24"/>
    <line x1="8" y1="21.5" x2="16" y2="21.5"/>
  </svg>
)

const IconDelTable = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2"/>
    <line x1="3" y1="9" x2="21" y2="9"/>
    <line x1="3" y1="15" x2="21" y2="15"/>
    <line x1="9" y1="3" x2="9" y2="21"/>
    <line x1="15" y1="3" x2="15" y2="21"/>
    <line x1="9" y1="9" x2="21" y2="21" strokeWidth="2.5" stroke="#e53e3e"/>
    <line x1="21" y1="9" x2="9" y2="21" strokeWidth="2.5" stroke="#e53e3e"/>
  </svg>
)

function TipTapEditor({ args }) {
  const { value = '', height = 480 } = args
  const lastHtml = useRef(value)

  const editor = useEditor({
    extensions: [
      StarterKit,
      Table.configure({ resizable: true }),
      TableRow,
      TableHeader,
      TableCell,
    ],
    content: value,
    onUpdate({ editor }) {
      const html = editor.getHTML()
      if (html !== lastHtml.current) {
        lastHtml.current = html
        Streamlit.setComponentValue(html)
      }
    },
  })

  // Sync external value changes (e.g. AI writes new content).
  //
  // IMPORTANT: compare against editor.getHTML() — not lastHtml.current.
  //
  // Why: Streamlit reruns app.py on every keystroke. Each rerun passes
  // `value=work_content_val` as a prop, but work_content_val is always
  // one rerun behind the live editor content. If we compared against
  // lastHtml.current (which was set to the just-typed HTML), the prop
  // arriving from the stale rerun would appear "different" and trigger
  // setContent(), overwriting the user's live text and jumping the cursor.
  //
  // By comparing against editor.getHTML() directly, we only call setContent
  // when the prop carries content that the editor doesn't already display —
  // i.e., a genuine AI-driven external update, not a stale echo.
  useEffect(() => {
    if (!editor) return
    const currentEditorHtml = editor.getHTML()
    if (value !== currentEditorHtml) {
      editor.commands.setContent(value, false)
      lastHtml.current = value
    }
  }, [value, editor])

  useEffect(() => {
    Streamlit.setFrameHeight(height)
  }, [height])

  const addTable = () => {
    if (!editor) return

    // If cursor is inside a table, move it to after the table first
    // to avoid unsupported nested-table insertion
    if (editor.isActive('table')) {
      const { state } = editor
      const { $from } = state.selection
      for (let d = $from.depth; d >= 0; d--) {
        if ($from.node(d).type.name === 'table') {
          const afterTable = $from.after(d)
          // If there's no node after the table, insert a paragraph first
          if (afterTable >= state.doc.content.size) {
            editor.chain().focus()
              .insertContentAt(afterTable, { type: 'paragraph' })
              .insertTable({ rows: 3, cols: 3, withHeaderRow: true })
              .run()
          } else {
            const sel = TextSelection.near(state.doc.resolve(afterTable), 1)
            editor.view.dispatch(state.tr.setSelection(sel))
            editor.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run()
          }
          return
        }
      }
    }

    editor.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run()
  }

  return (
    <div className="editor-wrapper" style={{ height }}>
      <div className="toolbar">
        <button
          onClick={() => editor?.chain().focus().toggleBold().run()}
          className={editor?.isActive('bold') ? 'active' : ''}
          title="Bold"
        >
          <IconBold />
        </button>
        <button
          onClick={() => editor?.chain().focus().toggleItalic().run()}
          className={editor?.isActive('italic') ? 'active' : ''}
          title="Italic"
        >
          <IconItalic />
        </button>
        <button
          onClick={() => editor?.chain().focus().toggleHeading({ level: 1 }).run()}
          className={editor?.isActive('heading', { level: 1 }) ? 'active' : ''}
          title="Heading 1"
        >
          <IconH1 />
        </button>
        <button
          onClick={() => editor?.chain().focus().toggleHeading({ level: 2 }).run()}
          className={editor?.isActive('heading', { level: 2 }) ? 'active' : ''}
          title="Heading 2"
        >
          <IconH2 />
        </button>
        <button
          onClick={() => editor?.chain().focus().toggleBulletList().run()}
          className={editor?.isActive('bulletList') ? 'active' : ''}
          title="Bullet List"
        >
          <IconBulletList />
        </button>
        <button
          onClick={() => editor?.chain().focus().toggleOrderedList().run()}
          className={editor?.isActive('orderedList') ? 'active' : ''}
          title="Ordered List"
        >
          <IconOrderedList />
        </button>
        <span className="toolbar-sep" />
        <button onClick={addTable} title="Insert Table"><IconTable /></button>
        <button onClick={() => editor?.chain().focus().addColumnAfter().run()} title="Add Column"><IconAddCol /></button>
        <button onClick={() => editor?.chain().focus().addRowAfter().run()} title="Add Row"><IconAddRow /></button>
        <button onClick={() => editor?.chain().focus().deleteTable().run()} title="Delete Table"><IconDelTable /></button>
      </div>
      <EditorContent editor={editor} className="editor-content" />
    </div>
  )
}

export default withStreamlitConnection(TipTapEditor)
