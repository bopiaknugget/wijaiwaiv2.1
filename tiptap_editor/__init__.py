"""
tiptap_editor — Streamlit custom component wrapper
Provides a rich text editor (TipTap React) via
st.components.v1.declare_component, served from ./frontend-editor/dist/
"""

import os
import streamlit.components.v1 as components

# Absolute path to the built frontend
_BUILD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # project root
    "frontend-editor",
    "dist"
)

# Dev mode: set DEV_MODE=true to use Vite dev server
if os.getenv("DEV_MODE") == "true":
    _component_func = components.declare_component(
        "tiptap_editor",
        url="http://localhost:5173"
    )
else:
    _component_func = components.declare_component(
        "tiptap_editor",
        path=_BUILD_DIR
    )


def st_tiptap(value: str = "", height: int = 480, key: str = None) -> str:
    """Render a TipTap rich-text editor and return current HTML content.

    Parameters
    ----------
    value : str
        Initial HTML content to display in the editor.
    height : int
        Approximate iframe height in pixels (default 480).
    key : str
        Unique Streamlit widget key. Required when multiple editors are shown.

    Returns
    -------
    str
        The editor's current HTML content as a string.
        Falls back to `value` if the component has not yet sent a value back
        (e.g., on the very first render before any interaction).
    """
    result = _component_func(value=value, height=height, key=key, default=value)
    if result is None:
        return value
    return result
