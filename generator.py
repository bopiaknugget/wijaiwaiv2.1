"""
Response Generation Module – Agentic Editor Edition
Handles AI-powered answer generation and editor manipulation using OpenThaiGPT API.

Improvements in this version:
- Query routing: is_small_talk() detects queries that don't need vector retrieval
- Streaming: _call_api_stream() yields tokens as they arrive (SSE)
- generate_answer_stream(): streaming variant of generate_answer for use with
  st.write_stream() in Streamlit
"""

import json
import os
import re
import textwrap
from typing import Generator, Optional
import requests
from pathlib import Path
from dotenv import load_dotenv

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_PARAMETRIC_WARNING = (
    "⚠️ หมายเหตุ: ข้อมูลนี้มาจากฐานความรู้ทั่วไปของ AI "
    "เนื่องจากไม่พบข้อมูลในเอกสารหรือโน้ตของคุณ"
)

API_URL = "http://thaillm.or.th/api/openthaigpt/v1/chat/completions"
MODEL = "/model"

# ── Query Routing: small-talk patterns ────────────────────────────────────────
# Queries matching these patterns are answered directly from the LLM without
# touching the vector store.  This eliminates unnecessary Pinecone latency for
# conversational turns that carry no document-retrieval intent.

_SMALL_TALK_PATTERNS = re.compile(
    r"^("
    r"สวัสดี|หวัดดี|ดีครับ|ดีค่ะ|hello|hi\b|hey\b|"
    r"คุณชื่ออะไร|คุณเป็นใคร|คุณทำอะไรได้|ช่วยอะไรได้|"
    r"ขอบคุณ|thank(s|\s+you)|เยี่ยม|ดีมาก|โอเค|ok\b|okay\b|"
    r"ลาก่อน|bye\b|goodbye|"
    r"คุณเป็น ai|คุณเป็นหุ่นยนต์|are you (an? )?ai|"
    r"กี่โมง|วันนี้วันที่|today|what time"
    r")",
    re.IGNORECASE,
)


def is_small_talk(query: str) -> bool:
    """
    Return True if the query is conversational small talk that does not need
    vector database retrieval.

    The check is intentionally conservative: only obvious greetings and
    meta-questions about the assistant are flagged.  Any query mentioning
    research, documents, or content will pass through to retrieval normally.

    Args:
        query: The raw user input string

    Returns:
        bool: True if retrieval should be skipped
    """
    stripped = query.strip()
    if len(stripped) > 80:
        # Long queries almost certainly have research intent
        return False
    return bool(_SMALL_TALK_PATTERNS.match(stripped))


def _call_api(messages, api_key, max_tokens=2048, temperature=0.3):
    """
    Call the OpenThaiGPT API and return (content, input_tokens, output_tokens).
    """
    headers = {
        "Content-Type": "application/json",
        "apikey": api_key,
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    return content, input_tokens, output_tokens


def _call_api_stream(messages, api_key, max_tokens=2048,
                     temperature=0.3) -> Generator[str, None, None]:
    """
    Call the OpenThaiGPT API with streaming enabled (Server-Sent Events).

    Yields individual token strings as they arrive.  Callers should accumulate
    the yielded strings to reconstruct the full response.

    Note: If the API does not support streaming or returns a non-streaming
    response, this function falls back gracefully by yielding the full content
    as a single chunk.

    Args:
        messages: OpenAI-format message list
        api_key: OPENTHAI_API_KEY string
        max_tokens: Maximum output tokens
        temperature: Sampling temperature

    Yields:
        str: Token strings (may be multi-character chunks)
    """
    headers = {
        "Content-Type": "application/json",
        "apikey": api_key,
        "Accept": "text/event-stream",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    try:
        with requests.post(
            API_URL, headers=headers, json=payload,
            timeout=90, stream=True
        ) as response:
            response.raise_for_status()

            # Check if the server is actually streaming
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" not in content_type and "stream" not in content_type:
                # Server returned a regular JSON response — yield it wholesale
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    yield content
                return

            # Parse SSE stream
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if raw_line.startswith("data: "):
                    data_str = raw_line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    except requests.HTTPError as e:
        # Re-raise HTTP errors so callers can handle them
        raise
    except requests.RequestException as e:
        raise


def _extract_json(text: str):
    """
    Robustly extract the first JSON object from potentially mixed LLM output.
    Handles markdown code fences and conversational wrapper text.
    Returns a dict or None.
    """
    text = text.strip()
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Try direct parse first (ideal case — LLM returned only JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Scan for a JSON object using balanced-brace matching
    # This correctly handles nested strings containing braces
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
    return None


def generate_answer(query, retrieved_docs, chat_history=None, editor_content=None,
                    research_mode=False):
    """
    Generate an AI-powered answer or editor action using OpenThaiGPT API.

    The LLM classifies user intent as "chat" (Q&A) or "edit" (editor manipulation)
    and returns a structured JSON response. When no RAG context is available,
    the LLM may use parametric knowledge and a warning is prepended to the response.

    Args:
        query (str): The user's message / instruction
        retrieved_docs (list): Retrieved Document objects from vector store (may be empty)
        chat_history (list, optional): Previous messages for context re-phrasing
        editor_content (str, optional): Current content of the work editor
        research_mode (bool): When True, use research-optimised prompt with higher token budget

    Returns:
        tuple: (action, response_text, new_editor_content, input_tokens, output_tokens)
            action (str): "chat", "edit", or "research"
            response_text (str): Message to display in chat
            new_editor_content (str | None): New editor text when action="edit"/"research", else None
            input_tokens (int): Cumulative API input token count
            output_tokens (int): Cumulative API output token count
    """
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("OPENTHAI_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ OPENTHAI_API_KEY not found. Please ensure:\n"
            f"  1. You have a .env file at: {env_path.resolve()}\n"
            "  2. OPENTHAI_API_KEY=your_key_here is set (no quotes, no spaces)"
        )

    total_input = 0
    total_output = 0

    # ── Step 1: Build main generation prompt ──────────────────────────────────
    has_context = bool(retrieved_docs)
    # Cap each doc at 1,500 chars and total context at 6,000 chars (~1,500 tokens)
    # to stay well inside the 16K context window and keep LLM latency low
    _MAX_DOC_CHARS = 1500
    _MAX_CONTEXT_CHARS = 6000
    if retrieved_docs:
        parts = [doc.page_content[:_MAX_DOC_CHARS] for doc in retrieved_docs]
        context_text = "\n\n".join(parts)[:_MAX_CONTEXT_CHARS]
    else:
        context_text = ""

    # ── Language instruction ───────────────────────────────────────────────────
    language_instruction = (
        "ภาษา: ตอบภาษาเดียวกับคำถาม (default ไทย), technical terms ใช้อังกฤษได้"
    )

    if research_mode:
        # ── Research mode: deep analysis → JSON with editor_content ──────────
        if has_context:
            knowledge_instruction = "ใช้บริบทจากเอกสารเป็นหลักในการวิเคราะห์และสังเคราะห์"
        else:
            knowledge_instruction = "ไม่มีเอกสารถูกโหลดไว้ — ใช้ความรู้ทั่วไปของ AI"

        system_prompt = (
            "คุณเป็นนักวิจัยผู้เชี่ยวชาญ ค้นคว้า วิเคราะห์ และสังเคราะห์ข้อมูลเชิงลึก\n"
            f"{knowledge_instruction}\n"
            f"{language_instruction}\n\n"
            "กฎการเขียน:\n"
            "- เนื้อหา ≥1,000 คำ, ≥10 ย่อหน้า, แต่ละย่อหน้า ≥80 คำ พร้อมตัวอย่างและการวิเคราะห์\n"
            "- โครงสร้าง: บทนำ → ทบทวนวรรณกรรม → วิเคราะห์เชิงลึก → ผลการศึกษา → สรุป/ข้อเสนอแนะ\n\n"
            "ตอบ JSON เท่านั้น:\n"
            "{\"action\":\"research\",\"response\":\"สรุป 1-2 ประโยค\",\"editor_content\":\"เนื้อหาฉบับเต็ม ≥1,000 คำ\"}"
        )
    else:
        # ── Answer mode: pure Q&A — plain text, no edit action ───────────────
        if has_context:
            knowledge_instruction = (
                "ใช้บริบทจากเอกสารด้านล่างเป็นหลักในการตอบ "
                "หากบริบทไม่เพียงพอ ให้ขึ้นต้นคำตอบด้วย "
                "\"⚠️ หมายเหตุ: ข้อมูลนี้มาจากความรู้ทั่วไปของ AI "
                "เนื่องจากไม่พบข้อมูลในเอกสารหรือโน้ตของคุณ\""
            )
        else:
            knowledge_instruction = (
                "ไม่มีเอกสารถูกโหลดไว้ — ตอบจากความรู้ทั่วไปของ AI ได้เลย"
            )

        system_prompt = (
            "คุณเป็นผู้ช่วยวิจัย ทำหน้าที่ตอบคำถามอย่างตรงประเด็นและถูกต้อง\n"
            f"{knowledge_instruction}\n"
            f"{language_instruction}\n\n"
            "แนวทางการตอบ:\n"
            "- ตอบตรงคำถาม ชัดเจน ครบถ้วน\n"
            "- อ้างอิงเนื้อหาจากเอกสารเมื่อเกี่ยวข้อง\n"
            "- ตอบเป็น plain text ธรรมดา ไม่ต้องมี JSON\n"
            "- ห้ามสร้างหรือแก้ไขเอกสาร — ตอบคำถามเพียงอย่างเดียว"
        )

    user_parts = [f"คำถาม: {query}"]
    if context_text:
        user_parts.append(f"=== บริบทจากเอกสาร ===\n{context_text}")
    # ส่ง editor_content เฉพาะ research mode — ตัด tail 1,500 chars เพื่อลด tokens
    if research_mode and editor_content and editor_content.strip():
        editor_tail = editor_content.strip()[-1500:]
        user_parts.append(f"=== เนื้อหาในตัวแก้ไขปัจจุบัน (ส่วนท้าย) ===\n{editor_tail}")

    # Build messages: system + chat history (last 4, each capped at 400 chars) + current user message
    # Capping history prevents long research-mode answers from bloating the context window
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        for msg in chat_history[-4:]:
            role = msg.get("role", "user")
            content = _THINK_RE.sub('', msg.get("content", "")).strip()[:400]
            if content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    # ── Step 2: Call API ───────────────────────────────────────────────────────
    # OpenThaiGPT context window = 16,384 tokens (input + output).
    # Reserve headroom for input tokens to avoid 400 errors.
    api_max_tokens = 8192 if research_mode else 2048
    api_temperature = 0.65 if research_mode else 0.3
    try:
        raw, ri, ro = _call_api(messages, api_key, max_tokens=api_max_tokens, temperature=api_temperature)
        total_input += ri
        total_output += ro
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        if status in (401, 403):
            raise ValueError(f"❌ API key invalid or unauthorized (HTTP {status}).")
        elif status == 429:
            raise ValueError("❌ API quota exceeded (429). รอสักครู่แล้วลองใหม่")
        else:
            raise ValueError(f"❌ HTTP error {status}: {e}")
    except requests.RequestException as e:
        raise ValueError(f"❌ Request failed: {e}")

    # ── Step 3: Parse response ─────────────────────────────────────────────────
    think_matches = re.findall(r'<think>.*?</think>', raw, re.DOTALL)
    think_prefix = '\n'.join(think_matches).strip()
    raw_clean = _THINK_RE.sub('', raw).strip()

    # ── Answer mode: plain text — no JSON parsing needed ─────────────────────
    if not research_mode:
        response_text = raw_clean
        if not has_context and not response_text.startswith("⚠️"):
            response_text = _PARAMETRIC_WARNING + "\n\n" + response_text
        if think_prefix:
            response_text = (think_prefix + "\n\n" + response_text).strip()
        return "chat", response_text, None, total_input, total_output

    # ── Research mode: parse JSON → push editor_content to editor ────────────
    parsed = _extract_json(raw_clean)
    if parsed is None:
        # JSON parse failed — try regex extraction of editor_content
        editor_match = re.search(
            r'"editor_content"\s*:\s*"((?:[^"\\]|\\.)*)"',
            raw_clean, re.DOTALL
        )
        if editor_match:
            editor_text = editor_match.group(1)
            try:
                editor_text = json.loads(f'"{editor_text}"')
            except (json.JSONDecodeError, ValueError):
                pass
        else:
            editor_text = raw_clean  # fallback: full text to editor

        response_text = "🔬 ผลการค้นคว้าถูกส่งไปยัง Research Workbench แล้ว"
        if think_prefix:
            response_text = (think_prefix + "\n\n" + response_text).strip()
        return "research", response_text, editor_text, total_input, total_output

    response_text = str(parsed.get("response", "")).strip()
    if think_prefix:
        response_text = (think_prefix + "\n\n" + response_text).strip()

    new_editor = parsed.get("editor_content")
    if isinstance(new_editor, str):
        new_editor = new_editor.strip() or None
    else:
        new_editor = None

    # Safety net: LLM forgot editor_content → use response text
    if not new_editor and response_text:
        new_editor = _THINK_RE.sub('', response_text).strip() or None

    # Must have content to be a research action
    if not new_editor:
        return "chat", response_text, None, total_input, total_output

    return "research", response_text, new_editor, total_input, total_output


def generate_section(topic, section_instruction, retrieved_docs=None,
                     existing_content=""):
    """
    Generate a single section of long-form content to APPEND to the editor.
    Used by the Section-by-Section mode for building documents incrementally.

    Args:
        topic: The overall document topic
        section_instruction: What to write in this section
        retrieved_docs: Optional RAG context
        existing_content: Current editor content (for continuity)

    Returns:
        tuple: (section_text, input_tokens, output_tokens)
    """
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("OPENTHAI_API_KEY")
    if not api_key:
        raise ValueError("OPENTHAI_API_KEY not found")

    context_text = ""
    _MAX_DOC_CHARS = 1500
    _MAX_CONTEXT_CHARS = 6000
    if retrieved_docs:
        parts = [doc.page_content[:_MAX_DOC_CHARS] for doc in retrieved_docs]
        context_text = "\n\n".join(parts)[:_MAX_CONTEXT_CHARS]

    system_prompt = (
        "คุณเป็นนักเขียนวิชาการผู้เชี่ยวชาญ เขียนเนื้อหาทีละ section\n"
        "ภาษา: ตอบภาษาเดียวกับคำสั่ง (default ไทย), technical terms ใช้อังกฤษได้\n\n"
        "กฎการเขียน:\n"
        "- เขียนเฉพาะส่วนที่ได้รับมอบหมาย ห้ามเขียนส่วนอื่นหรือซ้ำกับเนื้อหาเดิม\n"
        "- ความยาวขั้นต่ำ 400 คำ, ≥4 ย่อหน้า — ขยายความละเอียด ยกตัวอย่าง อ้างทฤษฎี อธิบายกลไก\n"
        "- ตอบเป็น plain text เท่านั้น ห้าม JSON ห้ามคำอธิบายเพิ่ม"
    )

    user_parts = [f"หัวข้อเอกสาร: {topic}", f"ส่วนที่ต้องเขียน: {section_instruction}"]
    if context_text:
        user_parts.append(f"=== บริบทจากเอกสาร ===\n{context_text}")
    if existing_content and existing_content.strip():
        # Send only the last 1500 chars for continuity context
        tail = existing_content.strip()[-1500:]
        user_parts.append(f"=== เนื้อหาที่เขียนไปแล้ว (ส่วนท้าย) ===\n{tail}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]
    content, input_tokens, output_tokens = _call_api(
        messages, api_key, max_tokens=4096, temperature=0.65
    )
    content = _THINK_RE.sub('', content).strip()
    return content, input_tokens, output_tokens


def generate_selection_edit(selected_text, instruction):
    """
    Edit a selected piece of text according to user instruction.
    Returns (edited_text, input_tokens, output_tokens).
    """
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("OPENTHAI_API_KEY")
    if not api_key:
        raise ValueError("OPENTHAI_API_KEY not found")

    system_prompt = (
        "คุณเป็นผู้ช่วยแก้ไขข้อความ ผู้ใช้จะส่งข้อความที่เลือกไว้ "
        "พร้อมคำสั่งว่าต้องการแก้ไขอย่างไร\n"
        "ตอบเฉพาะข้อความที่แก้ไขแล้วเท่านั้น ห้ามมีคำอธิบายหรือข้อความเพิ่มเติม "
        "ห้ามใส่เครื่องหมายคำพูดครอบข้อความ\n"
        "แก้ไขโดยรักษาภาษาเดิมของข้อความ หากคำสั่งเป็นภาษาไทยให้ตอบเป็นภาษาไทย"
    )
    _MAX_SELECTION_CHARS = 4000
    selected_text = selected_text[:_MAX_SELECTION_CHARS]
    user_msg = (
        f"ข้อความที่เลือก:\n\"\"\"\n{selected_text}\n\"\"\"\n\n"
        f"คำสั่งแก้ไข: {instruction}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    content, input_tokens, output_tokens = _call_api(
        messages, api_key, max_tokens=2048, temperature=0.3
    )
    content = _THINK_RE.sub('', content).strip()
    return content, input_tokens, output_tokens


def generate_insertion(context_before, context_after, instruction):
    """
    Generate text to insert at a cursor position based on surrounding context.
    Returns (inserted_text, input_tokens, output_tokens).
    """
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("OPENTHAI_API_KEY")
    if not api_key:
        raise ValueError("OPENTHAI_API_KEY not found")

    system_prompt = (
        "คุณเป็นผู้ช่วยเขียนข้อความ ผู้ใช้จะระบุบริบทก่อนและหลังตำแหน่งที่ต้องการแทรก "
        "พร้อมคำสั่งว่าต้องการแทรกข้อความอะไร\n"
        "ตอบเฉพาะข้อความที่ต้องแทรกเท่านั้น ห้ามมีคำอธิบายหรือข้อความเพิ่มเติม "
        "ห้ามใส่เครื่องหมายคำพูดครอบข้อความ\n"
        "เขียนให้เข้ากับบริบทรอบข้าง รักษาภาษาเดิม หากคำสั่งเป็นภาษาไทยให้ตอบเป็นภาษาไทย"
    )
    before_preview = context_before[-300:] if len(context_before) > 300 else context_before
    after_preview = context_after[:300] if len(context_after) > 300 else context_after
    user_msg = (
        f"บริบทก่อนตำแหน่งแทรก:\n\"\"\"\n{before_preview}\n\"\"\"\n\n"
        f"บริบทหลังตำแหน่งแทรก:\n\"\"\"\n{after_preview}\n\"\"\"\n\n"
        f"คำสั่ง: {instruction}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    content, input_tokens, output_tokens = _call_api(
        messages, api_key, max_tokens=2048, temperature=0.3
    )
    content = _THINK_RE.sub('', content).strip()
    return content, input_tokens, output_tokens


def generate_answer_stream(query: str, retrieved_docs: list,
                           chat_history: Optional[list] = None,
                           editor_content: Optional[str] = None) -> Generator[str, None, None]:
    """
    Streaming variant of generate_answer for plain chat mode (non-research).

    Yields tokens as they arrive from the OpenThaiGPT API so that Streamlit's
    st.write_stream() can render them incrementally, dramatically improving
    perceived latency.

    Only supports plain chat mode — research mode requires structured JSON
    output and cannot be streamed reliably.

    Args:
        query: User message
        retrieved_docs: RAG context documents (may be empty)
        chat_history: Previous messages for context
        editor_content: Current editor text (unused in chat mode but kept for API symmetry)

    Yields:
        str: Token chunks

    Raises:
        ValueError: On API key error or HTTP 4xx errors
    """
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("OPENTHAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENTHAI_API_KEY not found. "
            f"Please set OPENTHAI_API_KEY in {env_path.resolve()}"
        )

    has_context = bool(retrieved_docs)
    _MAX_DOC_CHARS = 1500
    _MAX_CONTEXT_CHARS = 6000
    if retrieved_docs:
        parts = [doc.page_content[:_MAX_DOC_CHARS] for doc in retrieved_docs]
        context_text = "\n\n".join(parts)[:_MAX_CONTEXT_CHARS]
    else:
        context_text = ""

    language_instruction = (
        "ภาษา: ตอบภาษาเดียวกับคำถาม (default ไทย), technical terms ใช้อังกฤษได้"
    )

    if has_context:
        knowledge_instruction = (
            "ใช้บริบทจากเอกสารด้านล่างเป็นหลักในการตอบ "
            "หากบริบทไม่เพียงพอ ให้ขึ้นต้นคำตอบด้วย "
            "\"⚠️ หมายเหตุ: ข้อมูลนี้มาจากความรู้ทั่วไปของ AI\""
        )
    else:
        knowledge_instruction = (
            "ไม่มีเอกสารถูกโหลดไว้ — ตอบจากความรู้ทั่วไปของ AI ได้เลย"
        )

    system_prompt = (
        "คุณเป็นผู้ช่วยวิจัย ทำหน้าที่ตอบคำถามอย่างตรงประเด็นและถูกต้อง\n"
        f"{knowledge_instruction}\n"
        f"{language_instruction}\n\n"
        "แนวทางการตอบ:\n"
        "- ตอบตรงคำถาม ชัดเจน ครบถ้วน\n"
        "- ตอบเป็น plain text ธรรมดา ไม่ต้องมี JSON"
    )

    user_parts = [f"คำถาม: {query}"]
    if context_text:
        user_parts.append(f"=== บริบทจากเอกสาร ===\n{context_text}")

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        for msg in chat_history[-4:]:
            role = msg.get("role", "user")
            content = _THINK_RE.sub('', msg.get("content", "")).strip()[:400]
            if content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    try:
        yield from _call_api_stream(messages, api_key, max_tokens=2048, temperature=0.3)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        if status in (401, 403):
            raise ValueError(f"API key invalid or unauthorized (HTTP {status}).")
        elif status == 429:
            raise ValueError("API quota exceeded (429). รอสักครู่แล้วลองใหม่")
        else:
            raise ValueError(f"HTTP error {status}: {e}")
    except requests.RequestException as e:
        raise ValueError(f"Request failed: {e}")


def print_generated_answer(query, answer):
    """
    Pretty-print the AI-generated answer (CLI use).
    """
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + "🤖 AI-GENERATED ANSWER".center(78) + "║")
    print("╠" + "═" * 78 + "╣")
    print(f"║ Question: {query}")
    print("║" + "─" * 78 + "║")
    print("║ Answer:")
    print("║" + "─" * 78 + "║")
    for line in answer.split('\n'):
        if line.strip():
            wrapped_lines = textwrap.wrap(line, width=70)
            for wrapped_line in wrapped_lines:
                print(f"║ {wrapped_line:<76} ║")
        else:
            print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝\n")
