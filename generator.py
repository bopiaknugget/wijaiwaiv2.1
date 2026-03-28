"""
Response Generation Module – Agentic Editor Edition
Handles AI-powered answer generation and editor manipulation using OpenThaiGPT API.
"""

import json
import os
import re
import textwrap
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

    # ── Step 1: Context-aware query re-phrasing ────────────────────────────────
    contextual_query = query
    if chat_history:
        history_text = "\n".join([
            f"{msg['role']}: {_THINK_RE.sub('', msg['content']).strip()}"
            for msg in chat_history[-6:]
        ])
        rephrase_prompt = (
            f"จากประวัติการสนทนา:\n{history_text}\n\n"
            f"คำถาม/คำสั่งล่าสุดของผู้ใช้: \"{query}\"\n\n"
            "ถ้าคำถามอ้างอิงถึงสิ่งที่พูดคุยมาก่อนหน้านี้ ให้ re-phrase ให้ชัดเจนขึ้น "
            "ถ้าไม่ ให้ใช้คำถามเดิม\n\nคำถามที่ re-phrase แล้ว:"
        )
        try:
            rephrased, ri, ro = _call_api(
                [
                    {"role": "system", "content": "คุณเป็นผู้ช่วย re-phrase คำถามให้ชัดเจนตามบริบท"},
                    {"role": "user", "content": rephrase_prompt},
                ],
                api_key,
                max_tokens=256,
                temperature=0.1,
            )
            contextual_query = rephrased.strip()
            total_input += ri
            total_output += ro
        except Exception:
            contextual_query = query

    # ── Step 2: Build main generation prompt ──────────────────────────────────
    has_context = bool(retrieved_docs)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs) if retrieved_docs else ""

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

    user_parts = [f"คำถาม: {contextual_query}"]
    if context_text:
        user_parts.append(f"=== บริบทจากเอกสาร ===\n{context_text}")
    # ส่ง editor_content เฉพาะ research mode เพื่อให้ AI รู้เนื้อหาปัจจุบัน
    if research_mode and editor_content and editor_content.strip():
        user_parts.append(f"=== เนื้อหาในตัวแก้ไขปัจจุบัน ===\n{editor_content.strip()}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]

    # ── Step 3: Call API ───────────────────────────────────────────────────────
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

    # ── Step 4: Parse response ─────────────────────────────────────────────────
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
    if retrieved_docs:
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

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
