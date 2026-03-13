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


def generate_answer(query, retrieved_docs, chat_history=None, editor_content=None):
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

    Returns:
        tuple: (action, response_text, new_editor_content, input_tokens, output_tokens)
            action (str): "chat" or "edit"
            response_text (str): Message to display in chat
            new_editor_content (str | None): New editor text when action="edit", else None
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

    if has_context:
        knowledge_instruction = (
            "ใช้บริบทจากเอกสารเป็นหลักในการตอบหรือแก้ไข "
            "หากบริบทไม่เพียงพอและจำเป็นต้องใช้ความรู้ทั่วไปของ AI "
            "ให้ขึ้นต้น response ด้วย: "
            "\"⚠️ หมายเหตุ: ข้อมูลนี้มาจากฐานความรู้ทั่วไปของ AI "
            "เนื่องจากไม่พบข้อมูลในเอกสารหรือโน้ตของคุณ\""
        )
    else:
        knowledge_instruction = (
            "ไม่มีเอกสารหรือโน้ตถูกโหลดไว้ "
            "ให้ใช้ความรู้ทั่วไปของ AI ในการตอบและแก้ไขเอกสารได้เลย"
        )

    system_prompt = (
        "คุณเป็น Agentic Research Editor ที่ช่วยนักวิจัยทั้งในการตอบคำถามและแก้ไขเอกสาร\n\n"
        f"{knowledge_instruction}\n\n"
        "จำแนกคำสั่งของผู้ใช้เป็น:\n"
        "- \"chat\" = ถามคำถาม ขอข้อมูล ขอคำอธิบาย สรุปเนื้อหา\n"
        "- \"edit\" = สั่งให้แก้ไข เขียนใหม่ สร้าง ปรับปรุง จัดรูปแบบ หรือ generate เนื้อหาในตัวแก้ไข\n\n"
        "ตอบในรูปแบบ JSON เท่านั้น ห้ามมีข้อความอื่นนอก JSON:\n"
        "{\n"
        "  \"action\": \"chat\" หรือ \"edit\",\n"
        "  \"response\": \"ข้อความที่จะแสดงในช่องสนทนา (สำหรับ edit ให้อธิบายสั้นๆ ว่าทำอะไรไป)\",\n"
        "  \"editor_content\": \"เนื้อหาใหม่สำหรับตัวแก้ไข (ใส่เมื่อ action=edit เท่านั้น มิฉะนั้นใส่ null)\"\n"
        "}"
    )

    user_parts = [f"คำสั่ง/คำถาม: {contextual_query}"]
    if context_text:
        user_parts.append(f"=== บริบทจากเอกสาร ===\n{context_text}")
    if editor_content and editor_content.strip():
        user_parts.append(f"=== เนื้อหาในตัวแก้ไขปัจจุบัน ===\n{editor_content.strip()}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]

    # ── Step 3: Call API ───────────────────────────────────────────────────────
    try:
        raw, ri, ro = _call_api(messages, api_key, max_tokens=3000, temperature=0.3)
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

    # ── Step 4: Parse structured response ─────────────────────────────────────
    parsed = _extract_json(raw)
    if parsed is None:
        # Fallback: treat raw response as a plain chat reply
        response_text = _THINK_RE.sub('', raw).strip()
        if not has_context and not response_text.startswith("⚠️"):
            response_text = _PARAMETRIC_WARNING + "\n\n" + response_text
        return "chat", response_text, None, total_input, total_output

    action = str(parsed.get("action", "chat")).lower()
    if action not in ("chat", "edit"):
        action = "chat"

    response_text = str(parsed.get("response", "")).strip()

    new_editor = parsed.get("editor_content")
    if isinstance(new_editor, str):
        new_editor = new_editor.strip() or None
    elif new_editor is not None:
        new_editor = None  # discard non-string values (e.g. null/None already fine)

    # Python-level safety net: always prepend warning when no RAG context was available
    if not has_context and response_text and not response_text.startswith("⚠️"):
        response_text = _PARAMETRIC_WARNING + "\n\n" + response_text

    # edit action must deliver content; demote to chat otherwise
    if action == "edit" and not new_editor:
        action = "chat"

    return action, response_text, new_editor, total_input, total_output


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
