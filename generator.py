"""
Response Generation Module
Handles AI-powered answer generation using OpenThaiGPT API.
"""

import os
import re
import textwrap
import requests
from pathlib import Path
from dotenv import load_dotenv

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


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


def generate_answer(query, retrieved_docs, chat_history=None):
    """
    Generate an AI-powered answer using OpenThaiGPT API.
    Uses retrieved documents as context to provide accurate, Thai-language responses.
    Incorporates chat history for contextual question re-phrasing.

    Args:
        query (str): The user's question
        retrieved_docs (list): List of retrieved Document objects from vector store
        chat_history (list, optional): List of previous messages for context

    Returns:
        tuple: (answer_text, input_tokens, output_tokens) - The AI-generated answer in Thai and token usage

    Raises:
        ValueError: If API key is missing or generation fails
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

    # Re-phrase query using chat history for context
    contextual_query = query
    if chat_history:
        history_text = "\n".join([f"{msg['role']}: {_THINK_RE.sub('', msg['content']).strip()}" for msg in chat_history[-6:]])
        rephrase_prompt = f"""
จากประวัติการสนทนา:
{history_text}

คำถามล่าสุดของผู้ใช้: "{query}"

ถ้าคำถามอ้างอิงถึงสิ่งที่พูดคุยมาก่อนหน้านี้ ให้ re-phrase คำถามให้ชัดเจนขึ้น
ถ้าไม่ ให้ใช้คำถามเดิม

คำถามที่ re-phrase แล้ว:
"""
        rephrase_messages = [
            {"role": "system", "content": "คุณเป็นผู้ช่วย re-phrase คำถามให้ชัดเจนตามบริบท"},
            {"role": "user", "content": rephrase_prompt},
        ]

        try:
            rephrased, _, _ = _call_api(rephrase_messages, api_key)
            contextual_query = rephrased.strip()
        except Exception:
            contextual_query = query  # fallback to original query

    # System prompt in Thai
    system_prompt = (
        "คุณเป็นผู้ช่วยวิจัย ให้สรุปคำตอบอย่างชัดเจนและกระชับโดยใช้เฉพาะบริบทที่ให้มา "
        "ตอบเป็นภาษาไทย และหากเป็นไปได้ ให้อ้างอิงส่วนของบริบทที่ใช้ในการตอบคำถาม "
        "หากคำตอบไม่อยู่ในบริบท ให้บอกว่าคุณไม่รู้ตามเอกสารที่ให้มา"
    )

    # Combine retrieved documents into context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"คำถาม: {contextual_query}\n\nบริบท:\n{context}"},
    ]

    try:
        return _call_api(messages, api_key)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        if status == 401 or status == 403:
            raise ValueError(f"❌ API key invalid or unauthorized (HTTP {status}).")
        elif status == 429:
            raise ValueError(f"❌ API quota exceeded (429). รอสักครู่แล้วลองใหม่")
        else:
            raise ValueError(f"❌ HTTP error {status}: {e}")
    except requests.RequestException as e:
        raise ValueError(f"❌ Request failed: {e}")


def print_generated_answer(query, answer):
    """
    Pretty-print the AI-generated answer.

    Args:
        query (str): The original user query
        answer (str): The AI-generated answer
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
