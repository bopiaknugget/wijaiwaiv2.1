"""
Research Reviewer Module — Strict Advisor Persona
Reviews research writing quality as a rigorous thesis advisor,
providing color-coded feedback (red/green/yellow).
"""

import os
import re
import requests
from pathlib import Path
from dotenv import load_dotenv

API_URL = "http://thaillm.or.th/api/openthaigpt/v1/chat/completions"
MODEL = "/model"
_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def _call_api(messages, api_key, max_tokens=4096, temperature=0.3):
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
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


ADVISOR_SYSTEM_PROMPT = """\
คุณคือ "อาจารย์ที่ปรึกษาวิจัย" เข้มงวด ประสบการณ์ 20+ ปี ตรวจงานระดับ ป.ตรี-เอก
ให้ feedback ชี้จุดแข็ง/จุดอ่อน น้ำเสียงเข้มงวดแต่ให้กำลังใจ

เกณฑ์ตรวจ (เฉพาะบทที่ปรากฏในงาน):
บท1: ความสำคัญของปัญหา, วัตถุประสงค์, สมมติฐาน, ขอบเขต, คำนิยาม
บท2: ทฤษฎี, งานวิจัยที่เกี่ยวข้อง, กรอบแนวคิด, การสังเคราะห์วรรณกรรม
บท3: รูปแบบวิจัย, กลุ่มตัวอย่าง, เครื่องมือ, การเก็บข้อมูล, สถิติ
บท4: การนำเสนอผล, ตาราง/แผนภูมิ, การแปลผล
บท5: สรุปผล, อภิปรายผล, ข้อเสนอแนะ

รูปแบบการตอบ — แต่ละข้อระบุ tag:
- [ต้องแก้ไข] ปัญหาร้ายแรง พร้อมวิธีแก้
- [ดีแล้ว] ส่วนที่ถูกต้องน่าชื่นชม
- [คำแนะนำ] ข้อเสนอแนะเพิ่มเติม

เริ่มด้วยสรุปภาพรวมสั้นๆ → review ทีละประเด็น → ปิดด้วยสรุปคะแนนภาพรวม
"""


def review_research(content: str, user_focus: str = "") -> tuple:
    """
    Review research content as a strict thesis advisor.

    Args:
        content: The research text from the workbench editor.
        user_focus: Optional user instruction about what to focus the review on.

    Returns:
        (review_text, input_tokens, output_tokens)
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

    _MAX_REVIEW_CHARS = 8000
    content = content[:_MAX_REVIEW_CHARS]
    user_parts = [f"=== งานวิจัยที่ต้อง Review ===\n{content}"]
    if user_focus and user_focus.strip():
        user_parts.append(
            f"\n=== สิ่งที่นักศึกษาอยากให้เน้น Review เป็นพิเศษ ===\n{user_focus.strip()}"
        )

    messages = [
        {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]

    raw, input_tokens, output_tokens = _call_api(
        messages, api_key, max_tokens=4096, temperature=0.3
    )

    # Strip <think> tags if present
    review_text = _THINK_RE.sub('', raw).strip()
    return review_text, input_tokens, output_tokens
