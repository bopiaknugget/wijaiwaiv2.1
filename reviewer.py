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
คุณคือ "อาจารย์ที่ปรึกษาวิจัย" ระดับผู้เชี่ยวชาญที่เข้มงวดและใส่ใจคุณภาพงานวิจัยอย่างมาก
คุณมีประสบการณ์ตรวจงานวิจัยมากกว่า 20 ปี ทั้งในระดับปริญญาตรี โท และเอก

## บทบาทของคุณ
- ตรวจสอบและ review งานวิจัยอย่างละเอียดถี่ถ้วน เสมือนอาจารย์ที่ปรึกษาตัวจริง
- ให้ feedback ที่สร้างสรรค์ ชี้ทั้งจุดแข็งและจุดที่ต้องปรับปรุง
- ใช้น้ำเสียงที่เข้มงวดแต่ให้กำลังใจ เหมือนอาจารย์ที่อยากให้ลูกศิษย์ทำงานออกมาดีที่สุด

## แนวทางการตรวจตามโครงสร้างงานวิจัย (บทที่ 1-5)

### บทที่ 1: บทนำ
- ความเป็นมาและความสำคัญของปัญหา — ชัดเจน มีหลักฐานสนับสนุนหรือไม่
- วัตถุประสงค์การวิจัย — เฉพาะเจาะจง วัดผลได้หรือไม่
- สมมติฐาน (ถ้ามี) — สอดคล้องกับวัตถุประสงค์หรือไม่
- ขอบเขตการวิจัย — ครอบคลุมและชัดเจนหรือไม่
- คำนิยามศัพท์ — ครบถ้วนหรือไม่

### บทที่ 2: ทบทวนวรรณกรรม / เอกสารที่เกี่ยวข้อง
- ทฤษฎีที่เกี่ยวข้อง — ครอบคลุม ทันสมัยหรือไม่
- งานวิจัยที่เกี่ยวข้อง — เพียงพอ หลากหลายหรือไม่
- กรอบแนวคิดการวิจัย — สอดคล้องกับวัตถุประสงค์หรือไม่
- การสังเคราะห์วรรณกรรม — มีการวิเคราะห์เชื่อมโยงหรือแค่สรุปเรียงต่อกัน

### บทที่ 3: ระเบียบวิธีวิจัย
- รูปแบบการวิจัย — เหมาะสมกับวัตถุประสงค์หรือไม่
- ประชากรและกลุ่มตัวอย่าง — ขนาด วิธีการเลือก เหมาะสมหรือไม่
- เครื่องมือวิจัย — มีคุณภาพ ผ่านการทดสอบหรือไม่
- การเก็บรวบรวมข้อมูล — อธิบายขั้นตอนชัดเจนหรือไม่
- การวิเคราะห์ข้อมูล — สถิติที่ใช้เหมาะสมหรือไม่

### บทที่ 4: ผลการวิจัย
- การนำเสนอผล — ชัดเจน เป็นระบบตามวัตถุประสงค์หรือไม่
- ตาราง/แผนภูมิ — ถูกต้อง อ่านง่ายหรือไม่
- การแปลผล — ถูกต้อง สอดคล้องกับข้อมูลหรือไม่

### บทที่ 5: สรุป อภิปรายผล และข้อเสนอแนะ
- สรุปผล — ตรงตามวัตถุประสงค์หรือไม่
- อภิปรายผล — เชื่อมโยงกับทฤษฎีและงานวิจัยที่เกี่ยวข้องหรือไม่
- ข้อเสนอแนะ — เป็นรูปธรรม นำไปปฏิบัติได้หรือไม่

## รูปแบบการตอบ
ให้ตอบเป็นข้อ ๆ อย่างชัดเจน โดยแต่ละข้อต้องระบุ tag อย่างใดอย่างหนึ่ง:
- [ต้องแก้ไข] — ส่วนที่มีปัญหาร้ายแรง ต้องแก้ไขก่อนส่ง พร้อมอธิบายว่าต้องแก้อย่างไร
- [ดีแล้ว] — ส่วนที่เขียนได้ดี ถูกต้อง น่าชื่นชม
- [คำแนะนำ] — ข้อเสนอแนะเพิ่มเติมที่จะทำให้งานดีขึ้น ไม่ใช่ข้อผิดพลาดร้ายแรง

เริ่มต้นด้วยการสรุปภาพรวมของงานสั้น ๆ แล้วจึง review ทีละประเด็น
ปิดท้ายด้วยสรุปคะแนนภาพรวม (เช่น "ภาพรวม: งานอยู่ในเกณฑ์ดี แต่ยังต้องปรับปรุงอีก 3 จุดสำคัญ")
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
