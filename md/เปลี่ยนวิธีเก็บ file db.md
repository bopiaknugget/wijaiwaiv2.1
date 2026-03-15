# คุณคือ AI engineer เชี่ยวชาญการทำ RAG ด้วย chromaDB, Langchain และ SQL3lite
## ก่อนเริ่มทำ : ตรวจสอบ logic การเก็บ ข้อมูลลง data base ของ app เพื่อให้เข้าใจว่าระบบเดิมทำงานยังไง
1. simplify and enhance database system : ทำการเก็บ database ให้เป็นระเบียบ โดยการเก็บใน folder ชื่แ Database ข้างในเก็บ record ของ vector db และ sql lite 
2. (หากว่าส่วนนี้ยังไม่มีการ implement) ให้ทั้ง doc, note และ web ให้เก็บใน collection เดียวกันเพื่อประสิทธิภาพในการทำ retrieveal โดยให้แยก record ด้วย metadata  Expected output : doc_id, note_id และ web_id 
3. ตรวจสอบ bug ในการใช้ข้อมูลในระบบดังนี้
    - ตรวจสอบว่าหากมีข้อมูลอยู่แล้วเวลาเปิด app ข้อมูลถูก load  และแสดงในหน้า Frontend UI อย่างถูกต้องหรือไม่
    - การเพิ่ม record ใน database เก็บถูกต้องหรือไม่ : การเก็บข้อมูลผ่าน interface แบ่งเป็น 3 ประเภท doc, note และ web ตรวจสอบว่าตอนเก็บมีการบันทึก metadata อย่างถูกต้องหรือไม่
    - การลบ record ใน database ถูกทำอย่างถูกต้องหรือไม่ : หลังจากกดปุ่มถังขยะหลัง record record นั้นต้องถูก clear จาก database และ update ใน app ที่เปิดอยู่ทันที
หมายเหตุ : รักษาความเรียบง่ายของ code เอาไว้ ให้อ่านเข้าใจง่าย และง่ายต่อการแก้ไขภายหลัง 