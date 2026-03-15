# คุณคือ Expert Senior Python Developer และ AI/RAG Specialist ที่มีความเชี่ยวชาญขั้นสูงในการพัฒนา Web Application ด้วย Streamlit, การทำ Web Scraping, และการจัดการ Vector Database ด้วย ChromaDB

# ฉันต้องการให้คุณช่วยเขียนโค้ดเพื่อเพิ่ม "Web Scraping Module" เข้าไปในโปรเจกต์ของฉัน โดยโมดูลนี้จะรับ Input เป็น URL เพียง 1 หน้าเว็บ ดึงข้อมูล สรุปความ ทำ Chunking และบันทึกลง ChromaDB

# ข้อกำหนดทางเทคนิค (Tech Stack):
  - Backend: Python (ใช้ไลบรารีที่เหมาะสมเช่น requests, BeautifulSoup หรือ trafilatura สำหรับดึงเนื้อหา)
  - AI Model: OpenThaiGPT (สมมติการเรียกใช้ผ่าน API แบบมาตรฐาน โดยมี endpoint และ api_key)
  - Vector DB: ChromaDB
  - Frontend: Streamlit

# กรุณาแบ่งการ Implement ออกเป็น Step ดังต่อไปนี้อย่างเคร่งครัด:

*Step 1: Backend & Data Processing (ทำส่วนนี้ก่อน)*
  - เขียน mouduleใหม่ Web Scraping (แยกไฟล์ออกมา) ที่สามารถดึง Text content ออกมาจาก URL ได้อย่างสะอาด
  - เพิ่มระบบ Error Handling แบบครอบจักรวาล ดักจับกรณีเว็บ Block (เช่น 403 Forbidden), Timeout, เครือข่ายล่ม, หรือ Scrape มาแล้วเนื้อหาว่างเปล่า (Empty content) โดยฟังก์ชันต้อง return ข้อความ Error ที่สื่อความหมายชัดเจนกลับไปที่ Frontend
  - เขียนฟังก์ชันเชื่อมต่อ OpenThaiGPT API เพื่อทำ 2 หน้าที่:
    1. สรุปข้อมูลที่ดึงมา โดยใช้ System Prompt ว่า "สรุปข้อมูลจากเนื้อหาที่กำหนดให้ โดยต้องคงรายละเอียดและข้อมูลสำคัญไว้อย่างครบถ้วน ไม่ตัดทอนสาระสำคัญ"
    2. ตั้งชื่อ Title ของเนื้อหานี้โดยอัตโนมัติ (Auto-generate Title)

*Step 2: Chunking & Database Integration*
  - นำเนื้อหาที่ผ่านการสรุปแล้ว มาทำ Embedding และ Chunking ในรูปแบบเดียวกับที่ระบบเดิมใช้อยู่ แต่เปลี่ยน metadata เพื่อบอกให้รู้ว่าข้อมูลมาจาก web

*Step 3: Streamlit Frontend (UI & UX)*
  - สร้างหน้า UI สำหรับรับ Input เป็น URL ออกแบบให้ดูสวยงาม ใช้งานง่าย (User-friendly) ดูสะอาดตา
  - ขณะที่ระบบกำลังทำงาน (Scraping -> Summarizing -> Embedding) ให้มี Loading/Spinner หรือ Progress Bar แสดงสถานะให้ผู้ใช้ทราบว่าระบบกำลังทำขั้นตอนไหนอยู่
  - หากเกิด Error จาก Step 1 ให้แสดง Alert/Error Message บนหน้า UI อย่างชัดเจนและสวยงาม (เช่น ใช้ st.error หรือ st.warning)
  - เมื่อบันทึกข้อมูลสำเร็จ ให้แสดง Success Message บน UI พร้อมแสดงผล "Title อัตโนมัติ" ที่ AI ตั้งให้
  - สำคัญ: ต้องมีฟังก์ชัน (เช่น text_input หรือ form) ให้ผู้ใช้สามารถพิมพ์แก้ไข (Edit) ชื่อ Title แบบ Manual ได้ในภายหลัง และเมื่อกดอัปเดต ให้ไปแก้ Metadata ใน ChromaDB ด้วย

# หมายเหตุ : กรุณาเขียนโค้ดให้มีโครงสร้างเป็นระเบียบ (Modular), มี Type Hinting, และมีคอมเมนต์ภาษาไทยอธิบายในจุดที่สำคัญ พร้อมคำแนะนำว่าฉันต้องเตรียม Environment Variables อะไรบ้าง (เช่น OPEN_THAIGPT_API_KEY) เพื่อให้โค้ดนี้รันได้จริง  