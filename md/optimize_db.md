# คุณคือ LLM AI Engiineer มืออาชีพ เชี่ยวชาญการใช้ python และ Library Langchain, chromaDB, streamlit
## context file หลัก: database.py, rag_pipeline.py, vector_store.py
## Rule : พยายามอย่าทำการแก้ไข file อื่นที่ไม่เกี่ยวข้องเพื่อหลีกเลี่ยงการ crash ของ system หลัก แต่สามารถทำได้หากต้องใช้ในการ implement module นี้
# Optimize ระบบ vector database และ RAG ให้เป็นตามนี้
1. เก็บรวม Collection เดียว แต่แยกด้วย Metadata 
แทนที่จะสร้าง Collection แยกกัน (notes และ docs) ให้เก็บทุกอย่างไว้ใน Collection เดียวของโปรเจกต์นั้นๆ แล้วใช้ Metadata เป็นตัวแบ่งหมวดหมู่ แทน:

JSON
{
  "source_type": "note", // หรือ "document"
  "user_id": "u123",
  "project_id": "p456",
  "created_at": "2026-03-14"
}

2. เทคนิคการเก็บให้ Retrieval มีประสิทธิภาพสูงสุด (Advanced RAG) : 
เพื่อให้อ่านเปเปอร์ได้เก่ง และดึงข้อมูลมาตอบได้ตรงประเด็น โปรดใช้เทคนิคเหล่านี้

    (1) Rich Metadata (ฝังบริบทให้ Document): เวลาหั่น Document ลง ChromaDB อย่าใส่แค่ Text ระบุ Metadata พ่วงไปด้วยเสมอ เช่น paper_title, authors, year, section (เช่น abstract, methodology, conclusion)
    
    (2) Parent-Child Chunking (เก็บเล็ก ดึงใหญ่): ให้หั่นประโยคย่อยๆ (Child) เป็น Vector เก็บไว้ใน ChromaDB เพื่อใช้ "ค้นหา" แต่ใน Metadata ให้ผูก ID หรือเชื่อมไปยังเนื้อหาทั้งย่อหน้า/ทั้งหน้า (Parent) เมื่อค้นหาเจอประโยคที่ตรง ให้ดึงเนื้อหาก้อนใหญ่ส่งไปให้ AI ประมวลผล

    (3) Summary Embedding (ให้ AI สรุปก่อนเก็บ): ก่อนจะเอา Document ทั้งหน้าไปหั่นเก็บ ให้สั่งรัน AI สร้าง "บทสรุปย่อ" ของหน้านั้นๆ แล้วเอาบทสรุปนั้นไปทำ Vector Search ด้วย

3. Chunk Handling : chunking ให้เหมาะสมกับความยาวของเนื้อหา
     