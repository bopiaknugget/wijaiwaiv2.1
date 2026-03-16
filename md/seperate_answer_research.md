# คุณคือ LLM AI engineer มืออาชีพ มีความเชียวชาญ Langchain และ ระบบ RAG
    ## ความเชี่ยวชาญในการ coding : ถนัดการใช้ภาษา python ร่วมกับ streamlit ui  
    ## หน้าที่ที่ต้องทำ : เพื่อเสถียรภาพในการตอบคำถามของ AI ให้แยก answer mode กับ research mode ให้ชัดเจนโดยทำดังนี้
    ## Frontend : 
        - สร้าง tab 2 tab ใน panel ขวา การทำงานของ Assistant เป็น answer mode และ reserach mode
        - แต่ละ tab แจ้ง user ว่าทำงานยังไง โดยส่วนที่แจ้ง expand , collapse ได้
    ## Backend : การ implement ของแต่ละ mode เป็นดังนี้
        - answer mode : ตอบคำถาม user สั้น ๆ ในช่องแชท 
        - research mode : ตอบคำถามแบบยาว สร้างบทความหรือบทวิจัย ลง editor research workbench
    ## Interface : ต้องดูง่ายไม่รกตา ใช้งานง่าย 
    ## หมายเหตุ : คงโครงสร้าง prompt และ structure ของ code ของ application เดิมไว้, implement สลับ mode ง่ายขึ้น