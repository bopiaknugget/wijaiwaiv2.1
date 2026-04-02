# UI refactor
## context : app.py, generator.py, 
## 1. Show AI thought for all functions (สร้างเนื้อหาวิจัย, ตรวจงานวิจัย, เพิ่มข้อความ, แก้ไขข้อความ) and other if any.
## 2. change how to show token usage to 'input token, output token not just tokens' this data can retrieve from openThaiGPT result. And total (input and output) must be recored to user database.
## 3. Limit uploaded doc to 5 files and not over 2mbs per file to prevent system crash, also notify user if doc is over 2mbs, and check and handle cases that may cause upload crash.
## 4. Limit charracters in editor to prevent system overload token crash. Limit saved document to 20 files
## 5. The working file showing (font size) is too small, make it larger.  

**Remark** : Perform action with minimal changes to existing system  

กำลังทำงานกับไฟล์: file_name , must below Research workbench caption before Title with optimal gap between each caption