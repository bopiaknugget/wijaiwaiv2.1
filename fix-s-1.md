# Severity ระดับที่ 1 (critical)

# Check first : เชคว่า user ที่ Oauth จาก Google มา มีการเก็บอยู่ใน sqlite จริงหรือไม่. ขณะนี้ app มีการ log in แล้วด้วย user 2 คน คือ tanawatl.cs@gmail.com กับ richmantanawat@gmail.com 2 user นี้ต้องเจอใน db , ให้คุณ validate and verify ส่วนนี้ก่อน

## issue 1  (fixed): 
## context : databse related files
    - every user share same documents (maybe include other datas) big security problem that must be handled.

## issue 2  :
    - plain documents of that is 'save, save as' are gone when logged out and re-log in. 
    - Recorded documents (plain text document in editor that user saved)  in sqlite
    - ensure that this document **must** not be shared
## issue 3 (process done, still not test) : 
    - Duplicate docs in processed_docs: When re-uploading the same file, extend() adds it again       
  without checking if it already exists.
    - Duplicate key: Same doc name appears twice → same Streamlit key.

 
  