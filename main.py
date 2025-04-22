from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from AI.connectLLM import ConnectLLM
from typing import Optional
import uvicorn
from swagger_config import custom_openapi  # Add this import

app = FastAPI(
    title="AI RAG API",
    description="API for text and image processing using LLM",
    version="1.0.0"
)

# Add this after FastAPI initialization
app.openapi = lambda: custom_openapi(app)

class TextRequest(BaseModel):
    prompt: str

class ImageRequest(BaseModel):
    """
    โมเดล Pydantic สำหรับจัดการคำขอรูปภาพพร้อมข้อความ

    คุณสมบัติ:
        prompt (str): ข้อความหรือคำอธิบายสำหรับรูปภาพ
        image (str): รูปภาพที่เข้ารหัสเป็น Base64 (ไม่ต้องมีคำนำหน้า 'data:')
    """
    prompt: str 
    image: str

@app.post("/process_text")
async def process_text(request: TextRequest):
    try:
        print(request.prompt)
        result = ConnectLLM.senText(request.prompt)
       
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_image")
async def process_image(request: ImageRequest):
    """
    ประมวลผลคำขอรูปภาพพร้อมข้อความและส่งคืนผลลัพธ์จากการวิเคราะห์ด้วย LLM

    พารามิเตอร์:
        request (ImageRequest): โมเดล Pydantic ที่ประกอบด้วย:
            - prompt (str): ข้อความหรือคำอธิบายสำหรับรูปภาพ
            - image (str): รูปภาพที่เข้ารหัสเป็น Base64

    ส่งคืน:
        dict: ผลลัพธ์การวิเคราะห์จาก LLM

    ข้อผิดพลาด:
        HTTPException: 
            - รหัส 400 หากการประมวลผล LLM ล้มเหลว
            - รหัส 500 สำหรับข้อผิดพลาดอื่นๆ
    """
    try:
        
        result = ConnectLLM.senTextandImage(request.prompt, request.image)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
