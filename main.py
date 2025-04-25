from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from AI.connectLLM import ConnectLLM
from typing import Optional, List
import uvicorn
from swagger_config import custom_openapi  # Add this import
from model.vector import VectorDB

app = FastAPI(
    title="AI RAG API",
    description="API for text and image processing using LLM",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add this after FastAPI initialization
app.openapi = lambda: custom_openapi(app)

# class TextRequest(BaseModel):
#     prompt: str

class ImageRequest(BaseModel):
    """
    โมเดล Pydantic สำหรับจัดการคำขอรูปภาพพร้อมข้อความ

    คุณสมบัติ:
        prompt (str): ข้อความหรือคำอธิบายสำหรับรูปภาพ
        image (str): รูปภาพที่เข้ารหัสเป็น Base64 (ไม่ต้องมีคำนำหน้า 'data:')
    """
    prompt: Optional[str] = None
    image: Optional[str] = None

# @app.post("/process_text")
# async def process_text(request: TextRequest):
#     try:
#         print(request.prompt)
#         result = ConnectLLM.senText(request.prompt)
       
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_text")
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
        # Initialize image variable
        image = request.image if request.image else None
        text = request.prompt if request.prompt else " "
        result = ConnectLLM.senText(text, image)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

vector_db = VectorDB()

class ChatRequest(BaseModel):
    message: str
    search_similar: bool = False

@app.post("/chat")
async def process_chat(request: ChatRequest):
    """
    Process chat messages using vector search and LLM
    """
    try:
        # Search for similar messages first
        similar_messages = vector_db.search(request.message, k=3)
        
        # Check if we have relevant context
        relevant_messages = [
            msg for msg in similar_messages 
            if msg['similarity_score'] > 0.5
        ]
        
        if relevant_messages:
            # มีข้อมูลที่เกี่ยวข้องใน vector database
            context = "\n".join([
                f"Previous Q&A ({msg['metadata']['timestamp']}):\n" +
                f"Q: {msg['text']}\n" +
                f"A: {msg['metadata']['additional_info']['response']}"
                for msg in relevant_messages
            ])
            result = ConnectLLM.senText(request.message, None, context)
        else:
            # ไม่มีข้อมูลที่เกี่ยวข้อง ส่งตรงไปที่ LLM
            print("No relevant context found, sending directly to LLM")
            result = ConnectLLM.senText(request.message)
        
        # Store both question and answer
        vector_db.add_text(request.message, {
            'type': 'question',
            'response': result.get('response', ''),
            'has_context': bool(relevant_messages)
        })
        
        # Include debug info in response
        result['debug'] = {
            'used_context': bool(relevant_messages),
            'similar_messages_count': len(relevant_messages),
            'vector_db_size': len(vector_db.texts)
        }
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
