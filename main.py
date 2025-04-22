from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from AI.connectLLM import ConnectLLM
from typing import Optional
import uvicorn

app = FastAPI(
    title="AI RAG API",
    description="API for text and image processing using LLM",
    version="1.0.0"
)

class TextRequest(BaseModel):
    prompt: str

class ImageRequest(BaseModel):
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
    try:
        result = ConnectLLM.senTextandImage(request.prompt, request.image)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
