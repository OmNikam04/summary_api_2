import requests
from dotenv import load_dotenv
import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
import re
load_dotenv()
API_URL = os.getenv("API_URL")
hug_api_token = os.getenv("HUGGING_FACE_API_KEY")

app = FastAPI()

class RequestBody(BaseModel):
    text: str

headers = {"Authorization": f"Bearer {hug_api_token}"}


@app.get('/')
def index():
    return {
        "message": "summary with huggingface"
    }


@app.post('/api/v1/summerize')
async def get_summary(request: Request, payload: RequestBody):
    
    data = payload.dict()
    text = data["text"]
    clean_text = re.sub(r'(?<=\S)\s+(?=\S)', ' ', text)
    response = requests.post(API_URL, headers=headers, json={
                             "inputs": clean_text, "parameters": {"min_length": 50, "max_length": 100, "top_k": 4}})
    return response.json()[0]

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, timeout=120)
