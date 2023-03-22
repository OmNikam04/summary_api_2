import requests
from dotenv import load_dotenv
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
app = FastAPI()

# Set up CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
API_URL = os.getenv("API_URL")
hug_api_token = os.getenv("HUGGING_FACE_API_KEY")


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
    try:
        data = payload.dict()
        text = data["text"]
        text_len = len(text)
        # text = text.replace("'", "'\'")
        # Adjust minL based on the length of the input text
        minL = 10
        if text_len < 100:
            minL = 10
        elif text_len < 500:
            minL = 50
        elif text_len < 2000:
            minL = 100
        else:
            minL = 140

        inputs = {
            "inputs": text,
            "parameters": {
                "min_length": minL,
                "max_length": 300,
                "top_k": 40
            },
        }
        response = requests.post(API_URL, headers=headers, json=inputs)
        return response.json()[0]["summary_text"]

    except ValidationError as e:
        # Handle validation error in the request body
        return {"detail": e.errors()}

    except requests.exceptions.RequestException as e:
        # Handle any error that occurs during the request (e.g. network error, timeout)
        return {"detail": str(e)}

    except Exception as e:
        # Handle any other unexpected error
        return {"detail": str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, timeout=120)
