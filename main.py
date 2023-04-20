import requests
from dotenv import load_dotenv
import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import openai
import random
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
open_ai_api_key = os.getenv("OPEN_AI_API_KEY")
openai.api_key = open_ai_api_key


class RequestBody(BaseModel):
    text: str


headers = {"Authorization": f"Bearer {hug_api_token}"}

@app.get('/')
def index():
    return {
        "message": "summary with huggingface"
    }

def query(payload):
    response =  requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def get_summary(text: str) -> str:
    maxL = 1000
    minL = maxL // 4
    output = query({
        "inputs": text,
        "parameters": {"min_length": minL, "max_length": maxL},
        "options": {
            "use_cache": False,
            "wait_for_model": True
        }
    })
    return output[0]["summary_text"]

def generate_quiz(summary: str) -> dict:
    prompt = f"Your expertise as a quiz generator can be valuable to teachers. Given a summary of a session, your role is to generate a multiple choice quiz question with four options. The question should be related to the summary and have a clear correct answer. The output should include the question, the four options, correct answer and explanation. When provided with a summary, generate a quiz question that would be suitable for a classroom setting.And easily distinguish question, options and correct answer with explanation.\nHuman: {summary}\nAI:"

    response_2 = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    output = response_2.choices[0].text.strip()
    output = output.split("\n")
    output = list(filter(None, output))

    # Extract question, options, and correct answer
    question = output[0].strip()
    options = [option.strip() for option in output[1:5]]
    correct_answer = output[5].strip().split(":")
    correct_answer = correct_answer[1].strip()

    # Extract explanation
    explanation = output[6].strip().split(":")
    explanation = explanation[1].strip()

    # Convert into a dictionary
    quiz = {
        "question": question,
        "options": options,
        "correct_answer": correct_answer,
        "explanation": explanation
    }

    return quiz


@app.post('/api/v1/summerize')
async def get_summary_and_quiz(payload: RequestBody):
    try:
        data = payload.dict()
        text = data["text"]
        summary = get_summary(text)
        quiz = generate_quiz(summary)

        return {
            "summary": summary,
            "quiz": quiz
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, timeout=120)