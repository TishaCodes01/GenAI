# healthcare_api.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# from health_chatbot import MedBot

# # Initialize FastAPI app
# app = FastAPI()

# # Initialize the chatbot
# chatbot = MedBot()

# # Request model for user input
# class ChatRequest(BaseModel):
#     user_input: str

# # API Endpoint to get chatbot response
# @app.post("/chat")
# async def chat(request: ChatRequest):
#     response = chatbot.generate_response(request.user_input)
#     return {"response": response}

from fastapi import FastAPI, Body
from fastapi.responses import PlainTextResponse
from gemini_health_chatbot import MedBot

# Initialize FastAPI app
app = FastAPI()

# Initialize the chatbot
chatbot = MedBot()

# API Endpoint to get chatbot response
@app.post("/chat", response_class=PlainTextResponse)
async def chat(user_input: str = Body(..., media_type="text/plain")):
    """
    Receives user input as plain text and returns chatbot response.
    """
    response = chatbot.generate_response(user_input)
    return response