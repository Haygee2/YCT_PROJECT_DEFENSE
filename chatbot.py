import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize conversation history
conversation_history = []

def chat_with_ai(query):
    """AI Chatbot interaction using OpenRouter API."""
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        return "API key not found."

    # Add user query to conversation history
    conversation_history.append({"role": "user", "content": query})

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
                "X-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
            },
            data=json.dumps({
                "model": "deepseek/deepseek-r1:free",  # Updated model
                "messages": conversation_history,
            })
        )
        response_data = response.json()
        ai_response = response_data['choices'][0]['message']['content'].strip()

        # Add AI response to conversation history
        conversation_history.append({"role": "assistant", "content": ai_response})

        return ai_response
    except Exception as e:
        return f"An error occurred: {e}"