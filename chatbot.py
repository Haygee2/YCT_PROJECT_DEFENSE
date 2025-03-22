import os
from dotenv import load_dotenv
import openai

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
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
                "X-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
            },
            extra_body={},
            model="deepseek/deepseek-r1:free",
            messages=conversation_history
        )

        ai_response = response.choices[0].message.content.strip()

        # Add AI response to conversation history
        conversation_history.append({"role": "assistant", "content": ai_response})

        return ai_response
    except Exception as e:
        return f"Error: {e}"