import os
from dotenv import load_dotenv
import openai
from PIL import Image
import pytesseract
import fitz  # PyMuPDF

# Load environment variables from .env file
load_dotenv()

# Initialize conversation history
conversation_history = []

def extract_text_from_file(file_path):
    """Extract text from a file (PDF or image)."""
    if file_path.endswith(".pdf"):
        text = ""
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        return text
    elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)
    else:
        return "Unsupported file format."

def chat_with_ai(query, file_path=None):
    """AI Chatbot interaction using OpenRouter API."""
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        return "API key not found."

    # If a file is provided, extract its text and append it to the query
    if file_path:
        file_text = extract_text_from_file(file_path)
        query += f"\n\nDocument Content:\n{file_text}"

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