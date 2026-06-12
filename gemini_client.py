import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


def ask_gemini(prompt):
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text