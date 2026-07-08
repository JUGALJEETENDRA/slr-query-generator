import os

from dotenv import load_dotenv

load_dotenv()


def ask_gemini(prompt, model="gemini-2.5-flash", api_key=None):
    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_key:
        raise RuntimeError("Gemini API key was not provided.")

    from google import genai

    client = genai.Client(
        api_key=resolved_key
    )

    response = client.models.generate_content(
        model=model if model and model != "gemini" else "gemini-2.5-flash",
        contents=prompt
    )

    return response.text
