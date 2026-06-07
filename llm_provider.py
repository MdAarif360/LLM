import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def ask_llm(prompt):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text