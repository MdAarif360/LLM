import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

def get_openai_api_key():
    # 1. Streamlit Cloud secrets
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]

    # 2. Local environment variable fallback
    return os.getenv("OPENAI_API_KEY")
    


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#load_dotenv()

def ask_llm(prompt):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text
