import os
import streamlit as st
from openai import OpenAI


def _get_api_key() -> str:
    # Streamlit Cloud secrets take priority
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")


def ask_llm(prompt: str) -> str:
    client = OpenAI(api_key=_get_api_key())
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        input=prompt,
    )
    return response.output_text
