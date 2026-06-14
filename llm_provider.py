import os
import base64
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


def _model(name_env: str, default: str) -> str:
    """Resolve a model name from st.secrets, then env, then default."""
    try:
        if name_env in st.secrets:
            return st.secrets[name_env]
    except Exception:
        pass
    return os.getenv(name_env, default)


def ask_llm(prompt: str) -> str:
    client = OpenAI(api_key=_get_api_key())
    response = client.responses.create(
        model=_model("OPENAI_MODEL", "gpt-4.1-mini"),
        input=prompt,
    )
    return response.output_text


def ask_llm_vision(prompt: str, images: list, detail: str = "high") -> str:
    """
    Send a prompt PLUS one or more page images to a vision-capable model.

    images : list of PNG/JPEG byte strings.
    Returns the model's text output.

    This is what lifts extraction quality to "reads the receipt like a human"
    level — the model sees the actual pixels instead of noisy OCR text.
    """
    client = OpenAI(api_key=_get_api_key())

    content = [{"type": "input_text", "text": prompt}]
    for img in images:
        b64 = base64.b64encode(img).decode("ascii")
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{b64}",
            "detail": detail,
        })

    response = client.responses.create(
        model=_model("OPENAI_VISION_MODEL", "gpt-4o"),
        input=[{"role": "user", "content": content}],
    )
    return response.output_text
