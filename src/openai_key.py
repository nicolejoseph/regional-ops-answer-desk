def get_openai_api_key():
    import os

    try:
        import streamlit as st

        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY")
