"""Build chroma_db/ once locally before deploy. Run from src/: python build_chroma.py"""

from dotenv import load_dotenv

load_dotenv()

from backend import build_vector_store, load_policy_documents

if __name__ == "__main__":
    build_vector_store(load_policy_documents())
    print("Done. The vector database is in src/chroma_db — commit it for Streamlit Cloud.")
