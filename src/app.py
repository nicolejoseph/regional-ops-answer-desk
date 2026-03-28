"""
Your field partner — Streamlit entrypoint for local and Streamlit Community Cloud.

Local:  streamlit run app.py  (from the src/ directory)
Requirements: see ../requirements.txt at repo root
"""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from backend import CHROMA_PERSIST_DIR, run_assistant_query
from openai_key import get_openai_api_key

# Mock internal URLs for verifying retrieved policies (prototype only).
MOCK_POLICY_PORTAL_BASE = "https://policy-library.internal.example"

# One-tap starters for busy regional managers (edit body, then submit).
QUICK_SCENARIOS = [
    (
        "Phantom on-hand / complaints",
        "What should a manager do if a high-demand item keeps showing as in stock "
        "but cannot be found and customers are complaining?",
    ),
    (
        "Same-day staffing gap",
        "We have a same-day staffing shortage before opening. What should the store manager do first?",
    ),
    (
        "Shrink / missing product pattern",
        "Repeated missing items in one category and unusual loss. What steps should management take?",
    ),
    (
        "Customer out-of-stock recovery",
        "A customer is upset because a promoted item is out of stock. How should associates and the manager respond?",
    ),
]


def mock_policy_document_url(policy_id: str) -> str:
    safe_id = policy_id or "unknown"
    return f"{MOCK_POLICY_PORTAL_BASE}/policies/{safe_id}"


def chroma_database_ready() -> bool:
    p = Path(CHROMA_PERSIST_DIR)
    if not p.is_dir():
        return False
    return any(p.iterdir())


@st.cache_resource
def load_vector_store():
    load_dotenv()
    embedding_function = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=get_openai_api_key(),
    )
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_function,
    )


def render_sidebar_quick_scenarios():
    st.sidebar.markdown("### Common Situations:")
    st.sidebar.caption(
        "Pick one to pre-fill your question—you can edit anything before you submit."
    )
    for i, (label, text) in enumerate(QUICK_SCENARIOS):
        if st.sidebar.button(label, key=f"quick_{i}", use_container_width=True):
            st.session_state.question_input = text


def render_retrieved_docs(docs, *, compact=False):
    if not compact:
        st.subheader("Sources")
        st.caption(
            "Open any link to read the full policy in the internal library (demo URLs)."
        )
    if not docs:
        st.info("We didn't find matching policy excerpts for this one.")
        return

    lines = []
    for i, doc in enumerate(docs, start=1):
        policy_id = doc.metadata.get("id", "")
        title = doc.metadata.get("title", "Unknown Title")
        url = mock_policy_document_url(policy_id)
        lines.append(f"{i}. [{title}]({url})")

    st.markdown("\n\n".join(lines))


def render_cross_store_summary(summary):
    if not summary:
        st.info("We couldn't match this to a network pattern yet—that doesn't mean you're on your own.")
        return

    st.markdown(f"**What we're seeing:** {summary['label']}")
    st.markdown(
        f"- **Stores affected:** {summary['distinct_store_count']}\n"
        f"- **Regions:** {len(summary['regions'])}\n"
        f"- **Recent reports (rows):** {summary['incident_count']}"
    )
    st.markdown(f"**Regions:** {', '.join(summary['regions'])}")
    st.markdown(f"**Store IDs:** {', '.join(summary['store_ids'])}")

    st.markdown("**Recent examples from the field**")
    for row in summary["recent_incidents"]:
        st.markdown(
            f"- `{row['store_id']}` ({row['region']}, {row['reported_week']}): {row['summary']}"
        )


def main():
    st.set_page_config(
        page_title="Regional Operations Answer Desk!",
        page_icon=":shopping_cart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not chroma_database_ready():
        st.error(
            "**Vector database missing.** The app needs a pre-built folder `chroma_db/` next to the code "
            "(under `src/`). On your machine, from the `src/` directory, run:\n\n"
            "`python build_chroma.py`\n\n"
            "Then commit the `chroma_db/` folder so Streamlit Cloud can load it. "
            "This app does not rebuild embeddings on each run."
        )
        return

    st.sidebar.markdown("### Hi there!")
    st.sidebar.markdown(
        "Glad you're here :) This tool pulls answers from **your internal playbooks** "
        "so you can support stores without digging through PDFs mid-crisis."
    )
    st.sidebar.markdown(
        "**What you'll see**\n\n"
        "- A clear recommendation you can act on or share\n"
        "- Whether similar situations have popped up at other locations\n"
        "- Links to the exact policies behind the answer"
    )
    st.sidebar.divider()
    st.sidebar.markdown("### Store IDs")
    st.sidebar.markdown(
        "If the situation is tied to one store, mention it as **STORE_214** or "
        "**store 214**. If we don't recognize a store ID, we'll hold the answer—"
        "we'd rather pause than send you off with the wrong location on record."
    )
    render_sidebar_quick_scenarios()

    st.title("Regional Operations Answer Desk :shopping_cart:")
    st.caption(
        "Clear, policy-grounded guidance for the moments when the floor needs you most—"
        "plus a quick pulse on whether your peers are seeing the same thing."
    )

    if "question_input" not in st.session_state:
        st.session_state.question_input = QUICK_SCENARIOS[0][1]

    with st.form("query_form", clear_on_submit=False):
        user_query = st.text_area(
            "What's going on?",
            key="question_input",
            height=120,
            placeholder="Describe it in your own words. Add a store ID if it's about a specific location.",
        )
        submitted = st.form_submit_button(
            "Get guidance",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    if not user_query.strip():
        st.warning("Share a quick description of what's happening—we'll take it from there.")
        return

    try:
        vector_store = load_vector_store()
    except Exception as e:
        st.error(
            "Could not load the policy vector database. "
            "Regenerate `chroma_db/` with `python build_chroma.py` and try again."
        )
        st.caption(str(e))
        return

    with st.spinner("Pulling the right policies and checking what other stores have reported…"):
        output = run_assistant_query(user_query, k=3, vector_store=vector_store)

    if output.get("blocked_reason") == "unknown_store":
        ids = ", ".join(output.get("invalid_stores") or [])
        st.warning(
            f"We couldn't find these store IDs in our records: `{ids}`. "
            "Double-check the ID and try again—we want to make sure we're answering for the right place."
        )
        return
    if output.get("blocked_reason"):
        st.error(
            "Something went sideways while we prepared your answer. Give it another try in a moment, "
            "or tweak how you described the situation."
        )
        return

    with st.container(border=True):
        st.markdown("### Here's your starting point")
        st.write(output["grounded_answer"])

    summary = output["cross_store_summary"]
    if summary:
        st.info(
            f"**Across the network:** we've seen this type of issue at "
            f"**{summary['distinct_store_count']} store(s)** in "
            f"**{len(summary['regions'])}** region(s) in recent reporting—you're not alone in spotting it."
        )
    else:
        st.caption(
            "We didn't find a matching network pattern for this exact wording—that's okay. "
            "The guidance above still stands; you can also try words like *inventory*, *staffing*, "
            "*shrink*, or *escalation* if you want a network snapshot next time."
        )

    c1, c2 = st.columns(2)
    with c1:
        with st.expander("See the policies behind this answer", expanded=False):
            render_retrieved_docs(
                output["retrieved_docs"],
                compact=True,
            )
    with c2:
        with st.expander("More detail from other stores", expanded=False):
            if summary:
                render_cross_store_summary(summary)
            else:
                st.caption(
                    "Nothing extra to show here—the question didn't match one of our tracked issue types."
                )


if __name__ == "__main__":
    main()
