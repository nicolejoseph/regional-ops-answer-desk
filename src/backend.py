"""
Requirements:
pip install langchain-openai langchain-chroma python-dotenv
"""

import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from openai_key import get_openai_api_key

# Persist next to this package so paths work locally and on Streamlit Cloud (main: src/app.py).
CHROMA_PERSIST_DIR = str(Path(__file__).resolve().parent / "chroma_db")


def load_policy_documents():
    synthetic_policy_docs = [
        {
            "id": "policy_001",
            "title": "Inventory Gap Response Policy",
            "topic": "inventory",
            "text": """
Store managers must review inventory gaps on high-demand items at the start of each shift.
If an item is out of stock but expected on hand, the manager should first confirm whether the item is misplaced in the stockroom, on a returns cart, or in a secondary display.
If the item is not found within 30 minutes, the manager should create an inventory discrepancy report and notify the inventory operations lead.
If more than 5 units of a high-demand item are missing for 2 consecutive days, the issue must be escalated to district operations for investigation.
Managers should place a shelf tag or customer notice for unavailable high-demand items and suggest substitute products when appropriate.
"""
        },
        {
            "id": "policy_002",
            "title": "Same-Day Staffing Shortage Playbook",
            "topic": "staffing",
            "text": """
When a same-day staffing shortage occurs, the store manager should first review the shift schedule and identify critical roles that must be covered, such as cashier, customer service desk, and receiving.
The manager should contact available part-time associates or approved on-call staff before reassigning employees from non-critical tasks.
If coverage cannot be found within 1 hour, the manager may reduce non-essential store activities such as shelf resets or routine backroom organization.
If the shortage affects store opening, closing, or customer safety, the manager must escalate the issue to the district manager immediately.
Overtime may only be approved for trained employees covering critical functions.
"""
        },
        {
            "id": "policy_003",
            "title": "Store Escalation Guidelines",
            "topic": "escalation",
            "text": """
Store-level issues should be escalated when they affect customer safety, cause repeated operational disruption, or create financial risk.
Examples include repeated stock discrepancies, severe staffing shortages, theft patterns, refrigeration failures, and unresolved customer incidents.
Managers should document the issue, actions taken, time of occurrence, and business impact before escalating.
Escalations to district leadership should include a summary of the problem, any temporary mitigation already attempted, and the support needed.
If the issue is urgent and impacts safety or store operations in real time, escalation should happen immediately and not wait for end-of-day reporting.
"""
        },
        {
            "id": "policy_004",
            "title": "Loss Prevention and Shrink Reporting Procedure",
            "topic": "shrink",
            "text": """
If a manager observes unusual inventory loss, repeated missing items, or suspicious customer or employee behavior, they must document the incident in the loss prevention log the same day.
Managers should not directly accuse employees or customers without following approved reporting procedures.
If shrink is concentrated in a specific product category or exceeds normal weekly levels, the case should be reported to the district loss prevention partner within 24 hours.
Relevant information includes SKU numbers, affected quantity, approximate dollar value, time of day, and any camera coverage available.
Managers may increase floor presence and secure high-risk items while awaiting further guidance.
"""
        },
        {
            "id": "policy_005",
            "title": "Customer Recovery for Out-of-Stock Items",
            "topic": "customer_service",
            "text": """
When a customer is affected by an out-of-stock item, store associates should first apologize and confirm whether inventory is available in the backroom or at another nearby location.
If the item is unavailable, the associate should offer a comparable substitute when possible.
For high-frustration situations involving essential items or promotional products, the manager may approve a small courtesy discount on a substitute item based on store guidelines.
Repeated customer complaints about the same unavailable item should be shared with the merchandising and inventory teams in the daily operations summary.
Customer-facing communication should remain professional, clear, and solution-oriented.
"""
        },
        {
            "id": "policy_006",
            "title": "Shift Coverage and Overtime Rules",
            "topic": "staffing",
            "text": """
Overtime should only be used when required to maintain critical store operations or customer safety.
Managers should first attempt schedule adjustments, voluntary shift extensions, or on-call coverage before approving overtime.
Employees assigned overtime must be trained for the role they are covering.
Unplanned overtime exceeding 2 hours for a single shift should be documented with a reason code and included in the weekly labor review.
If a location requires repeated overtime due to persistent vacancies, the issue should be escalated to workforce planning.
"""
        }
    ]
    return synthetic_policy_docs


def load_cross_store_issue_types():
    """
    Canonical issue taxonomy for cross-store reporting.

    Structure:
    - issue_key: stable ID used to join incidents to policies and analytics.
    - label: human-readable name for UI and LLM routing.
    - related_policy_topics: aligns with policy metadata["topic"] for retrieval.
    - match_keywords: simple routing hints for the prototype (replace with a
      classifier or embedding match in production).
    """
    return [
        {
            "issue_key": "inventory_gap_high_demand",
            "label": "High-demand inventory gap (system shows stock, floor cannot find)",
            "related_policy_topics": ["inventory", "customer_service"],
            "match_keywords": [
                "inventory gap",
                "high-demand",
                "on hand",
                "cannot find",
                "misplaced",
                "discrepancy",
                "complaining",
            ],
        },
        {
            "issue_key": "oos_customer_complaints",
            "label": "Customer complaints about out-of-stock or unavailable items",
            "related_policy_topics": ["customer_service", "inventory"],
            "match_keywords": [
                "out of stock",
                "oos",
                "unavailable",
                "complaints",
                "substitute",
                "courtesy",
            ],
        },
        {
            "issue_key": "same_day_staffing_shortage",
            "label": "Same-day staffing shortage / coverage gaps",
            "related_policy_topics": ["staffing"],
            "match_keywords": [
                "staffing",
                "shortage",
                "coverage",
                "callout",
                "shift",
                "on-call",
            ],
        },
        {
            "issue_key": "shrink_category_spike",
            "label": "Shrink concentrated in a category or above normal levels",
            "related_policy_topics": ["shrink", "escalation"],
            "match_keywords": [
                "shrink",
                "loss prevention",
                "missing items",
                "theft",
                "sku",
            ],
        },
        {
            "issue_key": "store_escalation_pattern",
            "label": "Repeated operational disruption or safety risk pattern",
            "related_policy_topics": ["escalation"],
            "match_keywords": [
                "escalate",
                "district",
                "disruption",
                "safety",
                "repeated",
            ],
        },
        {
            "issue_key": "refrigeration_cooler_failure",
            "label": "Refrigeration or cooler equipment failure",
            "related_policy_topics": ["escalation"],
            "match_keywords": [
                "refrigeration",
                "cooler",
                "temperature",
                "spoilage",
            ],
        },
    ]


def load_cross_store_incidents():
    """
    Incident fact table: one row per reported occurrence at a store.

    Grain: store + issue_key + time window (synthetic weekly buckets).
    In production this would come from a warehouse/API; here it is synthetic.
    """
    return [
        {
            "incident_id": "xinc_001",
            "issue_key": "inventory_gap_high_demand",
            "store_id": "STORE_214",
            "region": "Northeast",
            "reported_week": "2026-W10",
            "summary": "POS shows 12 on hand; repeated floor sweeps; 6 customer complaints in 3 days.",
        },
        {
            "incident_id": "xinc_002",
            "issue_key": "inventory_gap_high_demand",
            "store_id": "STORE_087",
            "region": "Southeast",
            "reported_week": "2026-W09",
            "summary": "High-demand SKU mismatch between backroom bin and salesfloor; discrepancy report filed.",
        },
        {
            "incident_id": "xinc_003",
            "issue_key": "inventory_gap_high_demand",
            "store_id": "STORE_402",
            "region": "West",
            "reported_week": "2026-W11",
            "summary": "System on-hand positive; item not located within 45 minutes; district notified after day 2.",
        },
        {
            "incident_id": "xinc_004",
            "issue_key": "inventory_gap_high_demand",
            "store_id": "STORE_091",
            "region": "Northeast",
            "reported_week": "2026-W08",
            "summary": "Seasonal item showing phantom inventory; secondary display checked; ops summary updated.",
        },
        {
            "incident_id": "xinc_005",
            "issue_key": "oos_customer_complaints",
            "store_id": "STORE_155",
            "region": "Midwest",
            "reported_week": "2026-W10",
            "summary": "Promotional item unavailable; multiple complaints; substitutes offered.",
        },
        {
            "incident_id": "xinc_006",
            "issue_key": "oos_customer_complaints",
            "store_id": "STORE_214",
            "region": "Northeast",
            "reported_week": "2026-W10",
            "summary": "Essential item OOS; manager approved courtesy discount on substitute per playbook.",
        },
        {
            "incident_id": "xinc_007",
            "issue_key": "oos_customer_complaints",
            "store_id": "STORE_330",
            "region": "Southeast",
            "reported_week": "2026-W09",
            "summary": "Daily ops summary flagged repeat complaints on same SKU.",
        },
        {
            "incident_id": "xinc_008",
            "issue_key": "same_day_staffing_shortage",
            "store_id": "STORE_044",
            "region": "West",
            "reported_week": "2026-W10",
            "summary": "Two cashier callouts; on-call used; non-essential resets deferred.",
        },
        {
            "incident_id": "xinc_009",
            "issue_key": "same_day_staffing_shortage",
            "store_id": "STORE_301",
            "region": "Midwest",
            "reported_week": "2026-W09",
            "summary": "Opening coverage gap; district contacted within 30 minutes.",
        },
        {
            "incident_id": "xinc_010",
            "issue_key": "same_day_staffing_shortage",
            "store_id": "STORE_087",
            "region": "Southeast",
            "reported_week": "2026-W11",
            "summary": "Receiving short; tasks reassigned; overtime limited to trained roles.",
        },
        {
            "incident_id": "xinc_011",
            "issue_key": "shrink_category_spike",
            "store_id": "STORE_220",
            "region": "Northeast",
            "reported_week": "2026-W09",
            "summary": "Category shrink above weekly threshold; LP partner notified within 24h.",
        },
        {
            "incident_id": "xinc_012",
            "issue_key": "shrink_category_spike",
            "store_id": "STORE_402",
            "region": "West",
            "reported_week": "2026-W10",
            "summary": "Repeat missing items; camera review requested; high-risk items secured.",
        },
        {
            "incident_id": "xinc_013",
            "issue_key": "store_escalation_pattern",
            "store_id": "STORE_014",
            "region": "Midwest",
            "reported_week": "2026-W08",
            "summary": "Repeated stock discrepancies plus customer incidents; escalation package prepared.",
        },
        {
            "incident_id": "xinc_014",
            "issue_key": "refrigeration_cooler_failure",
            "store_id": "STORE_199",
            "region": "Southeast",
            "reported_week": "2026-W10",
            "summary": "Cooler temp excursion; product quarantined; immediate district escalation.",
        },
    ]


def get_known_store_ids(incidents):
    return {row["store_id"] for row in incidents}


def _normalize_store_suffix(suffix: str) -> str:
    """Map a raw store slug to canonical STORE_* id for lookup."""
    if re.fullmatch(r"\d+", suffix):
        num = suffix
        return f"STORE_{num.zfill(3)}" if len(num) < 3 else f"STORE_{num}"
    return f"STORE_{suffix.upper()}"


def extract_store_ids_from_query(user_query):
    """Find store IDs: STORE_214, store_214, store_abc, or 'store 214' / 'store #214'."""
    ids = set()
    # Underscore form (digits or alphanumeric, e.g. store_abc / STORE_087)
    for m in re.finditer(r"\bstore_([a-z0-9]+)\b", user_query, re.I):
        ids.add(_normalize_store_suffix(m.group(1)))
    # Spelled-out number: "store 214", "store #214"
    for m in re.finditer(r"\bstore\s*#?\s*(\d+)\b", user_query, re.I):
        num = m.group(1)
        ids.add(f"STORE_{num.zfill(3)}" if len(num) < 3 else f"STORE_{num}")
    return ids


def validate_store_references(user_query, incidents):
    """
    If the query names one or more stores, every name must exist in incidents.
    If nothing is named, validation passes.
    Returns (ok, invalid_ids_or_none).
    """
    known = get_known_store_ids(incidents)
    mentioned = extract_store_ids_from_query(user_query)
    if not mentioned:
        return True, None
    invalid = sorted(s for s in mentioned if s not in known)
    if invalid:
        return False, invalid
    return True, None


def match_issue_key_from_query(user_query, issue_types):
    """Prototype routing: pick the issue_key with the strongest keyword overlap."""
    q = user_query.lower()
    best_key = None
    best_score = 0
    for it in issue_types:
        score = sum(1 for kw in it["match_keywords"] if kw.lower() in q)
        if score > best_score:
            best_score = score
            best_key = it["issue_key"]
    if best_score == 0:
        return None
    return best_key


def get_cross_store_summary(issue_key, issue_types, incidents):
    """Aggregate incidents for an issue_key: how many stores, regions, recent rows."""
    type_by_key = {t["issue_key"]: t for t in issue_types}
    info = type_by_key.get(issue_key)
    if not info:
        return None

    rows = [r for r in incidents if r["issue_key"] == issue_key]
    store_ids = sorted({r["store_id"] for r in rows})
    regions = sorted({r["region"] for r in rows})
    return {
        "issue_key": issue_key,
        "label": info["label"],
        "related_policy_topics": info["related_policy_topics"],
        "incident_count": len(rows),
        "distinct_store_count": len(store_ids),
        "store_ids": store_ids,
        "regions": regions,
        "recent_incidents": sorted(rows, key=lambda r: r["reported_week"], reverse=True)[:5],
    }


def print_cross_store_summary(summary):
    if not summary:
        print("\nCross-store: Could not classify the question into a known issue type.")
        return

    print("\n--- Cross-store signal ---")
    print(f"Issue: {summary['label']}")
    print(
        f"This pattern appears in {summary['distinct_store_count']} distinct store(s) "
        f"across {len(summary['regions'])} region(s) "
        f"({summary['incident_count']} incident row(s))."
    )
    print(f"Regions: {', '.join(summary['regions'])}")
    print(f"Store IDs: {', '.join(summary['store_ids'])}")
    print("Recent examples:")
    for r in summary["recent_incidents"]:
        print(
            f"  - {r['store_id']} ({r['region']}, {r['reported_week']}): {r['summary']}"
        )


def get_embedding_function():
    load_dotenv()
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Set it in .env (local) or Streamlit secrets (deployed)."
        )
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)


def build_vector_store(docs):
    # Recreate local Chroma directory on each run (run build_chroma.py to refresh)
    persist_dir = CHROMA_PERSIST_DIR
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    # Convert dict policies to LangChain Documents
    lc_docs = []
    for policy in docs:
        lc_docs.append(
            Document(
                page_content=policy["text"].strip(),
                metadata={
                    "id": policy["id"],
                    "title": policy["title"],
                    "topic": policy["topic"],
                },
            )
        )

    vector_store = Chroma.from_documents(
        documents=lc_docs,
        embedding=get_embedding_function(),
        persist_directory=persist_dir,
    )
    return vector_store


def retrieve_relevant_docs(vector_store, user_query, k=3):
    results = vector_store.similarity_search(user_query, k=k)
    return results


def print_results(results):
    for idx, doc in enumerate(results, start=1):
        title = doc.metadata.get("title", "Unknown Title")
        topic = doc.metadata.get("topic", "Unknown Topic")
        snippet = doc.page_content[:300].strip()

        print(f"\nResult {idx}")
        print(f"Title: {title}")
        print(f"Topic: {topic}")
        print(f"Content Preview: {snippet}")


def generate_grounded_answer(results, user_query):
    load_dotenv()
    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Set it in .env (local) or Streamlit secrets (deployed)."
        )

    # Build a compact context from retrieved policy documents
    context_parts = []
    for doc in results:
        context_parts.append(
            f"Title: {doc.metadata.get('title', '')}\n"
            f"Topic: {doc.metadata.get('topic', '')}\n"
            f"Content: {doc.page_content.strip()}"
        )
    context = "\n\n---\n\n".join(context_parts)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    prompt = f"""
You are a retail policy assistant.
Answer the user's question using ONLY the internal policy context below.
If the context is not enough, say you do not have enough policy information.
Keep the answer concise and actionable.

User Question:
{user_query}

Internal Policy Context:
{context}
"""
    response = llm.invoke(prompt)
    return response.content


def run_assistant_query(user_query, k=3, vector_store=None):
    incidents = load_cross_store_incidents()
    try:
        ok, invalid_stores = validate_store_references(user_query, incidents)
        if not ok:
            return {
                "grounded_answer": None,
                "retrieved_docs": [],
                "cross_store_summary": None,
                "blocked_reason": "unknown_store",
                "invalid_stores": invalid_stores,
            }
    except Exception:
        return {
            "grounded_answer": None,
            "retrieved_docs": [],
            "cross_store_summary": None,
            "blocked_reason": "validation_error",
            "invalid_stores": None,
        }

    # Build once per process unless a store is passed in (e.g. Streamlit cache)
    if vector_store is None:
        docs = load_policy_documents()
        vector_store = build_vector_store(docs)

    try:
        results = retrieve_relevant_docs(vector_store, user_query, k=k)
        grounded_answer = generate_grounded_answer(results, user_query)
    except Exception:
        return {
            "grounded_answer": None,
            "retrieved_docs": [],
            "cross_store_summary": None,
            "blocked_reason": "assistant_error",
            "invalid_stores": None,
        }

    issue_types = load_cross_store_issue_types()
    try:
        matched_issue_key = match_issue_key_from_query(user_query, issue_types)
        cross_summary = (
            get_cross_store_summary(matched_issue_key, issue_types, incidents)
            if matched_issue_key
            else None
        )
    except Exception:
        cross_summary = None

    return {
        "grounded_answer": grounded_answer,
        "retrieved_docs": results,
        "cross_store_summary": cross_summary,
        "blocked_reason": None,
        "invalid_stores": None,
    }


def main():
    query = (
        "What should a manager do if a high-demand item keeps showing as in stock "
        "but cannot be found and customers are complaining?"
    )
    output = run_assistant_query(query, k=3)
    if output.get("blocked_reason"):
        reason = output["blocked_reason"]
        if reason == "unknown_store":
            print(
                "\nNo recommendation: the following store(s) are not in the "
                f"cross-store database: {', '.join(output.get('invalid_stores') or [])}"
            )
        else:
            print("\nNo recommendation could be provided due to an error.")
        return

    results = output["retrieved_docs"]
    grounded_answer = output["grounded_answer"]
    cross_summary = output["cross_store_summary"]

    print_results(results)

    print("\nGrounded Answer:")
    print(grounded_answer)

    print_cross_store_summary(cross_summary)

    print("\nSuccess: Retrieval prototype ran successfully.")


if __name__ == "__main__":
    main()
