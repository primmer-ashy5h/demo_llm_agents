
from __future__ import annotations
import json
import os
import sys
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

import streamlit as st

from pydantic import BaseModel, Field

# LangChain / Gemini (guarded import)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ModuleNotFoundError:  # makes script runnable without the package
    ChatGoogleGenerativeAI = None  # type: ignore

from langchain.prompts import ChatPromptTemplate

# LangGraph
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv
import os

# Load variables from .env into environment
load_dotenv()

# Retrieve the key
LLM_API_KEY = os.getenv("GOOGLE_API_KEY")

print("Loaded API Key:", LLM_API_KEY is not None)

# -----------------------------
# Schemas
# -----------------------------
class Transfer(BaseModel):
    # Core
    txn_id: str
    timestamp: str  # ISO 8601
    amount: float
    currency: str

    # Participants
    origin_account_id: str
    destination_account_id: str

    # Transfer type & routing
    transfer_type: str  # "INTERNAL" | "EXTERNAL"
    destination_bank_code: Optional[str] = None  # e.g., SWIFT/IFSC
    destination_country: Optional[str] = None  # ISO country for beneficiary bank
    is_international: Optional[bool] = None

    # Context
    channel: str  # e.g., APP, WEB, BRANCH
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    origin_country: Optional[str] = None  # where customer is initiating

    # Customer & account posture (origin side)
    customer_age: Optional[int] = None
    origin_account_tenure_days: Optional[int] = None

    # Beneficiary posture
    new_beneficiary: Optional[bool] = None
    payee_first_seen_days: Optional[int] = None  # 0 or None => first time
    beneficiary_account_tenure_days: Optional[int] = None

    # Velocity & history
    outbound_velocity_1h_count: Optional[int] = 0
    outbound_velocity_24h_count: Optional[int] = 0
    previous_fraud_reports: Optional[int] = 0

    # Free text
    memo: Optional[str] = None

class Decision(BaseModel):
    verdict: str  # APPROVE | REVIEW | DECLINE
    reasons: List[str] = Field(default_factory=list)
    score: float = 0.0  # composite 0..1
    sla_priority: str = "P3"  # P1 urgent, P2 high, P3 normal


class CaseReport(BaseModel):
    case_id: str
    txn_id: str
    created_at: str
    enrichment: Dict[str, Any]
    rules: Dict[str, Any]
    ml_score: float
    investigator_notes: str
    decision: Decision
    summary: str


# -----------------------------
# LangGraph state
# -----------------------------
class FraudState(TypedDict, total=False):
    transaction: Dict[str, Any]
    enrichment: Dict[str, Any]
    rules: Dict[str, Any]
    ml_score: float
    investigator_notes: str
    decision: Dict[str, Any]
    summary: Dict[str, Any]

# -----------------------------
# Utilities
# -----------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def has_gemini() -> bool:
    return bool(ChatGoogleGenerativeAI) and bool(LLM_API_KEY)


def build_gemini() -> ChatGoogleGenerativeAI:
    if not ChatGoogleGenerativeAI:
        raise RuntimeError("langchain-google-genai not installed. Please install to use Gemini.")
    return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=LLM_API_KEY,
            temperature=0.2)




# -----------------------------
# Node: Enrichment
# -----------------------------
def enrichment_node(state: FraudState) -> Dict[str, Any]:
    print("[enrichment_node] Running enrichment...")
    tx = Transfer(**state["transaction"])  # validate

    # Example reference (toy)
    high_risk_countries = {"RU", "NG", "VN", "PK"}
    risky_memo_keywords = {"crypto", "investment", "loan", "refund", "gift cards"}

    is_external = (tx.transfer_type or "").upper() == "EXTERNAL"
    is_new_beneficiary = bool(
        (tx.new_beneficiary is True)
        or (tx.payee_first_seen_days is not None and tx.payee_first_seen_days <= 0)
    )

    memo_lower = (tx.memo or "").lower()
    memo_risky = any(k in memo_lower for k in risky_memo_keywords) if memo_lower else False

    geo_risk = "LOW"
    dest_ctry = (tx.destination_country or "").upper()
    if dest_ctry and dest_ctry in high_risk_countries:
        geo_risk = "HIGH"

    enrichment = {
        "is_external": is_external,
        "is_new_beneficiary": is_new_beneficiary,
        "device_known": bool(tx.device_id and tx.origin_account_tenure_days and tx.origin_account_tenure_days > 60),
        "geo_risk": geo_risk,
        "memo_risky": memo_risky,
        "velocity": {
            "out_1h": tx.outbound_velocity_1h_count or 0,
            "out_24h": tx.outbound_velocity_24h_count or 0,
        },
        "prior_fraud": tx.previous_fraud_reports or 0,
        "is_young_customer": bool(tx.customer_age and tx.customer_age < 21),
        "origin_tenure_days": tx.origin_account_tenure_days or 0,
        "beneficiary_tenure_days": tx.beneficiary_account_tenure_days or 0,
        "destination_country": dest_ctry,
    }

    return {"enrichment": enrichment}



# -----------------------------
# Node: Rules (transfer‑focused)
# -----------------------------
def rules_node(state: FraudState) -> Dict[str, Any]:
    print("[rules_node] Running rules engine...")
    tx = Transfer(**state["transaction"])  # validate
    e = state.get("enrichment", {})

    flags: Dict[str, Any] = {}
    reasons: List[str] = []

    # R1: High amount thresholds differ for external vs internal
    if e.get("is_external") and tx.amount >= 10000:
        flags["R1_high_amount_external"] = True
        reasons.append("High amount external transfer >= 10000")
    elif not e.get("is_external") and tx.amount >= 5000:
        flags["R1_high_amount_internal"] = True
        reasons.append("High amount internal transfer >= 5000")

    # R2: First‑time / very new beneficiary
    if e.get("is_new_beneficiary"):
        flags["R2_new_beneficiary"] = True
        reasons.append("First‑time or very new beneficiary")

    # R3: High geo risk destination
    if e.get("geo_risk") == "HIGH":
        flags["R3_geo_high"] = True
        reasons.append("High geolocation risk destination")

    # R4: New / unknown device on account
    if not e.get("device_known", False):
        flags["R4_new_device"] = True
        reasons.append("New or unknown device for this account")

    # R5: Velocity spikes (outbound transfers)
    if e.get("velocity", {}).get("out_1h", 0) >= 5:
        flags["R5_velocity_out_1h"] = True
        reasons.append("Outbound velocity 1h >= 5")
    if e.get("velocity", {}).get("out_24h", 0) >= 15:
        flags["R6_velocity_out_24h"] = True
        reasons.append("Outbound velocity 24h >= 15")

    # R6: Prior fraud reports on the customer
    if (e.get("prior_fraud", 0) or 0) > 0:
        flags["R7_prior_fraud"] = True
        reasons.append("Customer has prior fraud reports")

    # R7: Very young origin account doing large transfers
    if (e.get("origin_tenure_days", 0) < 30) and (tx.amount >= 2000):
        flags["R8_new_account_large_transfer"] = True
        reasons.append("New account (<30d) making large transfer >= 2000")

    # R8: Suspicious memo keywords
    if e.get("memo_risky"):
        flags["R9_suspicious_memo"] = True
        reasons.append("Suspicious memo keywords detected")

    # Rule score (0..1)
    num_flags = len(flags)
    rule_score = min(1.0, num_flags / 9.0)

    rules = {
        "flags": flags,
        "reasons": reasons,
        "rule_score": round(rule_score, 3),
    }
    return {"rules": rules}



# -----------------------------
# Node: ML Risk Scoring (heuristic)
# -----------------------------
def ml_risk_node(state: FraudState) -> Dict[str, Any]:
    print("[ml_risk_node] Running ML risk scoring...")
    tx = Transfer(**state["transaction"])  # validate
    e = state.get("enrichment", {})
    r = state.get("rules", {})

    # Weighted factors → pseudo probability (0..1)
    w_amount = min(tx.amount / (15000.0 if e.get("is_external") else 8000.0), 1.0) * 0.35
    w_geo = 0.15 if e.get("geo_risk") == "HIGH" else 0.0
    w_new_bene = 0.15 if e.get("is_new_beneficiary") else 0.0
    w_device = 0.10 if not e.get("device_known", False) else 0.0
    w_velocity = min((e.get("velocity", {}).get("out_1h", 0) / 10.0), 1.0) * 0.10
    w_prior_fraud = min((e.get("prior_fraud", 0) / 3.0), 1.0) * 0.05
    w_new_acct = 0.05 if e.get("origin_tenure_days", 0) < 30 else 0.0
    w_memo = 0.05 if e.get("memo_risky") else 0.0
    w_rules = (r.get("rule_score", 0.0)) * 0.20

    score = w_amount + w_geo + w_new_bene + w_device + w_velocity + w_prior_fraud + w_new_acct + w_memo + w_rules
    score = max(0.0, min(score, 1.0))

    return {"ml_score": round(float(score), 3)}



# -----------------------------
# Node: Investigator (Gemini)
# -----------------------------
def investigator_node(state: FraudState) -> Dict[str, Any]:
    print("[investigator_node] Running investigator analysis...")
    tx = Transfer(**state["transaction"])  # validate
    e = state.get("enrichment", {})
    r = state.get("rules", {})
    ml = state.get("ml_score", 0.0)

    context = {
        "transfer": tx.model_dump(),
        "enrichment": e,
        "rule_flags": r.get("flags", {}),
        "rule_reasons": r.get("reasons", []),
        "rule_score": r.get("rule_score", 0.0),
        "ml_score": ml,
    }

    if not has_gemini():
        # Fallback, deterministic explanation
        notes = [
            "Gemini unavailable; using heuristic investigator.",
            f"ML risk score={ml:.3f}",
        ]
        if ml >= 0.75 or r.get("rule_score", 0.0) >= 0.75:
            notes.append("RECOMMENDATION: DECLINE")
            notes.append("Key drivers: high rules and/or ML score")
        elif ml >= 0.45 or r.get("rule_score", 0.0) >= 0.45:
            notes.append("RECOMMENDATION: REVIEW")
            notes.append("Key drivers: moderate risk indicators")
        else:
            notes.append("RECOMMENDATION: APPROVE")
            notes.append("Low combined signals")
        return {"investigator_notes": "\n".join(notes)}

    llm = build_gemini()

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a senior fraud investigator for a bank.\n"
            "Given transfer context, analyze risk drivers succinctly.\n"
            "Output 3 short bullet points and end with a single line:\n"
            "RECOMMENDATION: APPROVE|REVIEW|DECLINE\n"
            "Keep it objective and reference concrete factors."
        )),
        ("human", "Context JSON:\n{context}"),
    ])

    chain = prompt | llm
    resp = chain.invoke({"context": json.dumps(context, separators=(",", ":"))})
    notes = resp.content if hasattr(resp, "content") else str(resp)

    return {"investigator_notes": notes}


# -----------------------------
# Node: Decision
# -----------------------------
def decision_node(state: FraudState) -> Dict[str, Any]:
    print("[decision_node] Making final decision...")
    r = state.get("rules", {})
    ml = float(state.get("ml_score", 0.0))
    notes = state.get("investigator_notes", "")

    # Parse investigator recommendation if present
    rec = ""
    for line in str(notes).splitlines()[::-1]:
        line_u = line.strip().upper()
        if line_u.startswith("RECOMMENDATION:"):
            rec = line_u.replace("RECOMMENDATION:", "").strip()
            break

    # Composite score
    rule_score = float(r.get("rule_score", 0.0))
    composite = min(1.0, 0.6 * ml + 0.4 * rule_score)

    if rec in {"APPROVE", "REVIEW", "DECLINE"}:
        verdict = rec
    else:
        verdict = "DECLINE" if composite >= 0.75 else ("REVIEW" if composite >= 0.45 else "APPROVE")

    reasons = list(r.get("reasons", []))
    reasons.append(f"ML score={ml:.2f}")
    reasons.append(f"Rule score={rule_score:.2f}")
    if rec:
        reasons.append(f"Investigator suggested {rec}")

    sla = "P1" if verdict == "DECLINE" else ("P2" if verdict == "REVIEW" else "P3")

    decision = Decision(verdict=verdict, reasons=reasons, score=round(composite, 3), sla_priority=sla)
    return {"decision": decision.model_dump()}




# -----------------------------
# Node: Summary
# -----------------------------
def summary_node(state: FraudState) -> Dict[str, Any]:
    tx = Transfer(**state["transaction"])  # validate
    case_id = str(uuid.uuid4())
    created_at = now_iso()
    enrichment = state.get("enrichment", {})
    rules = state.get("rules", {})
    ml_score = float(state.get("ml_score", 0.0))
    investigator_notes = state.get("investigator_notes", "")
    decision = Decision(**state.get("decision", {}))

    narrative = (
        f"Transfer {tx.txn_id} {tx.amount:.2f} {tx.currency} {tx.transfer_type} "
        f"from {tx.origin_account_id} to {tx.destination_account_id}. "
        f"Dest {tx.destination_country or 'N/A'}. ML={ml_score:.2f}, "
        f"Rules={rules.get('rule_score', 0.0):.2f}, Verdict={decision.verdict}."
    )

    report = CaseReport(
        case_id=case_id,
        txn_id=tx.txn_id,
        created_at=created_at,
        enrichment=enrichment,
        rules=rules,
        ml_score=ml_score,
        investigator_notes=investigator_notes,
        decision=decision,
        summary=narrative,
    )

    return {"summary": {"case_report": report.model_dump()}}


# -----------------------------
# Build the graph
# -----------------------------
workflow = StateGraph(FraudState)
workflow.add_node("enrichment", enrichment_node)
workflow.add_node("rules", rules_node)
workflow.add_node("ml_risk", ml_risk_node)
workflow.add_node("investigator", investigator_node)
workflow.add_node("decision", decision_node)
workflow.add_node("summary", summary_node)

workflow.add_edge(START, "enrichment")
workflow.add_edge("enrichment", "rules")
workflow.add_edge("rules", "ml_risk")
workflow.add_edge("ml_risk", "investigator")
workflow.add_edge("investigator", "decision")
workflow.add_edge("decision", "summary")
workflow.add_edge("summary", END)

app = workflow.compile()


# -----------------------------
# Public function: run a case
# -----------------------------
def run_case(transfer_json: Dict[str, Any]) -> Dict[str, Any]:
    state_in: FraudState = {"transaction": transfer_json}
    final_state = app.invoke(state_in)
    print("[run_case] execution completed....")
    return final_state["summary"]["case_report"]

# -----------------------------
# Streamlit UI
# -----------------------------

# Set page layout to wide automatically
st.set_page_config(page_title="Fraud Payment Case Investigator", layout="wide")

st.title("💳 Fraud Payment Case Investigator")

st.markdown("### ⚙️ Workflow Steps")

st.markdown(
    """
    Enrichment → Rules → ML Risk → Investigator → Decision → Summary
    """
)

st.markdown(
    """
    ### 🔎  Details
    1. **🧩 Enrichment** — Add more features to the raw transaction on the fly for better case evaluation from varied sources.  
    2. **📜 Rules Application** — Apply pre-built rules to the enriched transaction to flag potential fraud.  
    3. **🤖 ML Risk Scoring** — Apply ML model (heuristic) to score the risk of the transaction.  
    4. **🕵️ Agent Based Investigation** — Analyze the transaction using LLM and provide a decison/recommendation.  
    5. **✅ Decision** — Make the final decision based on the rules and ML score - APPROVE | REVIEW | DECLINE.
    6. **📝 Summary** — Bind all above findings into a report. 
    """
)

default_example = {
    "txn_id": "TX-3003",
    "timestamp": "2025-08-19T14:38:00Z",
    "amount": 5200.0,
    "currency": "USD",
    "origin_account_id": "ACC-004",
    "destination_account_id": "ACC-005",
    "transfer_type": "INTERNAL",
    "destination_country": "US",
    "is_international": False,
    "channel": "APP",
    "device_id": "dev-new-xyz",
    "origin_country": "US",
    "customer_age": 33,
    "origin_account_tenure_days": 45,
    "new_beneficiary": True, 
    "payee_first_seen_days": 0,
    "beneficiary_account_tenure_days": 10,
    "outbound_velocity_1h_count": 2,
    "outbound_velocity_24h_count": 3,
    "previous_fraud_reports": 0,
    "memo": "gift"
}


# JSON input area
transaction_input = st.text_area("**Enter a payment transaction in JSON format here**", value=json.dumps(default_example, indent=2), height=300)

if st.button("Run Investigator"):
    try:
        tx_json = json.loads(transaction_input)
        case_report = run_case(tx_json)

        st.success("Case investigation completed.")

        # Layout: Summary card + details tabs
        st.subheader("📋 Case Summary")
        st.markdown(f"**Case ID:** {case_report['case_id']}")
        st.markdown(f"**Transaction ID:** {case_report['txn_id']}")
        st.markdown(f"**Created At:** {case_report['created_at']}")
        st.markdown(f"**Summary:** {case_report['summary']}")

        decision = case_report["decision"]
        st.markdown("---")
        st.markdown(f"### 🛑 DECISION: **{decision['verdict']}**")
        st.markdown(f"**SLA Priority:** {decision['sla_priority']}")
        if "next_best_action" in decision:
            st.markdown(f"**Next Best Action:** {decision['next_best_action']}")

        # Tabs for enrichment, rules, ML, investigator notes
        tab1, tab2, tab3, tab4 = st.tabs(["Enrichment", "Rules", "ML & Scores", "Investigator Notes"])

        with tab1:
            st.json(case_report["enrichment"])

        with tab2:
            st.json(case_report["rules"])

        with tab3:
            st.markdown(f"**ML Score:** {case_report['ml_score']}")
            st.markdown(f"**Decision Score:** {decision['score']}")
            st.markdown("**Reasons:**")
            for reason in decision["reasons"]:
                st.markdown(f"- {reason}")

        with tab4:
            st.text_area("Investigator Notes", value=case_report["investigator_notes"], height=200)

        # Raw JSON output
        with st.expander("🔍 Full Case Report JSON"):
            st.json(case_report)

        st.markdown(
        """
        ### 🔎 Incremental Agentic Transformation - Bring Dynamism, Evolution and Scale
        1. **🧩 Enrichment Agent** — Connect to multiple data sources for dynamic data point enrichments. 
        2. **📜 Rules Agent** — Evolve rules based on new threat patterns.  
        3. **🤖 ML Risk Agent** — Manage dynamic risk thesholds catering to new patterns/ threat archetypes without need for full re-training.  
        4. **🕵️ Investigator Agent** — Apart from decision, agent can furthur suggest "next best action" recommendation for human agents, eg. call customer to confirm the transaction, etc.   
        5. **✅ Decision Agent** — Apart from recommending decision,actions, agent can take action on behalf of human agents.
        6. **📝 Reporting and Audit** — Generate report, ensure all case are audit compliant. 
        """
)

    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please check your input.")
    except Exception as e:
        st.error(f"Error running case investigation: {e}")
