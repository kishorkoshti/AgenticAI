import re
from langgraph.graph import StateGraph, END
from .state import AgentState
from agents.rag import load_rag_agent
from agents.structured_data import create_structured_agent
from agents.sar import create_sar_chain
from config import SAFETY_BLOCK_KEYWORDS

# ------------------------------------------------------------------
# Router node
# ------------------------------------------------------------------
def router_node(state: AgentState) -> AgentState:
    query = state["query"].lower()

    # Safety check – block evasion / concealment queries
    for kw in SAFETY_BLOCK_KEYWORDS:
        if kw in query:
            state["query_type"] = "refused"
            state["agent_response"] = "I cannot answer this question. It requests advice on evading detection or concealing transactions, which violates my safety guidelines."
            return state

    # Classify safe query
    if any(word in query for word in ["red flag", "guidance", "policy", "jmlsg", "aml", "typology"]):
        state["query_type"] = "policy"
    elif "draft a sar" in query or "sar narrative" in query or "alert #" in query:
        state["query_type"] = "sar"
    else:
        state["query_type"] = "structured_data"

    return state

# ------------------------------------------------------------------
# Agent nodes
# ------------------------------------------------------------------
def rag_policy_node(state: AgentState) -> AgentState:
    rag_agent = load_rag_agent()
    answer, citations = rag_agent(state["query"])
    state["agent_response"] = answer
    state["citations"] = citations
    return state

def structured_data_node(state: AgentState) -> AgentState:
    executor = create_structured_agent()
    result = executor.invoke({"input": state["query"]})
    state["agent_response"] = result["output"]
    return state

def sar_node(state: AgentState) -> AgentState:
    # For simplicity, assume we extract alert fields from the query.
    # In a real scenario, you'd parse or pass structured JSON.
    # Here we use a dummy extraction for demo – but in hackathon you can pass sample_alerts.json.
    # We'll handle via a hardcoded test case in main.py.
    # This node expects the state to contain a pre‑parsed alert dict.
    # We'll implement a safer version: if query contains "alert #A4821" we use that sample.
    # For full flexibility, we parse from query using regex or pass via state.
    # To keep the code runnable, we'll use a simple simulated extraction.
    chain = create_sar_chain()
    # Dummy data – in real usage you would populate from a parsed dictionary
    # We'll rely on main.py to pass the alert JSON in the state (extra field).
    # For hackathon, we add a field 'sar_input' in state (optional).
    if "sar_input" in state:
        inp = state["sar_input"]
    else:
        # Fallback: extract from query (very basic)
        inp = {
            "alert_id": re.search(r"alert #(\w+)", state["query"], re.I).group(1) if re.search(r"alert #(\w+)", state["query"], re.I) else "unknown",
            "alert_type": "suspicious transaction",
            "customer_behaviour": "multiple rapid payments",
            "amounts": "£47,500",
            "date_range": "6 days",
            "analyst_notes": "unusual pattern"
        }
    narrative = chain.run(inp)
    state["agent_response"] = narrative
    return state

# ------------------------------------------------------------------
# Build graph
# ------------------------------------------------------------------
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("rag_policy_agent", rag_policy_node)
    workflow.add_node("structured_data_agent", structured_data_node)
    workflow.add_node("sar_drafting_agent", sar_node)

    workflow.set_entry_point("router")

    # Conditional edges from router
    workflow.add_conditional_edges(
        "router",
        lambda state: state["query_type"],
        {
            "policy": "rag_policy_agent",
            "structured_data": "structured_data_agent",
            "sar": "sar_drafting_agent",
            "refused": END
        }
    )

    workflow.add_edge("rag_policy_agent", END)
    workflow.add_edge("structured_data_agent", END)
    workflow.add_edge("sar_drafting_agent", END)

    return workflow.compile()