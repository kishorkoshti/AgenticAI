import sys
import json
import re
from pipeline.load_data import load_all_data
from pipeline.build_store import build_vector_store
from graphs.graph import build_graph
from graphs.state import AgentState

def redact_pii(text: str) -> str:
    """Remove common PII: emails, phone numbers, account numbers."""
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[REDACTED EMAIL]', text)
    text = re.sub(r'\b\d{10,}\b', '[REDACTED ACCOUNT]', text)
    text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[REDACTED CARD]', text)
    return text

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<query>\"")
        sys.exit(1)

    query = sys.argv[1]

    # Load data and build vector store (idempotent)
    alerts_df, typology_df, policy_text = load_all_data()
    # Build Chroma if not already built
    build_vector_store(policy_text)

    # Inject DataFrames into structured_data agent
    import agents.structured_data as sd
    sd.init_dataframes(alerts_df, typology_df)

    # Compile graph
    app = build_graph()

    # Prepare initial state
    state: AgentState = {
        "query": query,
        "query_type": "",
        "agent_response": "",
        "citations": []
    }

    # For SAR queries, optionally parse JSON from sample_alerts.json
    if "draft a sar" in query.lower() or "sar narrative" in query.lower():
        with open("data/sample_alerts.json", "r") as f:
            sample_alerts = json.load(f)
        # Find alert by ID in query (simple)
        import re
        match = re.search(r"alert #(\w+)", query, re.I)
        if match:
            alert_id = match.group(1)
            for alert in sample_alerts:
                if alert["alert_id"] == alert_id:
                    state["sar_input"] = alert
                    break
        if "sar_input" not in state:
            # Use first alert as fallback
            state["sar_input"] = sample_alerts[0]

    # Invoke graph
    final_state = app.invoke(state)

    # Output response with redaction
    response = final_state["agent_response"]
    if final_state.get("citations"):
        response += f"\n\nCitations: {', '.join(final_state['citations'])}"
    print(redact_pii(response))

if __name__ == "__main__":
    main()