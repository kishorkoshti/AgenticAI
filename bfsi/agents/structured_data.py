import pandas as pd
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from config import LLM_MODEL, PROJECT_ID, LOCATION

# Global DataFrames (set once during startup)
_alerts_df = None
_typology_df = None

def init_dataframes(alerts_df, typology_df):
    global _alerts_df, _typology_df
    _alerts_df = alerts_df
    _typology_df = typology_df

# ------------------------------------------------------------------
# Tool 1: query_alerts_table
# ------------------------------------------------------------------
@tool
def query_alerts_table(risk_band: str = None, status: str = None, alert_type: str = None) -> dict:
    """
    Query alerts table with optional filters.
    Returns count, total amount, and a sample of rows as a string.
    """
    df = _alerts_df.copy()
    if risk_band:
        df = df[df["risk_band"].str.upper() == risk_band.upper()]
    if status:
        df = df[df["status"].str.upper() == status.upper()]
    if alert_type:
        df = df[df["alert_type"].str.upper() == alert_type.upper()]

    count = len(df)
    total = df["amount_gbp"].sum() if not df.empty else 0
    sample = df.head(5).to_string() if not df.empty else "No matching rows."
    return {"count": count, "total_amount_gbp": total, "sample": sample}

# ------------------------------------------------------------------
# Tool 2: query_typology_table
# ------------------------------------------------------------------
@tool
def query_typology_table(identifier: str) -> dict:
    """
    Look up typology by code or name.
    Returns threshold and reporting obligation.
    """
    df = _typology_df
    row = df[(df["typology_code"].str.lower() == identifier.lower()) |
             (df["typology_name"].str.lower() == identifier.lower())]
    if row.empty:
        return {"found": False, "message": f"Typology '{identifier}' not found."}
    return {
        "found": True,
        "threshold_value": row.iloc[0]["threshold_value"],
        "threshold_unit": row.iloc[0]["threshold_unit"],
        "reporting_obligation": row.iloc[0]["reporting_obligation"]
    }

# ------------------------------------------------------------------
# Agent Setup
# ------------------------------------------------------------------
def create_structured_agent():
    """Build a LangChain tool-calling agent."""
    llm = ChatVertexAI(model_name=LLM_MODEL, project=PROJECT_ID, location=LOCATION, temperature=0)

    tools = [query_alerts_table, query_typology_table]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data analyst. Answer questions using the available tools."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
    return executor