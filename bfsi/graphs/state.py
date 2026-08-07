from typing import TypedDict, List

class AgentState(TypedDict):
    query: str
    query_type: str      # 'policy' | 'structured_data' | 'sar' | 'refused'
    agent_response: str
    citations: List[str]   # used only by RAG agent