from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from config import LLM_MODEL, PROJECT_ID, LOCATION

def create_sar_chain():
    """Return an LLMChain that produces a 4‑section SAR narrative."""
    template = """
    You are a SAR (Suspicious Activity Report) writer. Based strictly on the alert data below, produce a SAR narrative with exactly four labelled sections.

    Alert data:
    Alert ID: {alert_id}
    Alert type: {alert_type}
    Customer behaviour: {customer_behaviour}
    Amounts: {amounts}
    Date range: {date_range}
    Analyst notes: {analyst_notes}

    Write the SAR in the following format:

    Subject Summary:
    [One paragraph summarising the subject and alert]

    Suspicious Activity Description:
    [Detailed description of the suspicious activity, using only provided facts]

    Supporting Evidence:
    [List of evidence from the alert data]

    Recommended Action:
    [One sentence recommending next step]

    Do not add any amounts, dates, or details not present in the input. If a field is sparse, write "Not provided".
    """
    prompt = PromptTemplate(
        input_variables=["alert_id", "alert_type", "customer_behaviour",
                         "amounts", "date_range", "analyst_notes"],
        template=template
    )
    llm = ChatVertexAI(model_name=LLM_MODEL, project=PROJECT_ID, location=LOCATION, temperature=0.2)
    return LLMChain(llm=llm, prompt=prompt)