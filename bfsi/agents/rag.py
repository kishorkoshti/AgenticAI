# agents/rag.py
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import OPENAI_CHAT_MODEL, OPENAI_API_KEY, TOP_K, SIMILARITY_THRESHOLD, CHROMA_DIR

def load_rag_agent():
    """Return a callable that takes a query and returns (answer, citations)."""
    # Load vector store
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    
    retriever = vector_store.as_retriever(
        search_kwargs={"k": TOP_K, "score_threshold": SIMILARITY_THRESHOLD}
    )

    # LLM
    llm = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.0
    )

    # Prompt
    system_prompt = """
    You are an AML policy expert. Answer the query using only the provided context.
    If the context does not contain enough information, reply: "I cannot answer based on the provided policy documents."
    Do not use your general knowledge.
    
    Context:
    {context}
    
    Query: {input}
    
    Answer (with citations to the source sections):
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Create chains
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    def invoke(query: str):
        result = rag_chain.invoke({"input": query})
        answer = result["answer"]
        citations = []
        for doc in result.get("context", []):
            citations.append(doc.metadata.get("source_section", "unknown"))
        return answer, citations

    return invoke