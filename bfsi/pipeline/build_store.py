from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


from config import CHUNK_SIZE, CHUNK_OVERLAP, OPENAI_EMBEDDING_MODEL, OPENAI_API_KEY, CHROMA_DIR
import re

def detect_section_header(line: str) -> bool:
    
    line = line.strip()
    if not line or len(line) > 100:
        return False
    if line.isupper():
        return True
    if line.endswith(':'):
        return True
    if any(marker in line for marker in ['Section', 'CHAPTER', 'Article', '§', '#', 'TYPOLOGY']):
        return True
    if re.match(r'^(\d+\.?\d*|\w\.|[IVXLC]+\.?)\s+', line):
        return True
    return False

def build_vector_store(policy_text: str):
    """Chunk policy text, embed with OpenAI, persist Chroma."""
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(policy_text)

    
    lines = policy_text.split('\n')
    header_map = [None] * len(policy_text)
    current_header = "unknown"
    pos = 0
    for line in lines:
        if detect_section_header(line):
            current_header = line.strip()[:80]   # truncate long headers
        for _ in range(len(line) + 1):   # +1 for newline
            if pos < len(header_map):
                header_map[pos] = current_header
            pos += 1

    metadatas = []
    for i, chunk in enumerate(chunks):
        start = policy_text.find(chunk)
        if start != -1:
            section = header_map[start] if header_map[start] else "unknown"
        else:
            section = "unknown"
        metadatas.append({"source_section": section, "chunk_id": i})

    
    embeddings = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_DIR
    )
    vector_store.persist()
    return vector_store