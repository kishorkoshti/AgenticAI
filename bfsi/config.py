import os
from dotenv import load_dotenv

load_dotenv()



# Embeddings & Vector Store
EMBEDDING_MODEL = "text-embedding-004"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
SIMILARITY_THRESHOLD = 0.7

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"   # or "text-embedding-ada-002"
OPENAI_CHAT_MODEL = "gpt-4o"

# LLM
LLM_MODEL = "gemini-1.5-pro"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")          # put your CSV, TXT, JSON here
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_store")


