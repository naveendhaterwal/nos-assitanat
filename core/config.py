import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import AsyncQdrantClient
from fastembed import TextEmbedding

load_dotenv()

# Initialize Qdrant Client
qdrant = AsyncQdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Initialize FastEmbed (Produces exact same 384d BGE embeddings as Cloudflare!)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Initialize LLMs based on diagram architecture
router_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
generator_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
troubleshooting_llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.6)
vision_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.2, 
    api_key=os.getenv("GOOGLE_API_KEY", "AIzaSyDaRqRrEaN-wMI57qGfRaZxSQUKJOtlDQQ")
)
