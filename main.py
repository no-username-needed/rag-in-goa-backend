import os
import time
import httpx
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from groq import AsyncGroq

app = FastAPI(title="Voice-Enabled RAG Agent")

# Allow the frontend to communicate with this backend safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Securely load your API keys from Render
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

groq_client = AsyncGroq(api_key=GROQ_API_KEY)

# THE FIX: Using the lightweight cloud API instead of the heavy local model
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the local database you uploaded
vector_db = FAISS.load_local(".", embeddings, allow_dangerous_deserialization=True)

# --- STRUCTURED I/O MODELS ---
class RAGResponse(BaseModel):
    transcription: str
    answer: str
    latency_ms: float
    guardrail_triggered: bool = False
    error: str = None

# --- ORCHESTRATION & RETRIES ---
@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.2))
async def transcribe_audio_sarvam(audio_bytes: bytes) -> str:
    """Uses Sarvam API with an automatic retry harness."""
    url = "https://api.sarvam.ai/speech-to-text/translate"
    headers = {"api-subscription-key": SARVAM_API_KEY}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, files=files, timeout=10.0)
        response.raise_for_status()
        return response.json().get("transcript", "")

async def generate_fast_answer(query: str, context: str) -> str:
    """Uses Groq to generate answers in milliseconds."""
    prompt = f"Answer the query strictly based on the context. If the context does not contain the answer, reply exactly with 'I don't know'.\n\nContext: {context}\n\nQuery: {query}"
    
    response = await groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
        temperature=0.1,
    )
    return response.choices[0].message.content

# --- GUARDRAILS ---
def check_safety(query: str) -> bool:
    """Guardrail 1: Reject unsafe or off-topic inputs immediately."""
    unsafe_words = ["ignore previous instructions", "violence", "hate", "harm"]
    if any(word in query.lower() for word in unsafe_words):
        return False
    return True

def hallucination_check(answer: str, context: str) -> bool:
    """Guardrail 2: Ensure the answer is grounded in the retrieved context."""
    if "I don't know" in answer:
        return True
    
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    overlap = len(answer_words.intersection(context_words))
    return overlap > 0 

# --- MAIN ENDPOINT ---
@app.post("/ask", response_model=RAGResponse)
async def process_voice_query(audio_file: UploadFile = File(...)):
    start_time = time.time()
    try:
        # Step 1: Voice to Text
        audio_bytes = await audio_file.read()
        transcription = await transcribe_audio_sarvam(audio_bytes)
        
        # Apply Guardrail 1
        if not check_safety(transcription):
            return RAGResponse(
                transcription=transcription,
                answer="Query rejected: Unsafe or off-topic input detected.",
                latency_ms=(time.time() - start_time) * 1000,
                guardrail_triggered=True
            )

        # Step 2: Vector DB Retrieval
        retrieved_docs = vector_db.similarity_search(transcription, k=3)
        context_text = " ".join([doc.page_content for doc in retrieved_docs])

        # Step 3: Answer Generation
        answer = await generate_fast_answer(transcription, context_text)

        # Apply Guardrail 2
        if not hallucination_check(answer, context_text):
            answer = "Guardrail triggered: The generated answer could not be verified against the dataset."

        end_time = time.time()
        
        return RAGResponse(
            transcription=transcription,
            answer=answer,
            latency_ms=(end_time - start_time) * 1000
        )

    except Exception as e:
        return RAGResponse(
            transcription="Error",
            answer="The pipeline encountered a critical failure and safely recovered.",
            latency_ms=(time.time() - start_time) * 1000,
            error=str(e)
        )
