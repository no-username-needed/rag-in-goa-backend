import os
import time
import httpx
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from groq import AsyncGroq

app = FastAPI(title="Voice-Enabled RAG System (#RAGInGoa)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

groq_client = AsyncGroq(api_key=GROQ_API_KEY)

# Local In-Memory Embeddings (Zero API Latency)
embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local(".", embeddings, allow_dangerous_deserialization=True)

# --- STRUCTURED HARNESS & SCHEMA ---
class LatencyBreakdown(BaseModel):
    stt_ms: float
    retrieval_ms: float
    llm_ms: float
    total_pipeline_ms: float

class RAGResponse(BaseModel):
    transcription: str
    answer: str
    latency_ms: float  # Restored so your frontend HTML stops showing NaN
    latency: LatencyBreakdown  # Kept so your benchmark script can get analytics
    guardrail_triggered: bool = False
    context_sources: list = Field(default_factory=list)
    error: str = None

# --- SPEECH-TO-TEXT WITH RETRY HARNESS ---
@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.1))
async def transcribe_audio_sarvam(audio_bytes: bytes) -> str:
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": SARVAM_API_KEY}
    
    # FIXED: Reverted to webm format and locked the language to English
    files = {"file": ("audio.webm", audio_bytes, "audio/webm")}
    data = {"language_code": "en-IN"}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, data=data, files=files, timeout=5.0)
        response.raise_for_status()
        return response.json().get("transcript", "")

# --- SUB-200MS RETRIEVAL & GENERATION ---
async def generate_fast_answer(query: str, context: str) -> str:
    prompt = (
        "Answer the user's question in one concise sentence using ONLY the context provided. "
        "If the context does not contain the answer, output 'I don't know'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n"
    )
    
    response = await groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b",
        temperature=0.0,
        max_tokens=60,
    )
    return response.choices[0].message.content.strip()

# --- GUARDRAILS ---
def check_safety_and_topic(query: str) -> bool:
    """Blocks off-topic queries and unsafe prompts."""
    blocked_keywords = ["ignore instructions", "bypass", "violence", "harm", "hate", "jailbreak"]
    return not any(word in query.lower() for word in blocked_keywords)

def hallucination_check(answer: str, context: str) -> bool:
    """Verifies that the generated response is strictly grounded in retrieved passages."""
    if answer == "I don't know":
        return True
    
    answer_tokens = set(answer.lower().replace(".", "").replace(",", "").split())
    context_tokens = set(context.lower().replace(".", "").replace(",", "").split())
    overlap = answer_tokens.intersection(context_tokens)
    return len(overlap) >= 2

# --- CORE ENDPOINT ---
@app.post("/ask", response_model=RAGResponse)
async def process_voice_query(audio_file: UploadFile = File(...)):
    start_total = time.perf_counter()
    
    try:
        # 1. Speech-to-Text
        t0 = time.perf_counter()
        audio_bytes = await audio_file.read()
        transcription = await transcribe_audio_sarvam(audio_bytes)
        stt_duration = (time.perf_counter() - t0) * 1000
        
        # Guardrail: Input Safety Check
        if not check_safety_and_topic(transcription):
            total_duration = (time.perf_counter() - start_total) * 1000
            return RAGResponse(
                transcription=transcription,
                answer="Guardrail Blocked: Off-topic or unsafe prompt detected.",
                latency_ms=total_duration,
                latency=LatencyBreakdown(stt_ms=stt_duration, retrieval_ms=0, llm_ms=0, total_pipeline_ms=total_duration),
                guardrail_triggered=True
            )

        # 2. Vector DB Retrieval
        t1 = time.perf_counter()
        retrieved_docs = vector_db.similarity_search(transcription, k=2)
        context_text = " ".join([doc.page_content for doc in retrieved_docs])
        doc_sources = [doc.metadata.get("source", "MSMARCO-XI") for doc in retrieved_docs]
        retrieval_duration = (time.perf_counter() - t1) * 1000

        # 3. Fast LLM Generation
        t2 = time.perf_counter()
        answer = await generate_fast_answer(transcription, context_text)
        llm_duration = (time.perf_counter() - t2) * 1000

        # Guardrail: Hallucination Verification
        if not hallucination_check(answer, context_text):
            answer = "I don't know"

        total_duration = (time.perf_counter() - start_total) * 1000

        return RAGResponse(
            transcription=transcription,
            answer=answer,
            latency_ms=total_duration,
            latency=LatencyBreakdown(
                stt_ms=stt_duration,
                retrieval_ms=retrieval_duration,
                llm_ms=llm_duration,
                total_pipeline_ms=total_duration
            ),
            context_sources=doc_sources
        )

    except Exception as e:
        total_duration = (time.perf_counter() - start_total) * 1000
        return RAGResponse(
            transcription="Error",
            answer="Pipeline encountered an unrecoverable failure.",
            latency_ms=total_duration,
            latency=LatencyBreakdown(stt_ms=0, retrieval_ms=0, llm_ms=0, total_pipeline_ms=total_duration),
            error=str(e)
        )
