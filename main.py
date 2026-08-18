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

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local(".", embeddings, allow_dangerous_deserialization=True)

class LatencyBreakdown(BaseModel):
    stt_ms: float
    retrieval_ms: float
    llm_ms: float
    total_pipeline_ms: float

class RAGResponse(BaseModel):
    transcription: str
    answer: str
    latency_ms: float
    latency: LatencyBreakdown
    guardrail_triggered: bool = False
    context_sources: list = Field(default_factory=list)
    error: str = None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(0.1))
async def transcribe_audio_sarvam(audio_bytes: bytes, content_type: str) -> str:
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": SARVAM_API_KEY}
    
    base_mime = content_type.split(";")[0] if content_type else "audio/webm"
    ext = "mp4" if "mp4" in base_mime or "aac" in base_mime else "webm"
    files = {"file": (f"audio.{ext}", audio_bytes, base_mime)}
    
    # We leave this at en-IN because Sarvam effortlessly captures Hindi/Bengali/regional languages as Hinglish/Benglish, which the LLM perfectly understands.
    data = {"language_code": "en-IN"}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, data=data, files=files, timeout=10.0)
        response.raise_for_status()
        return response.json().get("transcript", "")

# THE UPGRADE: The AI is now fully conversational and multi-lingual
async def generate_fast_answer(query: str, context: str) -> str:
    prompt = (
        "You are a highly intelligent, conversational AI assistant. "
        "1. Read the user's query carefully. The user might speak in English, Hindi, Bengali, or any Indian language (often written in Roman script). "
        "2. If the 'Context' provided below contains relevant information, use it to answer the question accurately. "
        "3. If the 'Context' is irrelevant or empty, IGNORE IT. Do not say 'I don't know'. Instead, answer the user naturally using your own vast knowledge. "
        "4. Always reply in the exact same language and tone that the user spoke to you in.\n\n"
        f"Context:\n{context}\n\n"
        f"User Query:\n{query}\n"
    )
    
    response = await groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b",
        temperature=0.3, # Slightly increased to allow for natural conversation
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()

def check_safety_and_topic(query: str) -> bool:
    blocked_keywords = ["ignore instructions", "bypass", "violence", "harm", "hate", "jailbreak"]
    return not any(word in query.lower() for word in blocked_keywords)

@app.post("/ask", response_model=RAGResponse)
async def process_voice_query(audio_file: UploadFile = File(...)):
    start_total = time.perf_counter()
    
    try:
        t0 = time.perf_counter()
        audio_bytes = await audio_file.read()
        browser_mime_type = audio_file.content_type
        transcription = await transcribe_audio_sarvam(audio_bytes, browser_mime_type)
        stt_duration = (time.perf_counter() - t0) * 1000
        
        if not check_safety_and_topic(transcription):
            total_duration = (time.perf_counter() - start_total) * 1000
            return RAGResponse(
                transcription=transcription,
                answer="Guardrail Blocked: Unsafe prompt detected.",
                latency_ms=total_duration,
                latency=LatencyBreakdown(stt_ms=stt_duration, retrieval_ms=0, llm_ms=0, total_pipeline_ms=total_duration),
                guardrail_triggered=True
            )

        t1 = time.perf_counter()
        retrieved_docs = vector_db.similarity_search(transcription, k=2)
        context_text = " ".join([doc.page_content for doc in retrieved_docs])
        doc_sources = [doc.metadata.get("source", "MSMARCO-XI") for doc in retrieved_docs]
        retrieval_duration = (time.perf_counter() - t1) * 1000

        # The strict Hallucination check is removed so the AI can freely chat
        t2 = time.perf_counter()
        answer = await generate_fast_answer(transcription, context_text)
        llm_duration = (time.perf_counter() - t2) * 1000

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
