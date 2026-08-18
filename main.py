import os
import time
import httpx
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from pinecone import Pinecone
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
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

groq_client = AsyncGroq(api_key=GROQ_API_KEY)

# Initialize direct connection to Pinecone Cloud
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("rag-in-goa")
embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
    
    data = {"language_code": "en-IN"}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, data=data, files=files, timeout=10.0)
        response.raise_for_status()
        return response.json().get("transcript", "")

async def generate_fast_answer(query: str, context: str) -> str:
    prompt = (
        "You are a highly intelligent, conversational AI assistant. "
        "1. Read the user's query carefully (which may be in English, Hindi, Bengali, or any Indian language). "
        "2. If the 'Database Context' below contains relevant information, use it to answer accurately. "
        "3. If the context is empty or irrelevant, IGNORE IT and answer naturally using your own vast internal knowledge. "
        "4. Always reply in the exact same language and conversational tone that the user spoke to you in.\n\n"
        f"Database Context:\n{context}\n\n"
        f"User Query:\n{query}\n"
    )
    
    response = await groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-70b-versatile",
        temperature=0.3,
        max_tokens=250,
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

        # Vector Search against Pinecone Cloud with foolproof safety checks
        t1 = time.perf_counter()
        query_vector = embeddings.embed_query(transcription)
        query_vector_list = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        
        search_results = index.query(vector=query_vector_list, top_k=2, include_metadata=True)
        
        context_texts = []
        for match in search_results.get("matches", []):
            meta = match.get("metadata", {})
            # Safely check for either 'text' or page_content keys
            text_val = meta.get("text") or meta.get("page_content") or ""
            if text_val:
                context_texts.append(text_val)
                
        context_text = " ".join(context_texts)
        retrieval_duration = (time.perf_counter() - t1) * 1000

        # LLM Generation
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
            context_sources=["Pinecone Cloud Index: rag-in-goa"]
        )

    except Exception as e:
        total_duration = (time.perf_counter() - start_total) * 1000
        # Return the exact exception string during debugging so you can see it instantly if anything arises
        return RAGResponse(
            transcription=transcription if 'transcription' in locals() else "Error",
            answer="Pipeline encountered an unrecoverable failure.",
            latency_ms=total_duration,
            latency=LatencyBreakdown(stt_ms=0, retrieval_ms=0, llm_ms=0, total_pipeline_ms=total_duration),
            error=str(e)
        )
