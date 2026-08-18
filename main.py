import os
import time
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
async def transcribe_audio_groq(audio_bytes: bytes, content_type: str) -> str:
    base_mime = content_type.split(";")[0] if content_type else "audio/webm"
    ext = "mp4" if "mp4" in base_mime or "aac" in base_mime else "webm"
    
    # THE FIX: A Pan-Indian prompt! This covers Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi, Marathi, and English.
    context_prompt = "नमस्ते, নমস্কার, வணக்கம், నమస్కారం, ನಮಸ್ಕಾರ, നമസ്കാരം, કેમ છો, ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ, नमस्कार, Hello! Speaking multiple Indian languages."
    
    response = await groq_client.audio.transcriptions.create(
        file=(f"audio.{ext}", audio_bytes),
        model="whisper-large-v3",
        prompt=context_prompt,  # Primes the AI for the vast diversity of Indian scripts
        response_format="json"
    )
    return response.text

async def generate_fast_answer(query: str, context: str) -> str:
    prompt = (
        "You are an advanced conversational Voice AI for India. "
        "1. OVERCOME STT ERRORS: The transcription might scramble Hindi and Urdu scripts, or mess up blended languages (like Hinglish or Benglish). Look past these typos and understand the true intent. "
        "2. BLENDING: If the user naturally mixes English with an Indian language, reply in that same natural, blended conversational style. "
        "3. SCRIPT FIX: If the language is Hindi, ALWAYS use Devanagari script (हिंदी). NEVER use Urdu script unless specifically requested. "
        "4. VOICE BOT TONE: Your exact output will be read aloud by a Text-to-Speech engine. Do not use markdown (like ** or #). Write in a warm, concise, and highly conversational tone.\n\n"
        f"Database Context:\n{context}\n\n"
        f"User Query:\n{query}\n"
    )
    
    response = await groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b",
        temperature=0.3,
        max_tokens=150, # Shorter output so the Voice Bot doesn't talk forever
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
        
        # Audio is now routed to Groq's Whisper API instead of Sarvam
        transcription = await transcribe_audio_groq(audio_bytes, browser_mime_type)
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

        # Vector Search against Pinecone Cloud
        t1 = time.perf_counter()
        query_vector = embeddings.embed_query(transcription)
        query_vector_list = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        
        search_results = index.query(vector=query_vector_list, top_k=2, include_metadata=True)
        
        context_texts = []
        for match in search_results.get("matches", []):
            meta = match.get("metadata", {})
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
        return RAGResponse(
            transcription=transcription if 'transcription' in locals() else "Error",
            answer=f"DEBUG ERROR: {str(e)}",
            latency_ms=total_duration,
            latency=LatencyBreakdown(stt_ms=0, retrieval_ms=0, llm_ms=0, total_pipeline_ms=total_duration),
            error=str(e)
        )
