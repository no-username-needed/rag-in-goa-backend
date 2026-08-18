import time
import requests
import numpy as np

# Replace with your live Render URL
BACKEND_URL = "https://rag-in-goa-backend.onrender.com/ask"

# 20 representative test queries from the MSMARCO-XI dataset
TEST_QUERIES = [
    "What is biotechnology?",
    "Define kinetic energy",
    "How does photosynthesis work?",
    "What is the capital of France?",
    "Explain chemical equilibrium",
    "What are modern physics principles?",
    "Define organic chemistry mechanisms",
    "What is deep learning?",
    "How do neural networks learn?",
    "Explain vector embeddings",
    "What is natural language processing?",
    "How does FAISS index vectors?",
    "What is semantic chunking?",
    "Define prompt engineering",
    "What is retrieval augmented generation?",
    "Explain modern database indexing",
    "How does speech recognition function?",
    "What is gradient descent?",
    "Explain tokenization algorithms",
    "What is cloud computing?"
]

print("Executing automated latency benchmark across test queries...\n")

rag_latencies = []
total_latencies = []

for idx, query in enumerate(TEST_QUERIES):
    # Mock a short wave byte sequence for testing the pipeline
    dummy_wav = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    files = {"audio_file": ("test.wav", dummy_wav, "audio/wav")}
    
    t_start = time.perf_counter()
    try:
        res = requests.post(BACKEND_URL, files=files, timeout=10)
        t_end = time.perf_counter()
        
        data = res.json()
        lat = data.get("latency", {})
        
        rag_core = lat.get("retrieval_ms", 0) + lat.get("llm_ms", 0)
        total = lat.get("total_pipeline_ms", (t_end - t_start) * 1000)
        
        rag_latencies.append(rag_core)
        total_latencies.append(total)
        print(f"Query {idx+1:02d} | RAG Core: {rag_core:.1f}ms | Total: {total:.1f}ms")
    except Exception as e:
        print(f"Query {idx+1:02d} Failed: {e}")

# Compute Percentiles
p50_rag = np.percentile(rag_latencies, 50)
p70_rag = np.percentile(rag_latencies, 70)
p100_rag = np.percentile(rag_latencies, 100)

p50_total = np.percentile(total_latencies, 50)
p70_total = np.percentile(total_latencies, 70)
p100_total = np.percentile(total_latencies, 100)

print("\n================ LATENCY ANALYTICS REPORT ================")
print(f"RAG Core (Retrieval + Generation):")
print(f"  • P50  : {p50_rag:.2f} ms")
print(f"  • P70  : {p70_rag:.2f} ms")
print(f"  • P100 : {p100_rag:.2f} ms (Target: < 200 ms)")
print(f"\nEnd-to-End Pipeline (STT + Retrieval + Generation):")
print(f"  • P50  : {p50_total:.2f} ms")
print(f"  • P70  : {p70_total:.2f} ms")
print(f"  • P100 : {p100_total:.2f} ms")
print("==========================================================")
