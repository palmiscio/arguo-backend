from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import asyncio
import os
import json
import time

app = FastAPI(title="Arguo API")

# CORS — permette chiamate dal browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ── API Keys (da variabili d'ambiente) ──────────────────────
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_KEY     = os.getenv("OPENAI_API_KEY", "")
GEMINI_KEY     = os.getenv("GEMINI_API_KEY", "")
MISTRAL_KEY    = os.getenv("MISTRAL_API_KEY", "")
GROQ_KEY       = os.getenv("GROQ_API_KEY", "")
PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY", "")

SYSTEM_PROMPT = """You are a senior legal counsel with expertise in international law, including US, EU, and civil law systems.
Answer legal questions clearly and directly — like a trusted lawyer talking to a client.
Be concise: maximum 2 sentences for the answer. No headers, no bullet lists, no markdown. Plain prose only.
Cite relevant law or regulation naturally only when directly asked or clearly relevant — do not default to any specific jurisdiction as an example.
Always respond in the same language as the question.
After your answer, add exactly 3 follow-up questions the user might want to explore, formatted as:
FOLLOW_UP_1: [question]
FOLLOW_UP_2: [question]
FOLLOW_UP_3: [question]"""

# ── Request/Response models ──────────────────────────────────
class QueryRequest(BaseModel):
    query: str

class ModelResult(BaseModel):
    id: str
    name: str
    text: str
    ok: bool
    skip: bool = False
    latency_ms: int = 0
    error: str = ""

class ConsensusResponse(BaseModel):
    results: list[ModelResult]
    consensus_score: int
    consensus_answer: str
    follow_up1: str = ""
    follow_up2: str = ""
    follow_up3: str = ""
    divergences: str
    model_alignments: dict

# ── Callers ──────────────────────────────────────────────────

async def call_claude(client: httpx.AsyncClient, query: str) -> ModelResult:
    if not ANTHROPIC_KEY:
        return ModelResult(id="claude", name="Claude", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
            json={"model": "claude-opus-4-5", "max_tokens": 4000, "system": SYSTEM_PROMPT,
                  "messages": [{"role": "user", "content": query}]},
            timeout=30
        )
        d = r.json()
        if "error" in d:
            raise Exception(d["error"]["message"])
        text = d["content"][0]["text"]
        return ModelResult(id="claude", name="Claude", text=text, ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="claude", name="Claude", text="", ok=False, error=str(e))

async def call_gpt(client: httpx.AsyncClient, query: str) -> ModelResult:
    if not OPENAI_KEY:
        return ModelResult(id="gpt", name="GPT-4o", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-4o", "max_tokens": 4000,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]},
            timeout=30
        )
        d = r.json()
        if "error" in d:
            raise Exception(d["error"]["message"])
        text = d["choices"][0]["message"]["content"]
        return ModelResult(id="gpt", name="GPT-4o", text=text, ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="gpt", name="GPT-4o", text="", ok=False, error=str(e))

async def call_gemini(client: httpx.AsyncClient, query: str) -> ModelResult:
    if not GEMINI_KEY:
        return ModelResult(id="gemini", name="Gemini", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}",
            headers={"Content-Type": "application/json"},
            json={"system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                  "contents": [{"parts": [{"text": query}]}],
                  "generationConfig": {"maxOutputTokens": 4000}},
            timeout=30
        )
        d = r.json()
        if "error" in d:
            raise Exception(d["error"]["message"])
        text = d["candidates"][0]["content"]["parts"][0]["text"]
        return ModelResult(id="gemini", name="Gemini", text=text, ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="gemini", name="Gemini", text="", ok=False, error=str(e))

async def call_mistral(client: httpx.AsyncClient, query: str) -> ModelResult:
    if not MISTRAL_KEY:
        return ModelResult(id="mistral", name="Mistral", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_KEY}", "Content-Type": "application/json"},
            json={"model": "mistral-large-latest", "max_tokens": 4000,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]},
            timeout=30
        )
        d = r.json()
        if "error" in d:
            raise Exception(str(d["error"]))
        text = d["choices"][0]["message"]["content"]
        return ModelResult(id="mistral", name="Mistral", text=text, ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="mistral", name="Mistral", text="", ok=False, error=str(e))

async def call_groq(client: httpx.AsyncClient, query: str) -> ModelResult:
    if not GROQ_KEY:
        return ModelResult(id="groq", name="Llama 3 (Groq)", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "max_tokens": 1500,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]},
            timeout=30
        )
        d = r.json()
        if "error" in d:
            raise Exception(str(d["error"]))
        text = d["choices"][0]["message"]["content"]
        return ModelResult(id="groq", name="Llama 3 (Groq)", text=text, ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="groq", name="Llama 3 (Groq)", text="", ok=False, error=str(e))

async def call_perplexity(client: httpx.AsyncClient, query: str) -> ModelResult:
    if not PERPLEXITY_KEY:
        return ModelResult(id="perplexity", name="Perplexity", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {PERPLEXITY_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-sonar-large-128k-online", "max_tokens": 4000,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]},
            timeout=30
        )
        d = r.json()
        if "error" in d:
            raise Exception(str(d["error"]))
        text = d["choices"][0]["message"]["content"]
        return ModelResult(id="perplexity", name="Perplexity", text=text, ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="perplexity", name="Perplexity", text="", ok=False, error=str(e))

# ── Synthesis ────────────────────────────────────────────────

async def synthesize(client: httpx.AsyncClient, query: str, results: list[ModelResult]) -> dict:
    valid = [r for r in results if r.ok and r.text]
    if not valid:
        return {
            "consensusScore": 50,
            "consensusAnswer": "No response available.",
            "follow_up1": "What are the legal requirements?",
            "follow_up2": "Are there any exceptions?",
            "follow_up3": "What remedies are available?",
            "divergences": "",
            "modelAlignments": {}
        }

    # Build short summaries - max 200 chars each
    summaries = []
    for r in valid:
        text = r.text[:200].strip()
        if text:
            summaries.append(f"{r.name}: {text}")
    summary_block = "\n".join(summaries)

    prompt = f"""You are a legal expert. Based on these AI responses, write a 2-sentence verdict in the same language as the question. Be complete, never cut off.

Question: {query}
Responses: {summary_block}

Reply with ONLY this format (nothing else, no JSON):
VERDICT: [your 2-sentence verdict here]
Q1: [short follow-up question, max 8 words]
Q2: [short follow-up question, max 8 words]
Q3: [short follow-up question, max 8 words]"""

    try:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={{"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}},
            json={{"model": "claude-opus-4-5", "max_tokens": 300,
                  "messages": [{{"role": "user", "content": prompt}}]}},
            timeout=30
        )
        text = r.json()["content"][0]["text"].strip()
        
        # Parse the simple format
        verdict = ""
        q1, q2, q3 = "", "", ""
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("VERDICT:"):
                verdict = line[8:].strip()
            elif line.startswith("Q1:"):
                q1 = line[3:].strip()
            elif line.startswith("Q2:"):
                q2 = line[3:].strip()
            elif line.startswith("Q3:"):
                q3 = line[3:].strip()
        
        if not verdict:
            verdict = valid[0].text[:200]
        if not q1:
            q1 = "What are the notice requirements?"
        if not q2:
            q2 = "What remedies are available?"
        if not q3:
            q3 = "Can damages be claimed?"

    except Exception as e:
        verdict = valid[0].text[:200]
        q1 = "What are the notice requirements?"
        q2 = "What remedies are available?"
        q3 = "Can damages be claimed?"

    score = min(95, 60 + len(valid) * 10)

    return {{
        "consensusScore": score,
        "consensusAnswer": verdict,
        "follow_up1": q1,
        "follow_up2": q2,
        "follow_up3": q3,
        "divergences": "",
        "modelAlignments": {{r.name: "agree" for r in valid}}
    }}


# ── Endpoints ────────────────────────────────────────────────

@app.get("/")
def root():
    active = []
    if ANTHROPIC_KEY:  active.append("claude")
    if OPENAI_KEY:     active.append("gpt")
    if GEMINI_KEY:     active.append("gemini")
    if MISTRAL_KEY:    active.append("mistral")
    if GROQ_KEY:       active.append("groq")
    if PERPLEXITY_KEY: active.append("perplexity")
    return {"status": "ok", "service": "Arguo API", "active_models": active}

@app.post("/analyze", response_model=ConsensusResponse)
async def analyze(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query vuota")

    async with httpx.AsyncClient() as client:
        # Chiama tutti i modelli in parallelo
        results = await asyncio.gather(
            call_claude(client, req.query),
            call_gpt(client, req.query),
            call_gemini(client, req.query),
            call_mistral(client, req.query),
            call_groq(client, req.query),
            call_perplexity(client, req.query),
        )
        results = list(results)

        if not any(r.ok for r in results):
            raise HTTPException(status_code=503, detail="Nessun modello disponibile")

        # Sintetizza il consenso
        syn = await synthesize(client, req.query, results)

    return ConsensusResponse(
        results=results,
        consensus_score=syn.get("consensusScore", 50),
        consensus_answer=syn.get("consensusAnswer", ""),
        follow_up1=syn.get("followUp1", ""),
        follow_up2=syn.get("followUp2", ""),
        follow_up3=syn.get("followUp3", ""),
        divergences=syn.get("divergences", ""),
        model_alignments=syn.get("modelAlignments", {}),
    )

@app.get("/health")
def health():
    return {"status": "healthy"}
