from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import asyncio
import os
import time
import base64
import io
try:
    import PyPDF2
    HAS_PDF = True
except:
    HAS_PDF = False
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except:
    HAS_DOCX = False

app = FastAPI(title="Arguo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_KEY     = os.getenv("OPENAI_API_KEY", "")
GEMINI_KEY     = os.getenv("GEMINI_API_KEY", "")
MISTRAL_KEY    = os.getenv("MISTRAL_API_KEY", "")
GROQ_KEY       = os.getenv("GROQ_API_KEY", "")
PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY", "")

SYSTEM_PROMPT = """You are a senior legal counsel with expertise in international law.
Answer legal questions clearly and directly in 2 sentences maximum.
No headers, no bullet lists, no markdown. Plain prose only.
Never mention a specific jurisdiction unless the question asks for it.
Respond in the same language as the question."""


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


async def call_claude(client, query):
    if not ANTHROPIC_KEY:
        return ModelResult(id="claude", name="Claude", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
            json={"model": "claude-opus-4-5", "max_tokens": 400, "system": SYSTEM_PROMPT,
                  "messages": [{"role": "user", "content": query}]},
            timeout=30
        )
        d = r.json()
        if "error" in d:
            raise Exception(str(d["error"]))
        text = d["content"][0]["text"]
        return ModelResult(id="claude", name="Claude", text=text, ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="claude", name="Claude", text="", ok=False, error=str(e)[:100])


async def call_gpt(client, query):
    if not OPENAI_KEY:
        return ModelResult(id="gpt", name="GPT-4o", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-4o", "max_tokens": 400,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]},
            timeout=30
        )
        d = r.json()
        if "error" in d:
            raise Exception(str(d["error"]))
        text = d["choices"][0]["message"]["content"]
        return ModelResult(id="gpt", name="GPT-4o", text=text, ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="gpt", name="GPT-4o", text="", ok=False, error=str(e)[:100])


async def call_gemini(client, query):
    if not GEMINI_KEY:
        return ModelResult(id="gemini", name="Gemini", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}",
            headers={"Content-Type": "application/json"},
            json={"system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                  "contents": [{"parts": [{"text": query}]}],
                  "generationConfig": {"maxOutputTokens": 400}},
            timeout=30
        )
        d = r.json()
        if "error" in d:
            raise Exception(str(d["error"]))
        text = d["candidates"][0]["content"]["parts"][0]["text"]
        return ModelResult(id="gemini", name="Gemini", text=text, ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="gemini", name="Gemini", text="", ok=False, error=str(e)[:100])


async def call_mistral(client, query):
    if not MISTRAL_KEY:
        return ModelResult(id="mistral", name="Mistral", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_KEY}", "Content-Type": "application/json"},
            json={"model": "mistral-large-latest", "max_tokens": 400,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]},
            timeout=30
        )
        d = r.json()
        if "error" in d:
            raise Exception(str(d["error"]))
        text = d["choices"][0]["message"]["content"]
        return ModelResult(id="mistral", name="Mistral", text=text, ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="mistral", name="Mistral", text="", ok=False, error=str(e)[:100])


async def call_groq(client, query):
    if not GROQ_KEY:
        return ModelResult(id="groq", name="Llama 3 (Groq)", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "max_tokens": 400,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]},
            timeout=30
        )
        d = r.json()
        if "error" in d:
            raise Exception(str(d["error"]))
        text = d["choices"][0]["message"]["content"]
        return ModelResult(id="groq", name="Llama 3 (Groq)", text=text, ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="groq", name="Llama 3 (Groq)", text="", ok=False, error=str(e)[:100])


async def synthesize(client, query, results):
    valid = [r for r in results if r.ok and r.text]

    if not valid:
        return {
            "consensusScore": 0,
            "consensusAnswer": "No response available.",
            "follow_up1": "What are the legal requirements?",
            "follow_up2": "Are there any exceptions?",
            "follow_up3": "What remedies are available?",
            "divergences": "",
            "modelAlignments": {}
        }

    # Build summaries
    summaries = "\n".join([f"- {r.name}: {r.text[:250]}" for r in valid])

    prompt = (
        f"Question: {query}\n\n"
        f"AI responses:\n{summaries}\n\n"
        f"Write exactly:\n"
        f"VERDICT: [2 complete sentences summarizing consensus, same language as question]\n"
        f"Q1: [follow-up question max 7 words, same language]\n"
        f"Q2: [follow-up question max 7 words, same language]\n"
        f"Q3: [follow-up question max 7 words, same language]\n\n"
        f"Only output these 4 lines. No other text."
    )

    verdict = valid[0].text[:250]
    q1 = "What are the notice requirements?"
    q2 = "What remedies are available?"
    q3 = "Can damages be claimed?"

    if ANTHROPIC_KEY:
        try:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json={"model": "claude-opus-4-5", "max_tokens": 250,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=25
            )
            text = r.json()["content"][0]["text"].strip()
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
        except Exception:
            pass

    alignments = {}
    for r in valid:
        alignments[r.name] = "agree"

    return {
        "consensusScore": min(95, 55 + len(valid) * 12),
        "consensusAnswer": verdict,
        "follow_up1": q1,
        "follow_up2": q2,
        "follow_up3": q3,
        "divergences": "",
        "modelAlignments": alignments
    }


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
        raise HTTPException(status_code=400, detail="Empty query")

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            call_claude(client, req.query),
            call_gpt(client, req.query),
            call_gemini(client, req.query),
            call_mistral(client, req.query),
            call_groq(client, req.query),
        )
        results = list(results)

        if not any(r.ok for r in results):
            raise HTTPException(status_code=503, detail="No model available")

        syn = await synthesize(client, req.query, results)

    return ConsensusResponse(
        results=results,
        consensus_score=syn["consensusScore"],
        consensus_answer=syn["consensusAnswer"],
        follow_up1=syn["follow_up1"],
        follow_up2=syn["follow_up2"],
        follow_up3=syn["follow_up3"],
        divergences=syn["divergences"],
        model_alignments=syn["modelAlignments"],
    )


@app.get("/health")
def health():
    return {"status": "healthy"}


# ── Document analysis endpoint ────────────────────────────────

class DocumentRequest(BaseModel):
    query: str
    document_text: str
    document_name: str = ""
    document_base64: str = ""
    document_ext: str = ""


async def call_claude_doc(client, query, doc_text):
    if not ANTHROPIC_KEY:
        return ModelResult(id="claude", name="Claude", text="", ok=False, skip=True)
    t0 = time.time()
    prompt = f"Document: {doc_text[:3000]}\n\nQuestion: {query}" if query.strip() else f"Analyze this legal document and provide a concise summary of key clauses, obligations, risks, and notable terms:\n\n{doc_text[:3000]}"
    try:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
            json={"model": "claude-opus-4-5", "max_tokens": 400, "system": SYSTEM_PROMPT,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=40
        )
        d = r.json()
        if "error" in d:
            raise Exception(str(d["error"]))
        text = d["content"][0]["text"]
        return ModelResult(id="claude", name="Claude", text=text, ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="claude", name="Claude", text="", ok=False, error=str(e)[:100])


async def call_mistral_doc(client, query, doc_text):
    if not MISTRAL_KEY:
        return ModelResult(id="mistral", name="Mistral", text="", ok=False, skip=True)
    t0 = time.time()
    prompt = f"Document: {doc_text[:3000]}\n\nQuestion: {query}" if query.strip() else f"Analyze this legal document and provide a concise summary of key clauses, obligations, risks, and notable terms:\n\n{doc_text[:3000]}"
    try:
        r = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_KEY}", "Content-Type": "application/json"},
            json={"model": "mistral-large-latest", "max_tokens": 400,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]},
            timeout=40
        )
        d = r.json()
        if "error" in d:
            raise Exception(str(d["error"]))
        text = d["choices"][0]["message"]["content"]
        return ModelResult(id="mistral", name="Mistral", text=text, ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="mistral", name="Mistral", text="", ok=False, error=str(e)[:100])


async def call_groq_doc(client, query, doc_text):
    if not GROQ_KEY:
        return ModelResult(id="groq", name="Llama 3 (Groq)", text="", ok=False, skip=True)
    t0 = time.time()
    prompt = f"Document: {doc_text[:3000]}\n\nQuestion: {query}" if query.strip() else f"Analyze this legal document and provide a concise summary of key clauses, obligations, risks, and notable terms:\n\n{doc_text[:3000]}"
    try:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "max_tokens": 400,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]},
            timeout=40
        )
        d = r.json()
        if "error" in d:
            raise Exception(str(d["error"]))
        text = d["choices"][0]["message"]["content"]
        return ModelResult(id="groq", name="Llama 3 (Groq)", text=text, ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="groq", name="Llama 3 (Groq)", text="", ok=False, error=str(e)[:100])


def extract_text(b64: str, ext: str) -> str:
    try:
        data = base64.b64decode(b64)
        if ext == "pdf" and HAS_PDF:
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            return " ".join([p.extract_text() or "" for p in reader.pages])[:4000]
        elif ext == "docx" and HAS_DOCX:
            doc = DocxDocument(io.BytesIO(data))
            return " ".join([p.text for p in doc.paragraphs])[:4000]
    except Exception:
        pass
    return ""

@app.post("/analyze-document", response_model=ConsensusResponse)
async def analyze_document(req: DocumentRequest):
    doc_text = req.document_text
    if req.document_base64 and req.document_ext:
        extracted = extract_text(req.document_base64, req.document_ext)
        if extracted:
            doc_text = extracted
    if not doc_text.strip():
        raise HTTPException(status_code=400, detail="Empty document")

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            call_claude_doc(client, req.query, req.document_text),
            call_mistral_doc(client, req.query, req.document_text),
            call_groq_doc(client, req.query, req.document_text),
        )
        results = list(results)

        if not any(r.ok for r in results):
            raise HTTPException(status_code=503, detail="No model available")

        syn = await synthesize(client, req.query or "Analyze this document", results)

    return ConsensusResponse(
        results=results,
        consensus_score=syn["consensusScore"],
        consensus_answer=syn["consensusAnswer"],
        follow_up1=syn["follow_up1"],
        follow_up2=syn["follow_up2"],
        follow_up3=syn["follow_up3"],
        divergences=syn["divergences"],
        model_alignments=syn["modelAlignments"],
    )
