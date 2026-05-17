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

SYSTEM_PROMPT = """You are a senior legal counsel with deep expertise in international, US, EU, and civil law systems.
Answer legal questions with precision and specificity — cite the most relevant legal principle, doctrine, or rule that applies.
Be concise but substantive: 2-3 sentences maximum. No headers, no bullet lists, no markdown. Plain prose only.
Include the key legal standard or test that governs the issue when relevant (e.g. material breach standard, frustration doctrine, good faith requirement).
If jurisdiction matters, note it briefly. Respond in the same language as the question."""


class QueryRequest(BaseModel):
    query: str


class DocumentRequest(BaseModel):
    query: str
    document_text: str
    document_name: str = ""
    document_base64: str = ""
    document_ext: str = ""


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


# ── Model callers ─────────────────────────────────────────────

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
        return ModelResult(id="claude", name="Claude", text=d["content"][0]["text"], ok=True, latency_ms=int((time.time()-t0)*1000))
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
        return ModelResult(id="gpt", name="GPT-4o", text=d["choices"][0]["message"]["content"], ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="gpt", name="GPT-4o", text="", ok=False, error=str(e)[:100])


async def call_gemini(client, query):
    if not GEMINI_KEY:
        return ModelResult(id="gemini", name="Gemini", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}",
            headers={"Content-Type": "application/json"},
            json={"system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                  "contents": [{"parts": [{"text": query}]}],
                  "generationConfig": {"maxOutputTokens": 400}},
            timeout=30
        )
        d = r.json()
        if "error" in d:
            raise Exception(str(d["error"]))
        return ModelResult(id="gemini", name="Gemini", text=d["candidates"][0]["content"]["parts"][0]["text"], ok=True, latency_ms=int((time.time()-t0)*1000))
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
        return ModelResult(id="mistral", name="Mistral", text=d["choices"][0]["message"]["content"], ok=True, latency_ms=int((time.time()-t0)*1000))
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
        return ModelResult(id="groq", name="Llama 3 (Groq)", text=d["choices"][0]["message"]["content"], ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="groq", name="Llama 3 (Groq)", text="", ok=False, error=str(e)[:100])


async def call_gpt_oss(client, query):
    if not GROQ_KEY:
        return ModelResult(id="mixtral_groq", name="GPT-OSS 120B", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": "openai/gpt-oss-120b", "max_tokens": 400,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]},
            timeout=30
        )
        d = r.json()
        if "error" in d:
            raise Exception(str(d["error"]))
        return ModelResult(id="mixtral_groq", name="GPT-OSS 120B", text=d["choices"][0]["message"]["content"], ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="mixtral_groq", name="GPT-OSS 120B", text="", ok=False, error=str(e)[:100])


async def call_qwen(client, query):
    if not GROQ_KEY:
        return ModelResult(id="deepseek", name="Qwen3 32B", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": "qwen/qwen3-32b", "max_tokens": 400,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]},
            timeout=30
        )
        d = r.json()
        if "error" in d:
            raise Exception(str(d["error"]))
        import re
        text = re.sub(r'<think>.*?</think>', '', d["choices"][0]["message"]["content"], flags=re.DOTALL).strip()
        return ModelResult(id="deepseek", name="Qwen3 32B", text=text, ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="deepseek", name="Qwen3 32B", text="", ok=False, error=str(e)[:100])


# ── Synthesis ─────────────────────────────────────────────────

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

    summaries = "\n".join([f"- {r.name}: {r.text[:300]}" for r in valid])

    prompt = (
        f"You are a legal verification engine. Analyze these AI responses to the same question.\n\n"
        f"Question: {query}\n\n"
        f"Responses:\n{summaries}\n\n"
        f"Output exactly this format (no other text):\n"
        f"VERDICT: [2 complete sentences summarizing the consensus, same language as question]\n"
        f"SCORE: [integer 50-95 representing how much the responses agree. 95=perfect agreement, 50=major disagreements]\n"
        f"DIVERGENCES: [one sentence describing key differences, or 'None' if all agree]\n"
        f"ALIGN_JSON: {{\"ModelName\": \"agree|partial|disagree\", ...}} for each model\n"
        f"Q1: [follow-up question max 7 words, same language as question]\n"
        f"Q2: [follow-up question max 7 words, same language as question]\n"
        f"Q3: [follow-up question max 7 words, same language as question]"
    )

    verdict = valid[0].text[:250]
    score = 75
    divergences = ""
    alignments = {r.name: "agree" for r in valid}
    q1 = "What are the notice requirements?"
    q2 = "What remedies are available?"
    q3 = "Can damages be claimed?"

    if ANTHROPIC_KEY:
        try:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json={"model": "claude-opus-4-5", "max_tokens": 400,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=25
            )
            text = r.json()["content"][0]["text"].strip()
            import json as _json
            import re as _re
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("VERDICT:"):
                    verdict = line[8:].strip()
                elif line.startswith("SCORE:"):
                    try:
                        score = int(_re.search(r"\d+", line[6:]).group())
                        score = min(95, max(50, score))
                    except:
                        pass
                elif line.startswith("DIVERGENCES:"):
                    div = line[12:].strip()
                    divergences = "" if div.lower() == "none" else div
                elif line.startswith("ALIGN_JSON:"):
                    try:
                        json_str = line[11:].strip()
                        parsed = _json.loads(json_str)
                        alignments = {k: v for k, v in parsed.items()}
                    except:
                        pass
                elif line.startswith("Q1:"):
                    q1 = line[3:].strip()
                elif line.startswith("Q2:"):
                    q2 = line[3:].strip()
                elif line.startswith("Q3:"):
                    q3 = line[3:].strip()
        except Exception:
            pass

    return {
        "consensusScore": score,
        "consensusAnswer": verdict,
        "follow_up1": q1,
        "follow_up2": q2,
        "follow_up3": q3,
        "divergences": divergences,
        "modelAlignments": alignments
    }


# ── Document text extraction ──────────────────────────────────

def extract_text(b64, ext):
    try:
        data = base64.b64decode(b64)
        if ext == "pdf" and HAS_PDF:
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            return " ".join([p.extract_text() or "" for p in reader.pages])[:4000]
        elif ext in ("docx", "doc") and HAS_DOCX:
            doc = DocxDocument(io.BytesIO(data))
            return " ".join([p.text for p in doc.paragraphs])[:4000]
    except Exception:
        pass
    return ""


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/")
def root():
    active = []
    if ANTHROPIC_KEY:  active.append("claude")
    if OPENAI_KEY:     active.append("gpt")
    if GEMINI_KEY:     active.append("gemini")
    if MISTRAL_KEY:    active.append("mistral")
    if GROQ_KEY:       active.extend(["groq", "mixtral_groq", "deepseek"])
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
            call_gpt_oss(client, req.query),
            call_qwen(client, req.query),
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


@app.post("/analyze-document", response_model=ConsensusResponse)
async def analyze_document(req: DocumentRequest):
    doc_text = req.document_text
    if req.document_base64 and req.document_ext:
        extracted = extract_text(req.document_base64, req.document_ext)
        if extracted:
            doc_text = extracted

    if not doc_text.strip():
        raise HTTPException(status_code=400, detail="Empty document")

    query = req.query if req.query.strip() else "Analyze this legal document. Summarize key clauses, obligations, risks, and notable terms."
    full_query = f"{query}\n\nDocument content:\n{doc_text[:3000]}"

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            call_claude(client, full_query),
            call_gemini(client, full_query),
            call_mistral(client, full_query),
            call_groq(client, full_query),
            call_gpt_oss(client, full_query),
            call_qwen(client, full_query),
        )
        results = list(results)

        if not any(r.ok for r in results):
            raise HTTPException(status_code=503, detail="No model available")

        syn = await synthesize(client, query, results)

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
