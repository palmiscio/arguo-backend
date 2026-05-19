from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import httpx
import asyncio
import os
import time
import base64
import io
import re
import json

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
ARGUO_SECRET   = os.getenv("ARGUO_API_SECRET", "")

def verify_secret(s: str) -> bool:
    if not ARGUO_SECRET:
        return True
    return s == ARGUO_SECRET

SYSTEM_PROMPT_BASE = """You are a senior legal counsel with deep expertise in international, US, EU, and civil law systems.
Answer legal questions with precision. You MUST follow this exact structure:

ANSWER: [1-2 sentences direct answer, same language as question]
JURISDICTION: [exact jurisdiction(s) — e.g. "US Federal Law", "EU Law", "Italian Civil Code", "Common Law"]
SOURCES: [specific law, statute, article — e.g. "UCC §2-609", "Art. 1453 Codice Civile", "GDPR Art. 17"]
CONFIDENCE: [HIGH / MEDIUM / LOW] — [one sentence explaining why]

Never omit any field. If no specific statute exists write: SOURCES: General common law principle — no specific statute."""

SYSTEM_PROMPT = SYSTEM_PROMPT_BASE

def build_system_prompt(jurisdiction_hint: str = "") -> str:
    if not jurisdiction_hint:
        return SYSTEM_PROMPT_BASE
    return SYSTEM_PROMPT_BASE + f"""

CRITICAL JURISDICTION INSTRUCTION: The user has explicitly selected {jurisdiction_hint}.
You MUST answer exclusively under {jurisdiction_hint}. 
Your JURISDICTION field MUST state exactly: {jurisdiction_hint}
Do NOT apply any other legal system."""


# ── Pydantic models ───────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    secret: str = ""
    jurisdiction: str = ""

class DocumentRequest(BaseModel):
    query: str
    document_text: str
    document_name: str = ""
    document_base64: str = ""
    document_ext: str = ""
    secret: str = ""
    jurisdiction: str = ""

class ModelResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    id: str
    name: str
    text: str
    ok: bool
    skip: bool = False
    latency_ms: int = 0
    error: str = ""

class ConsensusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    results: list[ModelResult]
    consensus_score: int
    consensus_answer: str
    follow_up1: str = ""
    follow_up2: str = ""
    follow_up3: str = ""
    divergences: str = ""
    model_alignments: dict = {}
    jurisdictions: list = []
    sources: list = []
    confidence_flags: list = []
    reliability_warning: str = ""
    disagreement_explanation: str = ""


# ── Model callers ─────────────────────────────────────────────

async def call_claude(client, query, jurisdiction_hint=""):
    if not ANTHROPIC_KEY:
        return ModelResult(id="claude", name="Claude", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
            json={"model": "claude-opus-4-5", "max_tokens": 600, "system": build_system_prompt(jurisdiction_hint),
                  "messages": [{"role": "user", "content": query}]},
            timeout=40
        )
        d = r.json()
        if "error" in d: raise Exception(str(d["error"]))
        return ModelResult(id="claude", name="Claude", text=d["content"][0]["text"], ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="claude", name="Claude", text="", ok=False, error=str(e)[:100])

async def call_gpt(client, query, jurisdiction_hint=""):
    if not OPENAI_KEY:
        return ModelResult(id="gpt", name="GPT-4o", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-4o", "max_tokens": 600,
                  "messages": [{"role": "system", "content": build_system_prompt(jurisdiction_hint)}, {"role": "user", "content": query}]},
            timeout=40
        )
        d = r.json()
        if "error" in d: raise Exception(str(d["error"]))
        return ModelResult(id="gpt", name="GPT-4o", text=d["choices"][0]["message"]["content"], ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="gpt", name="GPT-4o", text="", ok=False, error=str(e)[:100])

async def call_gemini(client, query, jurisdiction_hint=""):
    if not GEMINI_KEY:
        return ModelResult(id="gemini", name="Gemini", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}",
            headers={"Content-Type": "application/json"},
            json={"system_instruction": {"parts": [{"text": build_system_prompt(jurisdiction_hint)}]},
                  "contents": [{"parts": [{"text": query}]}],
                  "generationConfig": {"maxOutputTokens": 800}},
            timeout=40
        )
        d = r.json()
        if "error" in d: raise Exception(str(d["error"]))
        return ModelResult(id="gemini", name="Gemini", text=d["candidates"][0]["content"]["parts"][0]["text"], ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="gemini", name="Gemini", text="", ok=False, error=str(e)[:100])

async def call_mistral(client, query, jurisdiction_hint=""):
    if not MISTRAL_KEY:
        return ModelResult(id="mistral", name="Mistral", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_KEY}", "Content-Type": "application/json"},
            json={"model": "mistral-large-latest", "max_tokens": 600,
                  "messages": [{"role": "system", "content": build_system_prompt(jurisdiction_hint)}, {"role": "user", "content": query}]},
            timeout=40
        )
        d = r.json()
        if "error" in d: raise Exception(str(d["error"]))
        return ModelResult(id="mistral", name="Mistral", text=d["choices"][0]["message"]["content"], ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="mistral", name="Mistral", text="", ok=False, error=str(e)[:100])

async def call_groq(client, query, jurisdiction_hint=""):
    if not GROQ_KEY:
        return ModelResult(id="groq", name="Llama 3 (Groq)", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "max_tokens": 600,
                  "messages": [{"role": "system", "content": build_system_prompt(jurisdiction_hint)}, {"role": "user", "content": query}]},
            timeout=40
        )
        d = r.json()
        if "error" in d: raise Exception(str(d["error"]))
        return ModelResult(id="groq", name="Llama 3 (Groq)", text=d["choices"][0]["message"]["content"], ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="groq", name="Llama 3 (Groq)", text="", ok=False, error=str(e)[:100])

async def call_gpt_oss(client, query, jurisdiction_hint=""):
    if not GROQ_KEY:
        return ModelResult(id="mixtral_groq", name="GPT-OSS 120B", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": "openai/gpt-oss-120b", "max_tokens": 600,
                  "messages": [{"role": "system", "content": build_system_prompt(jurisdiction_hint)}, {"role": "user", "content": query}]},
            timeout=40
        )
        d = r.json()
        if "error" in d: raise Exception(str(d["error"]))
        return ModelResult(id="mixtral_groq", name="GPT-OSS 120B", text=d["choices"][0]["message"]["content"], ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="mixtral_groq", name="GPT-OSS 120B", text="", ok=False, error=str(e)[:100])

async def call_qwen(client, query, jurisdiction_hint=""):
    if not GROQ_KEY:
        return ModelResult(id="deepseek", name="Qwen3 32B", text="", ok=False, skip=True)
    t0 = time.time()
    try:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": "qwen/qwen3-32b", "max_tokens": 800,
                  "messages": [{"role": "system", "content": build_system_prompt(jurisdiction_hint)}, {"role": "user", "content": query}]},
            timeout=40
        )
        d = r.json()
        if "error" in d: raise Exception(str(d["error"]))
        text = re.sub(r'<think>.*?</think>', '', d["choices"][0]["message"]["content"], flags=re.DOTALL).strip()
        return ModelResult(id="deepseek", name="Qwen3 32B", text=text, ok=True, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return ModelResult(id="deepseek", name="Qwen3 32B", text="", ok=False, error=str(e)[:100])



def clean_text(t):
    """Remove markdown artifacts and clean text."""
    if not t:
        return t
    import re as _re
    t = _re.sub(r'\*+', '', t).strip()
    t = _re.sub(r'\s+', ' ', t).strip()
    return t if t else None

# ── Structured parser ─────────────────────────────────────────

def parse_structured(text):
    fields = {"answer": "", "jurisdiction": "", "sources": "", "confidence": "MEDIUM", "confidence_reason": "Auto-assessed"}
    lines = text.strip().split("\n")
    found = False
    for line in lines:
        line = line.strip().lstrip("*").strip()
        if line.upper().startswith("ANSWER:"):
            fields["answer"] = line[7:].strip()
            found = True
        elif line.upper().startswith("JURISDICTION:"):
            fields["jurisdiction"] = line[13:].strip()
            found = True
        elif line.upper().startswith("SOURCES:"):
            src = line[8:].strip()
            if src and "general common law" not in src.lower() and "no specific" not in src.lower():
                fields["sources"] = src
            found = True
        elif line.upper().startswith("CONFIDENCE:"):
            rest = line[11:].strip()
            for sep in [" — ", " - ", ": ", " – "]:
                if sep in rest:
                    parts = rest.split(sep, 1)
                    lvl = parts[0].strip().upper().replace("*","").replace("*","")
                    if lvl in ("HIGH","MEDIUM","LOW"):
                        fields["confidence"] = lvl
                        fields["confidence_reason"] = parts[1].strip()
                    break
            else:
                lvl = rest.strip().upper().split()[0] if rest.strip() else "MEDIUM"
                if lvl in ("HIGH","MEDIUM","LOW"):
                    fields["confidence"] = lvl
            found = True

    if not found or not fields["answer"]:
        fields["answer"] = text.strip()
        text_lower = text.lower()
        if any(w in text_lower for w in ["unclear","uncertain","may vary","depends","complex","disputed","varies"]):
            fields["confidence"] = "LOW"
            fields["confidence_reason"] = "Answer contains uncertainty signals"
        elif any(w in text_lower for w in ["clearly","well-established","settled","well-defined","universally"]):
            fields["confidence"] = "HIGH"
            fields["confidence_reason"] = "Answer contains certainty signals"
        for pat in [r"under (\w+ \w+ law|\w+ law|EU law|US law)", r"(Art\. \d+|§\d+|UCC|BGB|Code Civil|Codice Civile)"]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                fields["jurisdiction"] = m.group(0)
                break

    return fields


# ── Synthesis ─────────────────────────────────────────────────

async def synthesize(client, query, results, selected_jurisdiction=""):
    valid = [r for r in results if r.ok and r.text]

    empty_return = {
        "consensusScore": 0, "consensusAnswer": "No response available.",
        "follow_up1": "What are the legal requirements?",
        "follow_up2": "Are there any exceptions?",
        "follow_up3": "What remedies are available?",
        "divergences": "", "modelAlignments": {},
        "jurisdictions": [], "sources": [], "confidenceFlags": [],
        "reliabilityWarning": "No models responded.",
        "disagreementExplanation": ""
    }
    if not valid:
        return empty_return

    parsed = [parse_structured(r.text) for r in valid]
    jurisdictions = list(dict.fromkeys([p["jurisdiction"] for p in parsed if p["jurisdiction"] and len(p["jurisdiction"]) > 3]))
    raw_sources = [p["sources"] for p in parsed if p["sources"] and len(p["sources"]) > 5]
    sources = list(dict.fromkeys([s for s in raw_sources if clean_text(s) and len(clean_text(s)) > 5]))
    confidence_levels = [p["confidence"] for p in parsed]
    low_count = confidence_levels.count("LOW")
    high_count = confidence_levels.count("HIGH")

    answers_block = "\n".join([f"- {valid[i].name} [{parsed[i]['confidence']}]: {parsed[i]['answer']}" for i in range(len(valid))])
    sources_block = "\n".join([f"- {valid[i].name}: {parsed[i]['sources'] or 'No specific source'}" for i in range(len(valid))])
    juris_block   = "\n".join([f"- {valid[i].name}: {parsed[i]['jurisdiction'] or 'Not specified'}" for i in range(len(valid))])

    prompt = f"""You are a senior legal verification engine. Analyze these AI responses to the same legal question.

QUESTION: {query}

MODEL RESPONSES:
{answers_block}

JURISDICTIONS APPLIED:
{juris_block}

SOURCES CITED:
{sources_block}

Output exactly 8 lines, no other text:
VERDICT: [2 precise sentences synthesizing consensus. Cite the key legal standard or doctrine. Same language as question.]
SCORE: [integer 30-95. Agreement 40% + confidence levels 30% + source consistency 30%. Be honest.]
DIVERGENCES: [One sentence on what models disagree about, or "None detected."]
DISAGREEMENT: [If no disagreement: "All models aligned." Otherwise explain WHY they disagree — jurisdiction? standard? ambiguous statute? conflicting precedents?]
ALIGN_JSON: {{"ModelName": "agree|partial|disagree"}}
Q1: [follow-up question max 8 words, same language]
Q2: [follow-up question max 8 words, same language]
Q3: [follow-up question max 8 words, same language]"""

    verdict = parsed[0]["answer"][:300]
    score = 70
    divergences = ""
    disagreement = "All models aligned."
    alignments = {r.name: "agree" for r in valid}
    q1, q2, q3 = "What jurisdiction governs this?", "Are there exceptions to this rule?", "What remedies are available?"

    if ANTHROPIC_KEY:
        try:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json={"model": "claude-opus-4-5", "max_tokens": 700,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=35
            )
            text = r.json()["content"][0]["text"].strip()
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("VERDICT:"):
                    v = line[8:].strip(); verdict = v if v else verdict
                elif line.startswith("SCORE:"):
                    try: score = min(95, max(30, int(re.search(r"\d+", line[6:]).group())))
                    except: pass
                elif line.startswith("DIVERGENCES:"):
                    d = line[12:].strip()
                    if d.lower() not in ("none","none detected","none detected."): divergences = d
                elif line.startswith("DISAGREEMENT:"):
                    da = line[13:].strip()
                    disagreement = da if da else "All models aligned."
                elif line.startswith("ALIGN_JSON:"):
                    try: alignments = json.loads(line[11:].strip())
                    except: pass
                elif line.startswith("Q1:"): q = line[3:].strip(); q1 = q if q else q1
                elif line.startswith("Q2:"): q = line[3:].strip(); q2 = q if q else q2
                elif line.startswith("Q3:"): q = line[3:].strip(); q3 = q if q else q3
        except Exception:
            pass

    if low_count >= 2: score = min(score, 52)
    elif low_count == 1: score = min(score, 68)
    elif high_count == len(valid) and not divergences: score = max(score, 80)

    warning = ""
    if low_count >= 2:
        warning = f"⚠ {low_count} of {len(valid)} models flagged LOW confidence. Legally uncertain area — consult a qualified attorney."
    elif low_count == 1:
        warning = "⚠ One model flagged LOW confidence. Verify this applies to your specific circumstances."
    elif not selected_jurisdiction and len(set([j for j in [p['jurisdiction'] for p in parsed] if j])) > 2:
        warning = "⚠ Models applied different jurisdictions. Select a jurisdiction above for a more precise answer."
    elif not sources:
        warning = "⚠ No specific statutes cited. Based on general legal principles — verify with primary sources."

    return {
        "consensusScore": score,
        "consensusAnswer": verdict,
        "follow_up1": q1, "follow_up2": q2, "follow_up3": q3,
        "divergences": divergences,
        "modelAlignments": alignments,
        "jurisdictions": jurisdictions,
        "sources": sources,
        "confidenceFlags": [{"model": valid[i].name, "level": parsed[i]["confidence"], "reason": parsed[i]["confidence_reason"]} for i in range(len(valid))],
        "reliabilityWarning": warning,
        "disagreementExplanation": disagreement
    }


# ── Document extraction ───────────────────────────────────────

def extract_text(b64, ext):
    try:
        data = base64.b64decode(b64)
        if ext == "pdf" and HAS_PDF:
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            return " ".join([p.extract_text() or "" for p in reader.pages])[:4000]
        elif ext in ("docx","doc") and HAS_DOCX:
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
    if GROQ_KEY:       active.extend(["groq","mixtral_groq","deepseek"])
    if PERPLEXITY_KEY: active.append("perplexity")
    return {"status": "ok", "service": "Arguo API", "active_models": active}


@app.post("/analyze", response_model=ConsensusResponse)
async def analyze(req: QueryRequest):
    if not verify_secret(req.secret):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    async with httpx.AsyncClient() as client:
        jh = req.jurisdiction
        results = await asyncio.gather(
            call_claude(client, req.query, jh),
            call_gpt(client, req.query, jh),
            call_gemini(client, req.query, jh),
            call_mistral(client, req.query, jh),
            call_groq(client, req.query, jh),
            call_gpt_oss(client, req.query, jh),
            call_qwen(client, req.query, jh),
        )
        results = list(results)
        if not any(r.ok for r in results):
            raise HTTPException(status_code=503, detail="No model available")
        syn = await synthesize(client, req.query, results, req.jurisdiction)
    return ConsensusResponse(
        results=results,
        consensus_score=syn["consensusScore"],
        consensus_answer=syn["consensusAnswer"],
        follow_up1=syn["follow_up1"], follow_up2=syn["follow_up2"], follow_up3=syn["follow_up3"],
        divergences=syn["divergences"],
        model_alignments=syn["modelAlignments"],
        jurisdictions=syn["jurisdictions"],
        sources=syn["sources"],
        confidence_flags=syn["confidenceFlags"],
        reliability_warning=syn["reliabilityWarning"],
        disagreement_explanation=syn["disagreementExplanation"],
    )


@app.post("/analyze-document", response_model=ConsensusResponse)
async def analyze_document(req: DocumentRequest):
    if not verify_secret(req.secret):
        raise HTTPException(status_code=401, detail="Unauthorized")
    doc_text = req.document_text
    if req.document_base64 and req.document_ext:
        extracted = extract_text(req.document_base64, req.document_ext)
        if extracted: doc_text = extracted
    if not doc_text.strip():
        raise HTTPException(status_code=400, detail="Empty document")
    query = req.query.strip() or "Analyze this legal document. Summarize key clauses, obligations, risks, and notable terms."
    full_query = f"{query}\n\nDocument content:\n{doc_text[:3000]}"
    async with httpx.AsyncClient() as client:
        jh = req.jurisdiction
        results = await asyncio.gather(
            call_claude(client, full_query, jh),
            call_gemini(client, full_query, jh),
            call_mistral(client, full_query, jh),
            call_groq(client, full_query, jh),
            call_gpt_oss(client, full_query, jh),
            call_qwen(client, full_query, jh),
        )
        results = list(results)
        if not any(r.ok for r in results):
            raise HTTPException(status_code=503, detail="No model available")
        syn = await synthesize(client, query, results, req.jurisdiction)
    return ConsensusResponse(
        results=results,
        consensus_score=syn["consensusScore"],
        consensus_answer=syn["consensusAnswer"],
        follow_up1=syn["follow_up1"], follow_up2=syn["follow_up2"], follow_up3=syn["follow_up3"],
        divergences=syn["divergences"],
        model_alignments=syn["modelAlignments"],
        jurisdictions=syn["jurisdictions"],
        sources=syn["sources"],
        confidence_flags=syn["confidenceFlags"],
        reliability_warning=syn["reliabilityWarning"],
        disagreement_explanation=syn["disagreementExplanation"],
    )


@app.get("/health")
def health():
    return {"status": "healthy"}
