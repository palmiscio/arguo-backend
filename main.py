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


ARGUO_SECRET = os.getenv("ARGUO_API_SECRET", "")

def verify_secret(request_secret: str) -> bool:
    if not ARGUO_SECRET:
        return True  # If no secret set, allow all (dev mode)
    return request_secret == ARGUO_SECRET

ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_KEY     = os.getenv("OPENAI_API_KEY", "")
GEMINI_KEY     = os.getenv("GEMINI_API_KEY", "")
MISTRAL_KEY    = os.getenv("MISTRAL_API_KEY", "")
GROQ_KEY       = os.getenv("GROQ_API_KEY", "")
PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY", "")

SYSTEM_PROMPT = """You are a senior legal counsel with deep expertise in international, US, EU, and civil law systems.
Answer legal questions with precision. You MUST follow this exact structure in your response:

ANSWER: [1-2 sentences direct answer, same language as question]
JURISDICTION: [specify the exact jurisdiction(s) this answer applies to, e.g. "US Federal Law", "EU Law (GDPR)", "Italian Civil Code", "Common Law jurisdictions"]
SOURCES: [cite the specific law, statute, article, or legal principle — e.g. "UCC §2-609", "Art. 1453 Codice Civile", "EU Directive 2019/770"]
CONFIDENCE: [HIGH / MEDIUM / LOW] — [one sentence explaining why, e.g. "HIGH: well-settled law with consistent case precedent" or "LOW: jurisdiction unclear, conflicting interpretations exist"]

Never omit any of these four fields. If you cannot identify a specific source, write SOURCES: No specific statute — general common law principle applies."""


class QueryRequest(BaseModel):
    query: str
    secret: str = ""


class DocumentRequest(BaseModel):
    query: str
    secret: str = ""
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
    jurisdictions: list = []
    sources: list = []
    confidence_flags: list = []
    reliability_warning: str = ""
    disagreement_explanation: str = ""


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
                  "generationConfig": {"maxOutputTokens": 800}},
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
            json={"model": "qwen/qwen3-32b", "max_tokens": 800,
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
            "modelAlignments": {},
            "jurisdictions": [],
            "sources": [],
            "confidenceFlags": [],
            "reliabilityWarning": "No models responded.",
            "disagreementExplanation": ""
        }

    # ── Robust structured parser with fallback ──
    def parse_structured(text):
        """Parse structured response. Falls back gracefully if format not followed."""
        fields = {
            "answer": "",
            "jurisdiction": "",
            "sources": "",
            "confidence": "MEDIUM",
            "confidence_reason": "Format not structured — auto-assessed"
        }

        lines = text.strip().split("\n")
        found_any_field = False

        for line in lines:
            line = line.strip()
            if line.startswith("ANSWER:"):
                fields["answer"] = line[7:].strip()
                found_any_field = True
            elif line.startswith("JURISDICTION:"):
                fields["jurisdiction"] = line[13:].strip()
                found_any_field = True
            elif line.startswith("SOURCES:"):
                src = line[8:].strip()
                if "no specific" not in src.lower() and "general" not in src.lower():
                    fields["sources"] = src
                found_any_field = True
            elif line.startswith("CONFIDENCE:"):
                rest = line[11:].strip()
                for sep in ["—", "-", ":"]:
                    if sep in rest:
                        parts = rest.split(sep, 1)
                        level = parts[0].strip().upper()
                        if level in ("HIGH", "MEDIUM", "LOW"):
                            fields["confidence"] = level
                            fields["confidence_reason"] = parts[1].strip()
                        break
                else:
                    level = rest.strip().upper()
                    if level in ("HIGH", "MEDIUM", "LOW"):
                        fields["confidence"] = level
                found_any_field = True

        # Fallback: model did not follow structured format
        if not found_any_field or not fields["answer"]:
            fields["answer"] = text.strip()[:400]
            # Auto-detect confidence from text signals
            text_lower = text.lower()
            if any(w in text_lower for w in ["unclear", "uncertain", "may vary", "depends", "complex", "disputed"]):
                fields["confidence"] = "LOW"
                fields["confidence_reason"] = "Auto-detected: answer contains uncertainty signals"
            elif any(w in text_lower for w in ["clearly", "established", "settled", "well-defined", "generally"]):
                fields["confidence"] = "HIGH"
                fields["confidence_reason"] = "Auto-detected: answer contains certainty signals"
            else:
                fields["confidence"] = "MEDIUM"
                fields["confidence_reason"] = "Auto-assessed: no structured confidence provided"

            # Try to extract jurisdiction from text
            import re as _re
            juris_patterns = [
                r"under (\w+ \w+ law|\w+ law|EU law|US law|English law)",
                r"in (\w+) jurisdiction",
                r"(Article \d+|§\d+|UCC|BGB|Code Civil|Codice Civile)"
            ]
            for pat in juris_patterns:
                m = _re.search(pat, text, _re.IGNORECASE)
                if m:
                    fields["jurisdiction"] = m.group(0)
                    break

        return fields

    parsed = [parse_structured(r.text) for r in valid]

    # ── Aggregate data ──
    jurisdictions = list(dict.fromkeys([p["jurisdiction"] for p in parsed if p["jurisdiction"]]))
    all_sources = [p["sources"] for p in parsed if p["sources"]]
    sources = list(dict.fromkeys(all_sources))
    confidence_levels = [p["confidence"] for p in parsed]
    low_count = confidence_levels.count("LOW")
    high_count = confidence_levels.count("HIGH")
    medium_count = confidence_levels.count("MEDIUM")

    # ── Build answers block for synthesis ──
    answers_block = "\n".join([
        f"- {valid[i].name} [{parsed[i]['confidence']}]: {parsed[i]['answer']}"
        for i in range(len(valid))
    ])
    sources_block = "\n".join([f"- {valid[i].name}: {parsed[i]['sources'] or 'No specific source cited'}" for i in range(len(valid))])
    juris_block = "\n".join([f"- {valid[i].name}: {parsed[i]['jurisdiction'] or 'Not specified'}" for i in range(len(valid))])

    # ── Main synthesis prompt ──
    prompt = f"""You are a senior legal verification engine. Analyze these AI responses to the same legal question.

QUESTION: {query}

MODEL RESPONSES:
{answers_block}

JURISDICTIONS APPLIED:
{juris_block}

SOURCES CITED:
{sources_block}

Produce exactly this output (7 lines, no other text):
VERDICT: [2 precise sentences synthesizing the consensus answer. Be specific — cite the key legal standard, doctrine or rule. Same language as the question.]
SCORE: [integer 30-95. Calculate: agreement between answers 40% + confidence levels 30% + source consistency 30%. Be honest — if models disagree significantly, score below 60.]
DIVERGENCES: [If models agree: "None detected." If they disagree: one sentence describing WHAT they disagree on.]
DISAGREEMENT: [If no disagreement: "All models aligned." If disagreement exists: explain WHY they disagree — different jurisdiction? different legal standard? ambiguous statute? conflicting precedents? Be specific and educational.]
ALIGN_JSON: {{"ModelName": "agree|partial|disagree", ...}} for each model
Q1: [follow-up question max 8 words, same language as question]
Q2: [follow-up question max 8 words, same language as question]
Q3: [follow-up question max 8 words, same language as question]"""

    # Defaults
    verdict = parsed[0]["answer"][:300] if parsed else "No consensus available."
    score = 70
    divergences = ""
    disagreement_explanation = "All models aligned."
    alignments = {r.name: "agree" for r in valid}
    q1 = "What jurisdiction governs this issue?"
    q2 = "Are there any exceptions to this rule?"
    q3 = "What remedies are available?"

    if ANTHROPIC_KEY:
        try:
            import re as _re, json as _json
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json={"model": "claude-opus-4-5", "max_tokens": 600,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=35
            )
            text = r.json()["content"][0]["text"].strip()

            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("VERDICT:"):
                    v = line[8:].strip()
                    if v: verdict = v
                elif line.startswith("SCORE:"):
                    try:
                        s = int(_re.search(r"\d+", line[6:]).group())
                        score = min(95, max(30, s))
                    except: pass
                elif line.startswith("DIVERGENCES:"):
                    d = line[12:].strip()
                    if d.lower() not in ("none", "none detected", "none detected."):
                        divergences = d
                elif line.startswith("DISAGREEMENT:"):
                    da = line[13:].strip()
                    if da.lower() not in ("all models aligned.", "all models aligned"):
                        disagreement_explanation = da
                    else:
                        disagreement_explanation = "All models aligned on this question."
                elif line.startswith("ALIGN_JSON:"):
                    try:
                        alignments = _json.loads(line[11:].strip())
                    except: pass
                elif line.startswith("Q1:"):
                    q = line[3:].strip()
                    if q: q1 = q
                elif line.startswith("Q2:"):
                    q = line[3:].strip()
                    if q: q2 = q
                elif line.startswith("Q3:"):
                    q = line[3:].strip()
                    if q: q3 = q
        except Exception:
            pass

    # ── Adjust score based on confidence ──
    if low_count >= 2:
        score = min(score, 52)
    elif low_count == 1:
        score = min(score, 68)
    elif high_count == len(valid) and not divergences:
        score = max(score, 80)

    # ── Reliability warning ──
    reliability_warning = ""
    if low_count >= 2:
        reliability_warning = f"⚠ {low_count} of {len(valid)} models flagged LOW confidence. This is a legally uncertain area — consult a qualified attorney before acting."
    elif low_count == 1:
        reliability_warning = "⚠ One model flagged LOW confidence. Verify the answer applies to your specific jurisdiction and circumstances."
    elif len(set([j for j in [p['jurisdiction'] for p in parsed] if j])) > 2:
        reliability_warning = f"⚠ Models applied different jurisdictions. Confirm which legal system governs your situation."
    elif not sources:
        reliability_warning = "⚠ No specific statutes cited by any model. This answer is based on general legal principles — verify with primary sources."

    return {
        "consensusScore": score,
        "consensusAnswer": verdict,
        "follow_up1": q1,
        "follow_up2": q2,
        "follow_up3": q3,
        "divergences": divergences,
        "modelAlignments": alignments,
        "jurisdictions": jurisdictions,
        "sources": sources,
        "confidenceFlags": [
            {"model": valid[i].name, "level": parsed[i]["confidence"], "reason": parsed[i]["confidence_reason"]}
            for i in range(len(valid))
        ],
        "reliabilityWarning": reliability_warning,
        "disagreementExplanation": disagreement_explanation
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
    if not verify_secret(req.secret):
        raise HTTPException(status_code=401, detail="Unauthorized")
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
    if not verify_secret(req.secret):
        raise HTTPException(status_code=401, detail="Unauthorized")
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
        jurisdictions=syn.get("jurisdictions", []),
        sources=syn.get("sources", []),
        confidence_flags=syn.get("confidenceFlags", []),
        reliability_warning=syn.get("reliabilityWarning", ""),
    )


@app.get("/health")
def health():
    return {"status": "healthy"}
