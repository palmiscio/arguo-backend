# LexVerify — Guida al Deploy

## Struttura del progetto

```
lexverify-backend/
├── main.py              ← Backend FastAPI (tutte le API AI)
├── requirements.txt     ← Dipendenze Python
├── .env.example         ← Template per le API key
├── .env                 ← Le tue key (NON committare su Git)
├── Procfile             ← Per deploy su Railway/Render
└── .gitignore

lexverify-frontend.html  ← Il frontend (apri nel browser)
```

---

## 1. Avvio in locale (test)

### Requisiti
- Python 3.11+
- pip

### Passi

```bash
# Entra nella cartella backend
cd lexverify-backend

# Installa le dipendenze
pip install -r requirements.txt

# Crea il file .env con le tue key
cp .env.example .env
# Apri .env con un editor e inserisci le key

# Avvia il server
uvicorn main:app --reload --port 8000
```

Apri `lexverify-frontend.html` nel browser.
Il campo "Backend URL" deve essere: `http://localhost:8000`
Clicca "Verifica →" per controllare la connessione.

---

## 2. Deploy su Railway (gratis per iniziare)

Railway è la soluzione più semplice per rendere il backend accessibile online.

### Passi

1. Crea un account su [railway.app](https://railway.app)
2. Installa Railway CLI:
   ```bash
   npm install -g @railway/cli
   railway login
   ```
3. Dalla cartella `lexverify-backend`:
   ```bash
   railway init
   railway up
   ```
4. Nel pannello Railway → **Variables**, aggiungi tutte le key:
   ```
   ANTHROPIC_API_KEY  = sk-ant-...
   OPENAI_API_KEY     = sk-...
   GEMINI_API_KEY     = AIza...
   MISTRAL_API_KEY    = ...
   GROQ_API_KEY       = gsk_...
   PERPLEXITY_API_KEY = pplx-...
   ```
5. Railway ti darà un URL tipo `https://lexverify-production.up.railway.app`
6. Inserisci quell'URL nel campo "Backend URL" del frontend

---

## 3. Deploy alternativo su Render (gratuito)

1. Crea account su [render.com](https://render.com)
2. "New Web Service" → collega il tuo repo GitHub
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Aggiungi le variabili d'ambiente nel pannello Render

---

## 4. Come funziona il sistema

```
Browser dell'utente
      ↓ POST /analyze { query: "..." }
  Tuo Backend (Railway/Render)
      ↓ chiamate in parallelo (asyncio)
  ┌─────────────────────────────────┐
  │ Claude  GPT  Gemini  Mistral    │
  │ Groq    Perplexity              │
  └─────────────────────────────────┘
      ↓ raccoglie tutte le risposte
      ↓ sintetizza con Claude
  Risposta consensuale + score
      ↓
  Browser dell'utente
```

Le API key restano **solo sul server** — gli utenti non le vedono mai.

---

## 5. Aggiungere autenticazione utenti (prossimo step)

Quando vuoi aprire LexVerify a utenti esterni, aggiungi Supabase:
- Registrazione/login utenti
- Conteggio query per piano tariffario
- Dashboard admin

Documentazione: [supabase.com/docs](https://supabase.com/docs)

---

## API Key — dove ottenerle

| Modello | Link | Costo |
|---|---|---|
| Claude | [console.anthropic.com](https://console.anthropic.com) | Pay-per-use |
| GPT-4o | [platform.openai.com](https://platform.openai.com/api-keys) | Pay-per-use |
| Gemini | [aistudio.google.com](https://aistudio.google.com/app/apikey) | Gratuito |
| Mistral | [console.mistral.ai](https://console.mistral.ai) | Piano gratuito |
| Groq | [console.groq.com](https://console.groq.com/keys) | Gratuito |
| Perplexity | [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api) | Pro $20/mese |
