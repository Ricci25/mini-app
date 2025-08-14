from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import os
import logging

# lokálně čte .env; na Renderu se použijí env vars z UI
load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = os.getenv("MODEL", "gpt-4o-mini")
VECTOR_STORE_ID = os.environ["VECTOR_STORE_ID"]

OPENAI_URL = "https://api.openai.com/v1/responses"
TIMEOUT = 25  # s

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

class AskBody(BaseModel):
    question: str

@app.get("/")
def index():
    return FileResponse("index.html")

@app.get("/ping")
def ping():
    return {"ok": True}

@app.post("/ask")
def ask(body: AskBody):
    system = (
        "Odpovídej výhradně z poskytnutých úryvků (File Search). "
        "Nepoužívej žádné externí informace. "
        "Pokud odpověď není ve zdrojích, napiš: 'Nenašel jsem to ve zdrojích.' "
        "Buď stručný: max 5 vět (≈200 slov). Uváděj citace ve formátu [soubor: strana/sekce], pokud jsou dostupné."
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # >>> klíčová změna: starší schéma Responses API – vector_store_ids přímo v tools[0]
    payload = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": body.question}
        ],
        "temperature": 0.01,
        "max_output_tokens": 200,
        "tools": [
            {
                "type": "file_search",
                "vector_store_ids": [VECTOR_STORE_ID],
                "ranking_options": {"score_threshold": 0.35},
                "max_num_results": 8
            }
        ],
        "tool_choice": "auto"
    }

    try:
        logging.info(f"DOTAZ: {body.question!r}")
        r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=TIMEOUT)
        if not r.ok:
            logging.error("HTTP %s: %s", r.status_code, r.text[:1000])
            r.raise_for_status()
        data = r.json()

        # 1) jednoduchá cesta, pokud je přítomná
        answer = data.get("output_text")

        # 2) fallback – rekurzivně posbírat textové bloky
        def pick_text(obj):
            out = []
            if isinstance(obj, dict):
                if obj.get("type") in ("output_text", "text") and isinstance(obj.get("text"), str):
                    out.append(obj["text"])
                for v in obj.values():
                    out.extend(pick_text(v))
            elif isinstance(obj, list):
                for it in obj:
                    out.extend(pick_text(it))
            return out

        if not answer:
            pieces = pick_text(data)
            if pieces:
                answer = "\n".join(pieces).strip()

        if not answer:
            answer = "(prázdná odpověď)"

        logging.info("ODPOVĚĎ OK")
        return {"answer": answer}

    except requests.HTTPError:
        try:
            err = r.json()
        except Exception:
            err = {"error": r.text}
        logging.exception("HTTP ERROR při volání OpenAI")
        return {"answer": f"Chyba API: {err}"}
    except Exception as e:
        logging.exception("CHYBA SERVERU")
        return {"answer": f"Chyba serveru: {e}"}
