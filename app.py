from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import requests, os, logging

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = os.getenv("MODEL", "gpt-4o-mini")
VECTOR_STORE_ID = os.environ["VECTOR_STORE_ID"]

# HTTP nastavení
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

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/ping")
def ping():
    return {"ok": True}

class AskBody(BaseModel):
    question: str

@app.post("/ask")
def ask(body: AskBody):
    system = (
        "Odpovídej výhradně z poskytnutých úryvků (File Search). "
        "Nepoužívej žádné externí informace. "
        "Pokud odpověď není ve zdrojích, napiš: 'Nenašel jsem to ve zdrojích.' "
        "Buď stručný: max 5 vět (≈200 slov). Přidej citace ve formátu [soubor: strana/sekce], pokud jsou dostupné."
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": body.question}
        ],
        "temperature": 0.01,
        "max_output_tokens": 200,
        "tools": [{
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID],
            "ranking_options": {"score_threshold": 0.35}
        }],
    }

    try:
        logging.info(f"DOTAZ: {body.question!r}")
        r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()

        # pohodlné pole "output_text" by mělo být v kořeni; fallback parsování:
        answer = data.get("output_text")
        if not answer:
            # fallback: najdi text v output -> content -> text
            out = data.get("output", [])
            if out and "content" in out[0]:
                parts = out[0]["content"]
                # najdi první blok s klíčem 'text'
                for p in parts:
                    if isinstance(p, dict) and p.get("type") in ("output_text", "text"):
                        answer = p.get("text")
                        break
        if not answer:
            answer = "(prázdná odpověď)"

        logging.info("ODPOVĚĎ OK")
        return {"answer": answer}

    except requests.HTTPError as e:
        logging.exception("HTTP ERROR")
        try:
            err = r.json()
        except Exception:
            err = {"error": str(e)}
        return {"answer": f"Chyba API: {err}"}
    except Exception as e:
        logging.exception("CHYBA PŘI VOLÁNÍ OPENAI")
        return {"answer": f"Chyba serveru: {e}"}
