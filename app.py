from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # načte .env lokálně; na Renderu použijeme env vars v UI

MODEL = os.getenv("MODEL", "gpt-4o-mini")
VECTOR_STORE_ID = os.environ["VECTOR_STORE_ID"]

client = OpenAI()
app = FastAPI()

# CORS – pro jistotu povolíme všechny originy (Render stejně přijde z jedné domény)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# >>> NOVÉ: homepage vrátí index.html ze stejné složky
@app.get("/")
def home():
    return FileResponse("index.html")

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

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": body.question}
        ],
        temperature=0.01,
        max_output_tokens=250,
        tools=[{
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID],
            "ranking_options": {"score_threshold": 0.35}
        }],
    )
    return {"answer": resp.output_text}
