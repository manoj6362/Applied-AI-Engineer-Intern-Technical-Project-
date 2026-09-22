"""FastAPI chat endpoint over the same retrieval path used by backend/rag.py.

usage:
    uvicorn backend.server:app --reload
"""
import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import DB_DSN
from backend.rag import answer_chat, load_catalog

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
eng = sa.create_engine(DB_DSN)
_catalog = None


def _get_catalog():
    global _catalog
    if _catalog is None:
        with eng.begin() as cx:
            _catalog = load_catalog(cx)
    return _catalog


class Q(BaseModel):
    question: str


@app.post("/api/chat")
def answer(q: Q) -> dict:
    """Answer a free-text question using the same retrieve/reason path as backend/rag.py.

    :param q: request body with a single "question" field
    :return: {"answer": str, "evidence": [{"table", "id"|game_id/person_id, "doc"}]}
    """
    print("Received question")
    catalog = _get_catalog()
    with eng.begin() as cx:
        return answer_chat(cx, q.question, catalog)
