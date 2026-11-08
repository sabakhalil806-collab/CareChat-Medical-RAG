"""CareChat: custom web UI on top of the existing app.py (same retrieval + LLM logic).

Put this file and index.html in the SAME folder as app.py, chunks.json, config.json,
faiss.index, requirements.txt. Then run:   python carechat.py
and open http://127.0.0.1:8000
"""
import os
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import app as core  # your existing app.py: loads the models, index and the ask() pipeline

BASE = os.path.dirname(os.path.abspath(__file__))

# topic -> (display name, words used in questions, emoji, tint colour)
TOPIC_INFO = {
    "diabetes":       ("Diabetes",        "diabetes",            "💧", "#e3eeff"),
    "hypertension":   ("Blood pressure",  "high blood pressure", "🩸", "#fde6ea"),
    "asthma":         ("Asthma",          "asthma",              "🌬️", "#ece6fb"),
    "heart disease":  ("Heart disease",   "heart disease",       "❤️", "#fde9e0"),
    "stroke":         ("Stroke",          "stroke",              "⚡", "#fdf1cf"),
    "obesity":        ("Obesity",         "obesity",             "⚖️", "#e0f3ec"),
    "cholesterol":    ("Cholesterol",     "high cholesterol",    "🧪", "#e3f1fb"),
    "copd":           ("COPD",            "COPD",                "🫁", "#e8eefc"),
    "kidney disease": ("Kidney disease",  "chronic kidney disease", "🫘", "#f3e8dc"),
    "anemia":         ("Anemia",          "anemia",              "🩹", "#fbe3ee"),
    "malaria":        ("Malaria",         "malaria",             "🦟", "#e7f3df"),
}

FOLLOW_UPS = [
    ("diagnos", "How is {t} diagnosed?"),
    ("treat", "How is {t} treated?"),
    ("symptom", "What are the symptoms of {t}?"),
    ("prevent", "How can {t} be prevented?"),
]

api = FastAPI(title="CareChat")


class ChatRequest(BaseModel):
    message: str
    history: List[List[str]] = []


def term_for(topic):
    return TOPIC_INFO.get(topic, (topic.title(), topic, "", ""))[1]


@api.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(BASE, "index.html"), encoding="utf-8") as f:
        return f.read()


@api.get("/api/info")
def info():
    present = {c["topic"] for c in core.CHUNKS}
    ordered = [t for t in TOPIC_INFO if t in present] + sorted(t for t in present if t not in TOPIC_INFO)
    topics = []
    for t in ordered:
        name, query, emoji, tint = TOPIC_INFO.get(t, (t.title(), t, "🩺", "#e6f3ef"))
        topics.append({"id": t, "name": name, "query": query, "emoji": emoji, "tint": tint})
    orgs = sorted({c["org"] for c in core.CHUNKS})
    return {"topics": topics, "orgs": orgs, "disclaimer": core.DISCLAIMER}


@api.post("/api/chat")
def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        return {"answer": "Please type a health question.", "sources": [], "followups": []}

    history = [(h[0], h[1]) for h in req.history if len(h) >= 2 and h[0] and h[1]]
    answer, sources = core.ask(message, history)

    src = [{"n": n, "org": c["org"], "title": c["title"], "heading": c["heading"],
            "url": c["url"], "topic": c["topic"]} for n, c in sources]

    followups = []
    if src:
        t = term_for(src[0]["topic"])
        low = message.lower()
        for key, tpl in FOLLOW_UPS:
            if key not in low:
                followups.append(tpl.format(t=t))
            if len(followups) == 2:
                break

    return {"answer": answer, "sources": src, "followups": followups}


if __name__ == "__main__":
    print("\nCareChat is starting. Open  http://127.0.0.1:8000  in your browser.\n")
    uvicorn.run(api, host="127.0.0.1", port=8000)
