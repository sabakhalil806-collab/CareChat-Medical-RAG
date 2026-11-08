import os, re, json
import numpy as np
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

BASE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(BASE, "config.json"), encoding="utf-8"))
CHUNKS = json.load(open(os.path.join(BASE, "chunks.json"), encoding="utf-8"))
index = faiss.read_index(os.path.join(BASE, "faiss.index"))

REFUSAL = "I don't know based on the provided documents."
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DISCLAIMER = cfg["disclaimer"]
THRESHOLD = float(os.environ.get("RERANK_THRESHOLD", cfg["threshold"]))


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


bm25 = BM25Okapi([tokenize(c["embed_text"]) for c in CHUNKS])
embedder = SentenceTransformer(cfg["embed_model"], device="cpu")
reranker = CrossEncoder(cfg["rerank_model"], device="cpu")

BACKEND = os.environ.get("LLM_BACKEND", cfg["backend"])
LLM_MODEL = os.environ.get("LLM_MODEL", cfg["llm_model"])

if BACKEND == "hf_inference":
    from huggingface_hub import InferenceClient
    _client = InferenceClient(model=LLM_MODEL, token=os.environ.get("HF_TOKEN"))

    def chat(messages, max_new_tokens=300):
        try:
            out = _client.chat_completion(messages=messages, max_tokens=max_new_tokens, temperature=0.01)
            return out.choices[0].message.content.strip()
        except Exception as e:
            return "LLM_ERROR: " + str(e)[:200]
else:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    _model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
    _model.eval()

    def chat(messages, max_new_tokens=300):
        text = _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _tok(text, return_tensors="pt")
        with torch.no_grad():
            out = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  temperature=None, top_p=None, top_k=None,
                                  pad_token_id=_tok.eos_token_id)
        return _tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def retrieve(query, k_final=None, k_cand=None):
    k_final = k_final or cfg["top_k_final"]
    k_cand = k_cand or cfg["top_k_candidates"]
    q = embedder.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True).astype("float32")
    sims, ids = index.search(q, k_cand)
    dense = [int(i) for i in ids[0] if i >= 0]
    bm_ids = np.argsort(-bm25.get_scores(tokenize(query)))[:k_cand]
    rrf = {}
    for rank, i in enumerate(dense):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + rank + 1)
    for rank, i in enumerate(bm_ids):
        rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (60 + rank + 1)
    fused = sorted(rrf, key=rrf.get, reverse=True)[:k_cand]
    scores = reranker.predict([(query, CHUNKS[i]["embed_text"]) for i in fused])
    order = np.argsort(-scores)[:k_final]
    return [{"idx": fused[o], "score": float(scores[o])} for o in order]


EMERGENCY_PATTERN = re.compile(
    r"chest pain|can'?t breathe|cannot breathe|difficulty breathing|shortness of breath|"
    r"stroke|face droop|sudden weakness|unconscious|seizure|severe bleeding|suicid|overdose", re.I)
EMERGENCY_NOTE = ("**If this is happening to you or someone else right now, contact your local emergency "
                  "number or go to the nearest emergency department immediately.**")

SYSTEM_PROMPT = (
    "You are a careful medical information assistant.\n"
    "Answer the user's question using ONLY the numbered context passages provided.\n"
    "Rules:\n"
    "1. Do not use outside knowledge and do not add facts that are not in the context.\n"
    "2. Be concise: a short paragraph or a few bullet points.\n"
    "3. Mention passage numbers like [1] or [2] after the facts they support.\n"
    "4. If the context does not contain the answer, reply exactly: " + REFUSAL + "\n"
    "5. Never diagnose the user or prescribe doses; suggest talking to a doctor when relevant."
)


def rewrite_question(question, history):
    if not history:
        return question
    convo = "\n".join("User: " + u + "\nAssistant: " + a[:300] for u, a in history[-3:])
    msgs = [
        {"role": "system", "content": "Rewrite the user's last question as a single standalone question that makes "
                                      "sense without the conversation. Replace pronouns like 'it' or 'its' with the "
                                      "topic. If it is already standalone, return it unchanged. Output only the question."},
        {"role": "user", "content": "Conversation:\n" + convo + "\n\nLast question: " + question + "\n\nStandalone question:"},
    ]
    out = chat(msgs, max_new_tokens=60)
    if out.startswith("LLM_ERROR"):
        return question
    return out.split("\n")[0].strip() or question


def ask(question, history):
    standalone = rewrite_question(question, history)
    results = retrieve(standalone)
    best = results[0]["score"] if results else float("-inf")
    emergency = EMERGENCY_PATTERN.search(question) is not None

    if best < THRESHOLD:
        return REFUSAL + ("\n\n" + EMERGENCY_NOTE if emergency else ""), []

    context = "\n\n".join(
        "[" + str(n) + "] (" + CHUNKS[r["idx"]]["org"] + ": " + CHUNKS[r["idx"]]["title"] + " - "
        + CHUNKS[r["idx"]]["heading"] + ")\n" + CHUNKS[r["idx"]]["text"]
        for n, r in enumerate(results, start=1))
    answer = chat([{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": "Context:\n" + context + "\n\nQuestion: " + standalone}])

    if answer.startswith("LLM_ERROR"):
        return "The language model service is temporarily unavailable. Please try again in a moment.", []
    if REFUSAL.lower() in answer.lower() and len(answer) < len(REFUSAL) + 20:
        return REFUSAL + ("\n\n" + EMERGENCY_NOTE if emergency else ""), []

    sources, seen = [], set()
    for n, r in enumerate(results, start=1):
        c = CHUNKS[r["idx"]]
        key = (c["url"], c["heading"])
        if key not in seen:
            seen.add(key)
            sources.append((n, c))
    if emergency:
        answer += "\n\n" + EMERGENCY_NOTE
    return answer, sources


def to_pairs(history):
    pairs = []
    if history and isinstance(history[0], dict):
        user = None
        for m in history:
            content = m["content"] if isinstance(m["content"], str) else str(m["content"])
            if m["role"] == "user":
                user = content
            elif m["role"] == "assistant" and user is not None:
                pairs.append((user, content))
                user = None
    else:
        pairs = [(u, b) for u, b in (history or []) if u and b]
    return pairs


def respond(message, history):
    answer, sources = ask(message, to_pairs(history))
    text = answer
    if sources:
        lines = []
        for n, c in sources:
            label = "[" + str(n) + "] " + c["org"] + ": " + c["title"] + " - " + c["heading"]
            lines.append("- [" + label + "](" + c["url"] + ")" if c["url"].startswith("http") else "- " + label)
        text += "\n\n**Sources**\n" + "\n".join(lines)
    return text + "\n\n_" + DISCLAIMER + "_"


demo = gr.ChatInterface(
    fn=respond,
    title="Medical RAG Chatbot",
    description="Answers only from the public health pages in its knowledge base (MedlinePlus, WHO, NHS, CDC). " + DISCLAIMER,
    examples=["What are the symptoms of diabetes?", "How is high blood pressure treated?",
              "How is malaria transmitted?", "What are the warning signs of a stroke?"],
)

if __name__ == "__main__":
    demo.launch()
