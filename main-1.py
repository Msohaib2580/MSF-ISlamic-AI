import os, json
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import anthropic
from sentence_transformers import SentenceTransformer

load_dotenv()
client = anthropic.Anthropic()
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
app = FastAPI()

# ---- ماخذ اور search ----
_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
_docs = json.loads((Path(__file__).parent / "sources.json").read_text(encoding="utf-8"))
_vecs = _model.encode([d["text"] for d in _docs], normalize_embeddings=True)

def search(query, k=4, min_score=0.25):
    q = _model.encode([query], normalize_embeddings=True)[0]
    scores = _vecs @ q
    top = np.argsort(-scores)[:k]
    return [_docs[i] for i in top if scores[i] >= min_score]

# ---- prompt ----
SYSTEM = """آپ ایک اسلامی معلوماتی معاون ہیں۔ اصول:
1. صرف نیچے دیے گئے "ماخذ" سے جواب دیں اور ہر بات کے ساتھ حوالہ (ref) لکھیں۔
2. ماخذ میں جواب نہ ملے تو صاف کہیں: "میرے پاس موجود ماخذ میں یہ نہیں ملا۔" اندازہ نہ لگائیں، حوالہ نہ گھڑیں۔
3. اختلافی مسائل میں مختلف آراء مختصر بیان کریں، اپنی طرف سے فتویٰ نہ دیں۔
4. جواب مختصر، مؤدب اور سوال کی زبان میں ہو۔"""

def build_msg(question, sources):
    ctx = "\n\n".join(f"[{s['ref']}]\n{s['arabic']}\n{s['text']}" for s in sources) or "(کوئی ماخذ نہیں ملا)"
    return f"ماخذ:\n{ctx}\n\nسوال: {question}"

# ---- safety ----
HINTS = ["فتویٰ", "فتوی", "جائز ہے", "حرام ہے", "طلاق", "وراثت", "fatwa", "halal", "haram", "talaq"]
DISCLAIMER = "\n\n⚠️ یہ عمومی معلومات ہیں، فتویٰ نہیں۔ ذاتی صورتِ حال کے لیے کسی مستند عالم سے رجوع کریں۔"

# ---- API ----
class Msg(BaseModel):
    role: str
    content: str

class ChatIn(BaseModel):
    question: str
    history: list[Msg] = []

@app.post("/chat")
def chat(body: ChatIn):
    sources = search(body.question)
    messages = [m.model_dump() for m in body.history[-6:]]
    messages.append({"role": "user", "content": build_msg(body.question, sources)})
    r = client.messages.create(model=MODEL, max_tokens=800, system=SYSTEM, messages=messages)
    answer = r.content[0].text
    if any(h in body.question.lower() for h in HINTS):
        answer += DISCLAIMER
    return {"answer": answer, "references": [s["ref"] for s in sources]}
