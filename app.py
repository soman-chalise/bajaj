import os
import requests
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from PyPDF2 import PdfReader
from docx import Document

load_dotenv()

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "hackrx-docs")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("GEMINI_API_KEY and PINECONE_API_KEY are required.")

# --- Init Gemini ---
genai.configure(api_key=GEMINI_API_KEY)

# --- Init Pinecone ---
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
index = pc.Index(PINECONE_INDEX)

# --- FastAPI ---
app = FastAPI(title="HackRx Retrieval API")

# --- Request model ---
class HackRxRequest(BaseModel):
    document_url: str
    questions: list[str]

# --- PDF/DOCX processing ---
def process_file(file_path):
    chunks = []
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                chunks.append(text)
    elif file_path.lower().endswith(".docx"):
        doc = Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                chunks.append(para.text)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    return chunks

# --- Embed text chunks ---
def embed_texts(texts):
    model = "models/embedding-001"
    embeddings = []
    for i, t in enumerate(texts):
        res = genai.embed_content(model=model, content=t)
        embeddings.append((f"id-{i}", res["embedding"], {"text": t}))
    return embeddings

# --- Query Pinecone ---
def query_pinecone(question, top_k=5):
    emb = genai.embed_content(model="models/embedding-001", content=question)["embedding"]
    res = index.query(vector=emb, top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in res["matches"]]

# --- Ask Gemini ---
def ask_gemini(context_chunks, question):
    prompt = f"""
You are an insurance policy QA assistant.

Rules:
1. Use ONLY the context provided to answer the question.
2. Give a **complete sentence** in formal style, as if from the policy wording.
3. Do NOT include the question in your answer.
4. If something is not covered, say exactly "Not covered" or the explicit exclusion wording.
5. If the answer is not found in the context, say "Not found".
6. Ensure numbers, dates, and limits are stated exactly as in the policy.

Example:
Context:
Policy: Cataract surgery has a waiting period of two (2) years.
Question: What is the waiting period for cataract surgery?
Answer: The policy has a specific waiting period of two (2) years for cataract surgery.

Context:
Policy: Maternity expenses are excluded from coverage.
Question: Does this policy cover maternity expenses?
Answer: Not covered

Now, use the following context to answer:
Context:
{''.join(context_chunks)}

Question: {question}
Answer:
"""
    model = genai.GenerativeModel("gemini-2.5-pro")
    resp = model.generate_content(prompt)
    return resp.text.strip().split("\n")[0]  # first line only

# --- HackRx Run endpoint ---
@app.post("/hackrx/run")
async def hackrx_run(payload: HackRxRequest):
    # 1. Download document
    resp = requests.get(payload.document_url)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download document from URL")

    suffix = ".pdf" if ".pdf" in payload.document_url.lower() else ".docx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    # 2. Process file into chunks
    chunks = process_file(tmp_path)

    # 3. Embed + upsert into Pinecone
    embeddings = embed_texts(chunks)
    index.upsert(vectors=embeddings)

    # 4. Answer questions (return array of strings only)
    answers = []
    for q in payload.questions:
        context = query_pinecone(q, top_k=5)
        answer = ask_gemini(context, q)
        answers.append(answer)

    return {"answers": answers}
