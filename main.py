"""AI Memory Layer service implementing endpoints defined in specs.md."""
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
import uuid
import json
import time
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
import chromadb

from config import CHROMA_PORT, SERVER_PORT  # ChromaDB server port

# -----------------------------------------------------------------------------
# Database setup
# -----------------------------------------------------------------------------
# Connect to a running ChromaDB server (started separately, see README).
client = chromadb.HttpClient(host="0.0.0.0", port=CHROMA_PORT)
collection = client.get_or_create_collection(name="memory1")

# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class MemoryCreate(BaseModel):
    content: Optional[str] = Field(None, description="Raw text to store")
    metadata: Optional[Dict[str, Any]] = None

class MemoryResponse(BaseModel):
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    createdAt: str

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=50)
    filter: Optional[Dict[str, Any]] = None

class RetrievedMemory(MemoryResponse):
    relevance: float

# -----------------------------------------------------------------------------
# FastAPI app & middleware
LOG_FILE = Path("reports.log")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AI Memory Layer API", version="1.0.0")

@app.middleware("http")
async def report_middleware(request, call_next):
    start = time.time()
    req_body: Union[str, Dict[str, Any]]
    try:
        req_body_bytes = await request.body()
        req_body = req_body_bytes.decode()[:5000]  # truncate
    except Exception:
        req_body = "<unreadable>"

    status = "unknown"
    resp_body_preview = None
    try:
        response = await call_next(request)
        status = "success"
        if hasattr(response, "body"):
            resp_body_preview = response.body.decode()[:5000]
    except Exception as exc:
        status = "error"
        response = app.exception_handler(type(exc))(request, exc)  # type: ignore
    duration_ms = int((time.time() - start) * 1000)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": request.method,
        "path": request.url.path,
        "status": status,
        "status_code": response.status_code if hasattr(response, "status_code") else 500,
        "duration_ms": duration_ms,
        "request": req_body,
        "response": resp_body_preview,
    }
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logging.error("Failed to write log entry: %s", e)

    return response

@app.post("/memory", status_code=201, response_model=MemoryResponse)
def add_memory(body: MemoryCreate):
    if not body.content:
        raise HTTPException(status_code=400, detail="'content' is required")
    memory_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    metadata = body.metadata.copy() if body.metadata else {}
    metadata["createdAt"] = created_at

    # Store in ChromaDB
    collection.add(ids=[memory_id], documents=[body.content], metadatas=[metadata])

    return {
        "id": memory_id,
        "content": body.content,
        "metadata": metadata,
        "createdAt": created_at,
    }

@app.delete("/memory/{memory_id}", status_code=204)
def remove_memory(memory_id: str):
    try:
        collection.delete(ids=[memory_id])
    except Exception:
        # Chroma raises ValueError if id not found
        raise HTTPException(status_code=404, detail="Memory not found")
    return Response(status_code=204)

@app.post("/memory/retrieve", response_model=List[RetrievedMemory])
def retrieve_memories(req: RetrieveRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="'query' is required")

    try:
        results = collection.query(
            query_texts=[req.query],
            n_results=req.top_k,
            where=req.filter or {},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    memories: List[RetrievedMemory] = []
    for i, mem_id in enumerate(ids):
        distance = distances[i] if i < len(distances) else 1.0
        relevance = max(0.0, 1.0 - distance)  # crude conversion
        memories.append(
            RetrievedMemory(
                id=mem_id,
                content=docs[i],
                metadata=metadatas[i],
                createdAt=metadatas[i].get("createdAt", ""),
                relevance=relevance,
            )
        )
    return memories

# -----------------------------------------------------------------------------
# Reports endpoint
# -----------------------------------------------------------------------------
class ReportEntry(BaseModel):
    timestamp: str
    method: str
    path: str
    status: str
    status_code: int
    duration_ms: int
    request: Union[str, Dict[str, Any]]
    response: Optional[str] = None

@app.get("/reports", response_model=List[ReportEntry])
def get_reports(limit: int = 100):
    """Return the most recent `limit` request reports."""
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    selected = lines[-limit:]
    return [json.loads(line) for line in selected]

# -----------------------------------------------------------------------------
# Local dev entry-point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=SERVER_PORT, reload=True)
