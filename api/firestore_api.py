from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import datetime
import traceback, logging

import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK with service account credentials
cred = credentials.Certificate("production-build-v1-firebase-adminsdk-fbsvc-9f9ca003be.json")
# Avoid duplicate app initialization during hot-reload
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)

# Get Firestore client instance
db = firestore.Client(
    project="production-build-v1",
    database="steraflow-firestore",
)

# Create FastAPI router with /api/v1 prefix
router = APIRouter(prefix="/api/v1")

# ---------- Pydantic Data Models ----------

# Model for storing metadata about uploaded files
class FileMetadata(BaseModel):
    file_id: Optional[str] = None        # Unique identifier for the file
    filename: str                        # Original filename
    supplier: Optional[str] = None       # Name of supplier who provided file
    file_format: Optional[str] = None    # File extension/format (pdf, xlsx etc)
    source: Optional[str] = None         # How file was received (email, chat etc)
    uploaded_at: Optional[datetime.datetime] = None  # Upload timestamp
    extra: Optional[Dict[str, Any]] = None # Additional metadata fields

# Model for chat messages between users and AI agent
class ChatMessage(BaseModel):
    user_id: str                         # ID of user in conversation
    role: str                            # Message sender (user/agent)
    content: str                         # Message text content
    timestamp: Optional[datetime.datetime] = None  # Message timestamp
    thread_id: Optional[str] = None      # Group messages in conversation threads

# Model for storing knowledge chunks for AI context
class ContextChunk(BaseModel):
    tag: Optional[str] = None            # Category of information
    product: Optional[str] = None        # Related product identifier
    text: str                           # Actual content/knowledge
    source_doc: Optional[str] = None     # Source document reference
    created_at: Optional[datetime.datetime] = None  # When chunk was created
    metadata: Optional[Dict[str, Any]] = None  # Additional context info

# ---------- File Metadata API Endpoints ----------

@router.post("/files/metadata/save")
async def save_file_metadata(payload: FileMetadata):
    """
    Store metadata for uploaded files in Firestore.
    Auto-generates upload timestamp if not provided.
    """
    try:
        if payload.uploaded_at is None:
            payload.uploaded_at = datetime.datetime.utcnow()
        doc_ref = db.collection("files_metadata").document(payload.file_id or None)
        doc_ref.set(payload.dict(exclude_none=True))
        return {"status": "saved", "id": doc_ref.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/metadata/get")
async def get_file_metadata(file_id: Optional[str] = None,
                            supplier: Optional[str] = None,
                            file_format: Optional[str] = None,
                            source: Optional[str] = None):
    """
    Retrieve file metadata by ID or filter criteria.
    Returns single doc if file_id provided, otherwise returns filtered list.
    """
    try:
        col = db.collection("files_metadata")
        if file_id:
            doc = col.document(file_id).get()
            if doc.exists:
                return doc.to_dict() | {"id": doc.id}
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=404, detail="File not found")
        # Build query with provided filters
        query = col
        if supplier:
            query = query.where("supplier", "==", supplier)
        if file_format:
            query = query.where("file_format", "==", file_format)
        if source:
            query = query.where("source", "==", source)
        docs = query.stream()
        return [d.to_dict() | {"id": d.id} for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Chat History API Endpoints ----------

@router.post("/chat/history/save")
async def save_chat_history(payload: ChatMessage):
    """
    Save a new chat message to user's conversation history.
    Messages are stored in subcollections per user.
    """
    try:
        if payload.timestamp is None:
            payload.timestamp = datetime.datetime.utcnow()
        col = db.collection("chat_history").document(payload.user_id).collection("messages")
        col.document().set(payload.dict(exclude_none=True))
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/history/get")
async def get_chat_history(user_id: str, limit: int = 50):
    """
    Retrieve recent chat messages for a user.
    Returns messages in reverse chronological order.
    """
    try:
        col = db.collection("chat_history").document(user_id).collection("messages")
        docs = (
            col.order_by("timestamp", direction=firestore.Query.DESCENDING)
               .limit(limit)
               .stream()
        )
        return [d.to_dict() for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- AI Agent Context API Endpoints ----------

@router.post("/agent/context/save")
async def save_agent_context(payload: ContextChunk):
    """
    Store knowledge chunks for AI context retrieval.
    Used for Retrieval-Augmented Generation (RAG).
    """
    try:
        if payload.created_at is None:
            payload.created_at = datetime.datetime.utcnow()
        col = db.collection("agent_context")
        col.document().set(payload.dict(exclude_none=True))
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agent/context/get")
async def get_agent_context(tag: Optional[str] = None,
                            product: Optional[str] = None,
                            limit: int = 100):
    """
    Retrieve AI context chunks filtered by tag/product.
    Returns chunks in reverse chronological order.
    """
    try:
        col = db.collection("agent_context")
        query = col
        if tag:
            query = query.where("tag", "==", tag)
        if product:
            query = query.where("product", "==", product)
        docs = (
            query.order_by("created_at", direction=firestore.Query.DESCENDING)
                 .limit(limit)
                 .stream()
        )
        return [d.to_dict() | {"id": d.id} for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Health Check Endpoint ----------

@router.get("/firestore/ping")
async def ping_firestore():
    """
    Simple health check endpoint to verify Firestore connectivity.
    Returns 500 error if connection fails.
    """
    try:
        _ = db.collection("healthcheck").document("ping").get()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
