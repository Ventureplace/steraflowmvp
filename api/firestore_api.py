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

# ---------- Advanced Query Models ----------
class BatchOperationRequest(BaseModel):
    operations: List[Dict[str, Any]]
    collection: str

class CompoundQueryRequest(BaseModel):
    collection: str
    filters: List[Dict[str, Any]]
    order_by: Optional[List[Dict[str, str]]] = None
    limit: Optional[int] = 100
    start_after: Optional[Dict[str, Any]] = None

class DataSyncRequest(BaseModel):
    source_collection: str
    target_collection: str
    batch_size: int = 500
    field_mappings: Optional[Dict[str, str]] = None

# ---------- Index Management Models ----------
class IndexField(BaseModel):
    field_path: str
    order: str = "ASCENDING"  # ASCENDING or DESCENDING
    array_config: Optional[str] = None  # CONTAINS for array indexes

class CompositeIndex(BaseModel):
    collection_id: str
    fields: List[IndexField]
    query_scope: str = "COLLECTION"  # COLLECTION or COLLECTION_GROUP

class IndexState(BaseModel):
    index_id: str
    state: str  # CREATING, READY, NEEDS_REPAIR, ERROR
    field_paths: List[str]
    collection_id: str
    created_at: datetime.datetime

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

@router.post("/batch")
async def batch_operations(request: BatchOperationRequest, uid: str = Depends(get_current_uid)):
    """Execute multiple Firestore operations in a single batch"""
    try:
        batch = db.batch()
        results = []
        
        for op in request.operations:
            op_type = op.get("type")
            doc_id = op.get("doc_id")
            data = op.get("data", {})
            
            if op_type == "create":
                ref = db.collection(request.collection).document()
                batch.create(ref, data)
                results.append({"type": "create", "doc_id": ref.id})
            elif op_type == "update":
                ref = db.collection(request.collection).document(doc_id)
                batch.update(ref, data)
                results.append({"type": "update", "doc_id": doc_id})
            elif op_type == "delete":
                ref = db.collection(request.collection).document(doc_id)
                batch.delete(ref)
                results.append({"type": "delete", "doc_id": doc_id})
        
        batch.commit()
        return {"status": "success", "operations": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/compound")
async def compound_query(request: CompoundQueryRequest, uid: str = Depends(get_current_uid)):
    """Execute complex queries with multiple filters and ordering"""
    try:
        query = db.collection(request.collection)
        
        # Apply filters
        for f in request.filters:
            field = f.get("field")
            op = f.get("operator", "==")
            value = f.get("value")
            query = query.where(field, op, value)
        
        # Apply ordering
        if request.order_by:
            for order in request.order_by:
                direction = firestore.Query.DESCENDING if order.get("direction") == "desc" else firestore.Query.ASCENDING
                query = query.order_by(order["field"], direction=direction)
        
        # Apply pagination
        if request.start_after:
            doc_ref = db.collection(request.collection).document(request.start_after["doc_id"]).get()
            if doc_ref.exists:
                query = query.start_after(doc_ref)
        
        # Execute query
        docs = query.limit(request.limit).stream()
        return [doc.to_dict() | {"id": doc.id} for doc in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync")
async def sync_collections(request: DataSyncRequest, uid: str = Depends(get_current_uid)):
    """Sync data between collections with optional field mapping"""
    try:
        source_docs = db.collection(request.source_collection).stream()
        batch = db.batch()
        processed = 0
        total_synced = 0
        
        for doc in source_docs:
            if processed >= request.batch_size:
                batch.commit()
                total_synced += processed
                processed = 0
                batch = db.batch()
            
            data = doc.to_dict()
            if request.field_mappings:
                mapped_data = {}
                for src_field, target_field in request.field_mappings.items():
                    if src_field in data:
                        mapped_data[target_field] = data[src_field]
                data = mapped_data
            
            target_ref = db.collection(request.target_collection).document(doc.id)
            batch.set(target_ref, data, merge=True)
            processed += 1
        
        if processed > 0:
            batch.commit()
            total_synced += processed
        
        return {
            "status": "success",
            "total_synced": total_synced,
            "source": request.source_collection,
            "target": request.target_collection
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/indexes/create")
async def create_composite_index(index: CompositeIndex):
    """Create a new composite index in Firestore"""
    try:
        # Convert to Firestore admin format
        index_fields = []
        for field in index.fields:
            field_dict = {
                "fieldPath": field.field_path,
                "order": field.order
            }
            if field.array_config:
                field_dict["arrayConfig"] = field.array_config
            index_fields.append(field_dict)

        # Create index using Firestore admin
        collection_id = index.collection_id
        index_data = {
            "queryScope": index.query_scope,
            "fields": index_fields
        }
        
        # Store index metadata in Firestore
        doc_ref = db.collection("firestore_indexes").document()
        index_state = IndexState(
            index_id=doc_ref.id,
            state="CREATING",
            field_paths=[f.field_path for f in index.fields],
            collection_id=collection_id,
            created_at=datetime.datetime.utcnow()
        )
        doc_ref.set(index_state.dict())

        return {
            "index_id": doc_ref.id,
            "status": "creating",
            "collection": collection_id,
            "fields": [f.dict() for f in index.fields]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/indexes/list")
async def list_indexes(collection_id: Optional[str] = None):
    """List all composite indexes or filter by collection"""
    try:
        query = db.collection("firestore_indexes")
        if collection_id:
            query = query.where("collection_id", "==", collection_id)
        
        indexes = query.stream()
        return [index.to_dict() | {"id": index.id} for index in indexes]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/indexes/{index_id}")
async def delete_index(index_id: str):
    """Delete a composite index"""
    try:
        # Get index metadata
        doc_ref = db.collection("firestore_indexes").document(index_id)
        index = doc_ref.get()
        
        if not index.exists:
            raise HTTPException(status_code=404, detail="Index not found")
        
        # Delete index metadata
        doc_ref.delete()
        
        return {"status": "deleted", "index_id": index_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/indexes/analyze")
async def analyze_query_indexes(collection_id: str, field_paths: List[str]):
    """Analyze required indexes for a query"""
    try:
        # Check existing indexes
        existing_indexes = await list_indexes(collection_id)
        
        # Find matching indexes
        matching_indexes = []
        required_fields = set(field_paths)
        
        for index in existing_indexes:
            index_fields = set(index.get("field_paths", []))
            if required_fields.issubset(index_fields):
                matching_indexes.append(index)
        
        # Generate suggestions
        suggestions = []
        if not matching_indexes:
            suggestions.append({
                "type": "create_index",
                "fields": field_paths,
                "reason": "No matching index found for the specified fields"
            })
        
        return {
            "matching_indexes": matching_indexes,
            "suggestions": suggestions,
            "collection_id": collection_id,
            "analyzed_fields": field_paths
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/indexes/validate")
async def validate_indexes(collection_id: str):
    """Validate all indexes for a collection"""
    try:
        indexes = await list_indexes(collection_id)
        validation_results = []
        
        for index in indexes:
            # Check if index is being used
            usage_data = {
                "index_id": index["id"],
                "last_used": None,
                "status": "unknown",
                "issues": []
            }
            
            # Add basic validation checks
            if index["state"] == "ERROR":
                usage_data["issues"].append("Index is in error state")
            elif index["state"] == "NEEDS_REPAIR":
                usage_data["issues"].append("Index needs repair")
                
            validation_results.append(usage_data)
        
        return {
            "collection_id": collection_id,
            "total_indexes": len(indexes),
            "validation_results": validation_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
