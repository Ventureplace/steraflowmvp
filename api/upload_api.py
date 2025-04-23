"""
Upload API for Steraflow

Handles:
1. Creating drag‑and‑drop upload links for suppliers
2. File uploads (multipart/form‑data) directly to Firebase Storage
3. Optional confirmation step (could be called by a bot or webhook)
4. Listing uploaded files with flexible filters
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid, datetime, logging

import firebase_admin
from firebase_admin import credentials, firestore, storage as fb_storage

# ---------- Firebase initialisation ----------
# NOTE: Assumes the same service‑account JSON you used elsewhere.
cred_path = "production-build-v1-firebase-adminsdk-fbsvc-9f9ca003be.json"
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(
        credentials.Certificate(cred_path),
        {
            # This bucket name is visible in Firebase console
            "storageBucket": "production-build-v1.firebasestorage.app"
        }
    )

db = firestore.client()
bucket = fb_storage.bucket("production-build-v1.firebasestorage.app")
# ---------- FastAPI router ----------
router = APIRouter(prefix="/api/v1")

# ---------- Pydantic models ----------
class LinkCreateRequest(BaseModel):
    supplier: Optional[str] = None
    product: Optional[str] = None
    expires_hours: int = 72

class LinkCreateResponse(BaseModel):
    link_id: str
    upload_url: str
    expires_at: datetime.datetime

# ---------- Enhanced Upload Models ----------
class FileValidationConfig(BaseModel):
    allowed_formats: List[str] = ["pdf", "xlsx", "csv", "doc", "docx"]
    max_file_size_mb: int = 50
    virus_scan: bool = True
    content_validation: bool = True

class BatchUploadRequest(BaseModel):
    files_metadata: List[Dict[str, Any]]
    validation_config: Optional[FileValidationConfig] = None

class ProcessingStatus(BaseModel):
    file_id: str
    status: str  # pending, processing, completed, failed
    steps_completed: List[str] = []
    current_step: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime.datetime
    updated_at: datetime.datetime

# ---------- /upload/link/create ----------
@router.post("/upload/link/create", response_model=LinkCreateResponse)
async def create_upload_link(payload: LinkCreateRequest):
    """
    Generate a unique link ID that maps to an upload slot.
    A supplier will POST the file to /upload/submit?link_id={link_id}
    """
    try:
        link_id = str(uuid.uuid4())
        expires_at = datetime.datetime.utcnow() + datetime.timedelta(hours=payload.expires_hours)

        # Persist link metadata
        db.collection("upload_links").document(link_id).set({
            "supplier": payload.supplier,
            "product": payload.product,
            "expires_at": expires_at,
            "created_at": datetime.datetime.utcnow(),
            "confirmed": False
        })

        return LinkCreateResponse(
            link_id=link_id,
            upload_url=f"/api/v1/upload/submit?link_id={link_id}",
            expires_at=expires_at
        )
    except Exception as e:
        logging.exception("Error creating upload link")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- /upload/submit ----------
@router.post("/upload/submit")
async def upload_submit(link_id: str, file: UploadFile = File(...)):
    """
    Sellers (or an internal UI) POST a file along with the link_id obtained earlier.
    The file is saved to Firebase Storage, and metadata is written to Firestore.
    """
    try:
        # Validate link
        link_doc = db.collection("upload_links").document(link_id).get()
        if not link_doc.exists:
            raise HTTPException(status_code=404, detail="Invalid or expired link")

        link_data = link_doc.to_dict()
        if link_data["expires_at"] < datetime.datetime.utcnow():
            raise HTTPException(status_code=400, detail="Link expired")

        # Upload file to Storage
        blob_path = f"uploads/{link_id}/{file.filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_file(file.file, content_type=file.content_type)

        # Save file metadata
        metadata = {
            "file_id": blob_path,
            "filename": file.filename,
            "supplier": link_data.get("supplier"),
            "file_format": file.filename.split(".")[-1].lower(),
            "source": "upload_link",
            "uploaded_at": datetime.datetime.utcnow(),
            "link_id": link_id,
            "product": link_data.get("product")
        }
        db.collection("files_metadata").document(blob_path).set(metadata)

        return {"status": "uploaded", "file_id": blob_path}
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- /upload/confirm ----------
@router.post("/upload/confirm")
async def upload_confirm(link_id: str):
    """
    Mark an upload as confirmed. Can be called by a bot or an internal UI.
    """
    try:
        doc_ref = db.collection("upload_links").document(link_id)
        if not doc_ref.get().exists:
            raise HTTPException(status_code=404, detail="Link not found")
        doc_ref.update({
            "confirmed": True,
            "confirmed_at": datetime.datetime.utcnow()
        })
        return {"status": "confirmed"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.exception("Confirm failed")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- /upload/list ----------
@router.get("/upload/list")
async def upload_list(supplier: Optional[str] = None,
                      product: Optional[str] = None,
                      limit: int = 100):
    """
    List uploaded files, filterable by supplier or product.
    """
    try:
        query = db.collection("files_metadata")
        if supplier:
            query = query.where("supplier", "==", supplier)
        if product:
            query = query.where("product", "==", product)
        docs = (
            query.order_by("uploaded_at", direction=firestore.Query.DESCENDING)
                 .limit(limit)
                 .stream()
        )
        return [d.to_dict() | {"id": d.id} for d in docs]
    except Exception as e:
        logging.exception("List failed")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- /upload/test ----------
@router.post("/upload/test")
async def test_upload(file: UploadFile = File(...)):
    """
    Upload a file directly to Firebase Storage to test connectivity.
    """
    try:
        blob_path = f"test_uploads/{file.filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_file(file.file, content_type=file.content_type)
        public_url = f"https://storage.googleapis.com/{bucket.name}/{blob_path}"
        return {
            "status": "success",
            "public_url": public_url,
            "path": blob_path
        }
    except Exception as e:
        logging.exception("Test upload failed")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- File Validation & Processing ----------
def validate_file(file: UploadFile, config: FileValidationConfig) -> List[str]:
    """Validate file against configuration"""
    errors = []
    
    # Check file format
    file_format = file.filename.split(".")[-1].lower()
    if file_format not in config.allowed_formats:
        errors.append(f"File format {file_format} not allowed. Allowed formats: {config.allowed_formats}")
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)  # Reset position
    
    if size_mb > config.max_file_size_mb:
        errors.append(f"File size {size_mb:.1f}MB exceeds maximum allowed size of {config.max_file_size_mb}MB")
    
    return errors

@router.post("/upload/batch")
async def batch_upload(request: BatchUploadRequest):
    """Upload multiple files in a batch with shared configuration"""
    try:
        results = []
        for metadata in request.files_metadata:
            link_response = await create_upload_link(LinkCreateRequest(
                supplier=metadata.get("supplier"),
                product=metadata.get("product")
            ))
            results.append({
                "link_id": link_response.link_id,
                "upload_url": link_response.upload_url,
                "metadata": metadata
            })
        
        return {
            "batch_id": str(uuid.uuid4()),
            "uploads": results
        }
    except Exception as e:
        logging.exception("Batch upload creation failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/submit/validated")
async def upload_submit_validated(
    link_id: str,
    file: UploadFile = File(...),
    validation_config: Optional[FileValidationConfig] = None
):
    """Enhanced file upload with validation and processing status tracking"""
    try:
        # Use default config if none provided
        config = validation_config or FileValidationConfig()
        
        # Validate file
        errors = validate_file(file, config)
        if errors:
            raise HTTPException(status_code=400, detail={"validation_errors": errors})
        
        # Create processing status
        status_id = str(uuid.uuid4())
        status = ProcessingStatus(
            file_id=status_id,
            status="pending",
            created_at=datetime.datetime.utcnow(),
            updated_at=datetime.datetime.utcnow()
        )
        
        # Store status in Firestore
        db.collection("file_processing_status").document(status_id).set(status.dict())
        
        # Perform regular upload
        upload_result = await upload_submit(link_id, file)
        
        # Update status with file_id
        db.collection("file_processing_status").document(status_id).update({
            "file_id": upload_result["file_id"],
            "status": "completed",
            "steps_completed": ["validation", "upload"],
            "updated_at": datetime.datetime.utcnow()
        })
        
        return {**upload_result, "status_id": status_id}
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.exception("Validated upload failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/upload/status/{status_id}")
async def get_upload_status(status_id: str):
    """Get the current status of a file upload and processing"""
    try:
        doc = db.collection("file_processing_status").document(status_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Status not found")
        return doc.to_dict()
    except Exception as e:
        logging.exception("Status check failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/upload/delete/{file_id}")
async def delete_upload(file_id: str):
    """Delete an uploaded file and its metadata"""
    try:
        # Delete from Storage
        blob = bucket.blob(file_id)
        if blob.exists():
            blob.delete()
        
        # Delete metadata
        db.collection("files_metadata").document(file_id).delete()
        
        return {"status": "deleted", "file_id": file_id}
    except Exception as e:
        logging.exception("Delete failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/process/{file_id}")
async def process_upload(file_id: str):
    """Trigger post-upload processing for a file"""
    try:
        # Get file metadata
        doc = db.collection("files_metadata").document(file_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="File not found")
        
        metadata = doc.to_dict()
        file_format = metadata.get("file_format", "").lower()
        
        # Create processing status
        status = ProcessingStatus(
            file_id=file_id,
            status="processing",
            created_at=datetime.datetime.utcnow(),
            updated_at=datetime.datetime.utcnow()
        )
        
        status_id = str(uuid.uuid4())
        db.collection("file_processing_status").document(status_id).set(status.dict())
        
        # TODO: Add actual processing logic based on file type
        # For now, just mark as completed
        db.collection("file_processing_status").document(status_id).update({
            "status": "completed",
            "steps_completed": ["processing"],
            "updated_at": datetime.datetime.utcnow()
        })
        
        return {
            "status": "processing_complete",
            "file_id": file_id,
            "status_id": status_id
        }
    except Exception as e:
        logging.exception("Processing failed")
        raise HTTPException(status_code=500, detail=str(e))