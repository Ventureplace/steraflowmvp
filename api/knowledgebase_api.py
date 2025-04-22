"""
Knowledgebase & Parsing API  – Phase 1
--------------------------------------

This module currently exposes only the `/parse/email` endpoint.  It is a **stub**:
• OAuth / Gmail client creation is TODO.
• Actual LLM extraction is TODO.

Once the Gmail client fetches the `raw` or `full` message,
we will pass it to an extractor that returns structured RFQ metadata.

Add this router to `main.py` with:
    from api.knowledgebase_api import router as kb_router
    app.include_router(kb_router)
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Header, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import datetime, logging, uuid
import os, base64, json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import firebase_admin
from firebase_admin import credentials as fb_credentials, firestore, auth
import pdfplumber 
import re
from google.cloud import storage
import os
import textwrap



try:
    firebase_admin.get_app()
except ValueError:
    cred_path = os.getenv("FIREBASE_ADMIN_KEY", "production-build-v1-firebase-adminsdk-fbsvc-9f9ca003be.json")
    firebase_admin.initialize_app(fb_credentials.Certificate(cred_path))
fs_db = firestore.client()

# ---------- FastAPI Router ----------
router = APIRouter(prefix="/api/v1")

# ---------- Pydantic Models ----------

# --- Knowledgebase Add Model ---
class KnowledgebaseAddRequest(BaseModel):
    """Payload for adding text into the agent RAG knowledge‑base."""
    text: str
    tag: Optional[str] = None      # e.g. "quote", "spec", "email"
    product: Optional[str] = None  # e.g. "Hinge‑V3"
    metadata: Optional[Dict[str, Any]] = None
    chunk_size: int = 500          # characters per chunk
class EmailParseRequest(BaseModel):
    """Input body for /parse/email.
    The user_id will be injected from the verified Firebase token."""
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    raw_message: Optional[str] = None  # base64url if supplied via Pub/Sub

class EmailParseResult(BaseModel):
    """Minimal response indicating the parse job was queued."""
    parse_id: str
    parsed_at: datetime.datetime

def get_gmail_service(user_id: str):
    """
    Helper to create a Gmail API service client for a given user.
    """
    cred_doc = fs_db.collection("gmail_creds").document(user_id).get()
    if not cred_doc.exists:
        raise HTTPException(status_code=404, detail=f"No Gmail credentials found for user {user_id}")
    creds = Credentials.from_authorized_user_info(cred_doc.to_dict())
    service = build("gmail", "v1", credentials=creds)
    return service

# ---------- Background worker ----------
def _parse_email_bg(request: EmailParseRequest) -> Dict[str, Any]:
    """
    Background worker:
    • Fetch the full Gmail thread (all messages) using Gmail API.
    • Extract all message content including headers, text, and HTML.
    • Store complete thread data in Firestore.
    """
    logging.info(f"[parse_email] BG start – thread_id={request.thread_id}")

    # --- 0. Load Gmail credentials from Firestore ---
    cred_doc = fs_db.collection("gmail_creds").document(request.user_id).get()
    if not cred_doc.exists:
        logging.error(f"No Gmail credentials found for user {request.user_id}")
        return {}
    creds = Credentials.from_authorized_user_info(cred_doc.to_dict())

    # --- 1. Build Gmail service & fetch thread ---
    try:
        service = build("gmail", "v1", credentials=creds)
        thread = service.users().threads().get(userId="me", id=request.thread_id, format="full").execute()
    except Exception as e:
        logging.exception("Gmail thread fetch failed")
        return {}

    # --- 2. Extract content from every message ---
    messages_data = []
    for msg in thread.get("messages", []):
        try:
            # Get headers
            headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
            
            # Process message parts
            payload = msg.get("payload", {})
            text_content = []
            html_content = []
            attachments = []
            
            def process_parts(part, depth=0):
                if "parts" in part:
                    for p in part["parts"]:
                        process_parts(p, depth + 1)
                else:
                    mime_type = part.get("mimeType", "")
                    if mime_type.startswith("text/plain"):
                        data = part.get("body", {}).get("data")
                        if data:
                            decoded = base64.urlsafe_b64decode(data).decode(errors="ignore")
                            text_content.append(decoded)
                    elif mime_type.startswith("text/html"):
                        data = part.get("body", {}).get("data")
                        if data:
                            decoded = base64.urlsafe_b64decode(data).decode(errors="ignore")
                            html_content.append(decoded)
                    elif "attachmentId" in part.get("body", {}):
                        attachments.append({
                            "id": part["body"]["attachmentId"],
                            "filename": part.get("filename"),
                            "mime_type": mime_type,
                            "size": part["body"].get("size", 0)
                        })
            
            process_parts(payload)
            
            messages_data.append({
                "message_id": msg["id"],
                "thread_id": msg["threadId"],
                "headers": headers,
                "text_content": "\n".join(text_content) if text_content else None,
                "html_content": "\n".join(html_content) if html_content else None,
                "attachments": attachments,
                "internal_date": datetime.datetime.fromtimestamp(int(msg["internalDate"]) / 1000),
                "label_ids": msg.get("labelIds", [])
            })
        except Exception as e:
            logging.exception(f"Failed to process message {msg.get('id')} in thread {request.thread_id}")
            continue

    if not messages_data:
        logging.error(f"No messages could be processed in thread {request.thread_id}")
        return {}

    # --- 3. Store complete thread data in Firestore ---
    doc_id = str(uuid.uuid4())
    thread_data = {
        "user_id": request.user_id,
        "thread_id": request.thread_id,
        "messages": messages_data,
        "created_at": datetime.datetime.utcnow(),
        "extracted": {},  # Will be populated by LLM later
        "subject": messages_data[0]["headers"].get("subject"),
        "participants": list(set(
            addr for msg in messages_data 
            for field in ["from", "to", "cc"] 
            for addr in msg["headers"].get(field, "").split(",")
        )),
        "start_date": min(msg["internal_date"] for msg in messages_data),
        "end_date": max(msg["internal_date"] for msg in messages_data),
        "message_count": len(messages_data),
        "has_attachments": any(msg["attachments"] for msg in messages_data)
    }

    fs_db.collection("email_raw").document(doc_id).set(thread_data)
    logging.info(f"[parse_email] BG complete – stored doc {doc_id}")
    return {"parse_id": doc_id, "parsed_at": datetime.datetime.utcnow()}

# ---------- Firebase Auth dependency ----------
def get_current_uid(authorization: str = Header(...)) -> str:
    """
    Validate Firebase ID‑token from 'Authorization: Bearer <token>'
    and return the user's UID.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token")
    id_token = authorization.split()[1]
    try:
        decoded = auth.verify_id_token(id_token)
        return decoded["uid"]
    except Exception as e:
        raise HTTPException(401, "Invalid or expired Firebase token")

# # ---------- /parse/email ----------
# @router.post("/parse/email", response_model=EmailParseResult)
# async def parse_email(payload: EmailParseRequest,
#                       background_tasks: BackgroundTasks,
#                       uid: str = Depends(get_current_uid)):
#     """
#     Entry point for parsing Gmail threads or raw messages.
#     Runs heavy processing in a background task and immediately
#     returns a stubbed response object.
#     """
#     try:
#         immediate_id = str(uuid.uuid4())
#         req = EmailParseRequest(**payload.dict(), user_id=uid)
#         background_tasks.add_task(_parse_email_bg, req)
#         return EmailParseResult(
#             parse_id=immediate_id,
#             parsed_at=datetime.datetime.utcnow()
#         )
#     except Exception as e:
#         logging.exception("parse_email failed")
#         raise HTTPException(status_code=500, detail=str(e))

# # ---------- /test/gmail/threads ----------
# @router.get("/test/gmail/threads")
# async def test_gmail_threads(uid: str = Depends(get_current_uid)):
#     """
#     Dev-only: fetch list of recent Gmail threads for validation.
#     """
#     try:
#         service = get_gmail_service(uid)
#         threads = service.users().threads().list(userId="me", maxResults=5).execute()
#         return {"threads": threads.get("threads", [])}
#     except Exception as e:
#         logging.exception("Failed to list Gmail threads")
#         raise HTTPException(status_code=500, detail=str(e))


# ---------- Pydantic Models ----------
class PdfParseRequest(BaseModel):
    """Input for /parse/pdf/tables endpoint"""
    file_id: str

#---------- /download from firebase ----------
def download_from_firebase(file_id: str) -> str:
    """
    Downloads a file from Firebase Storage to /tmp inside a Cloud Run container.
    Returns the full local path to the downloaded file.
    """
    bucket_name = "production-build-v1.firebasestorage.app"
    temp_path = f"/tmp/{os.path.basename(file_id)}"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_id)
    blob.download_to_filename(temp_path)

    return temp_path

# ---------- /parse/pdf/tables ----------
# may need to tweak this to lower costs
@router.post("/parse/pdf/tables")
async def parse_pdf_tables(payload: PdfParseRequest, uid: str = Depends(get_current_uid)):
    """
    Extracts raw tables from a supplier PDF and returns a list of rows (no AI parsing yet).
    """
    file_path = download_from_firebase(payload.file_id)  # TODO: you need this util

    parsed_rows = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    if row and any(cell for cell in row):  # ignore empty rows
                        parsed_rows.append([cell.strip() if cell else "" for cell in row])

    doc_id = str(uuid.uuid4())
    fs_db.collection("pdf_table_raw").document(doc_id).set({
        "user_id": uid,
        "file_id": payload.file_id,
        "rows": parsed_rows,
        "created_at": datetime.datetime.utcnow()
    })

    return {"rows": parsed_rows, "doc_id": doc_id}

#--knowledgebase/add----------
@router.post("/knowledgebase/add")
async def add_knowledgebase_entry(payload: KnowledgebaseAddRequest,
                                  uid: str = Depends(get_current_uid)):
    """
    Break the supplied text into chunks and store them in Firestore
    under `agent_context` for Retrieval‑Augmented Generation (RAG).
    """
    try:
        chunks = [c.strip() for c in textwrap.wrap(payload.text, payload.chunk_size) if c.strip()]
        doc_ids = []
        for idx, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid4())
            fs_db.collection("agent_context").document(doc_id).set({
                "user_id": uid,
                "text": chunk,
                "tag": payload.tag,
                "product": payload.product,
                "metadata": payload.metadata or {},
                "chunk_index": idx,
                "chunk_count": len(chunks),
                "created_at": datetime.datetime.utcnow()
            })
            doc_ids.append(doc_id)
        return {"status": "stored", "chunks": len(chunks), "doc_ids": doc_ids}
    except Exception as e:
        logging.exception("Failed to add knowledge‑base entry")
        raise HTTPException(status_code=500, detail=str(e))

 # ---------- /knowledgebase/query ----------
@router.get("/knowledgebase/query")
async def query_knowledgebase(tag: Optional[str] = None,
                              product: Optional[str] = None,
                              limit: int = 100,
                              uid: str = Depends(get_current_uid)):
    """
    Retrieve knowledge‑base chunks filtered by tag/product for the
    authenticated user. Results are ordered by newest first.
    """
    try:
        query = (
            fs_db.collection("agent_context")
                 .where("user_id", "==", uid)
        )
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
        logging.exception("Knowledge‑base query failed")
        raise HTTPException(status_code=500, detail=str(e))
    
from typing import Optional

#---------- /knowledgebase/list ----------
@router.get("/knowledgebase/list")
async def list_knowledgebase_entries(uid: str = Depends(get_current_uid),
                                     cursor: Optional[str] = None,
                                     page_size: int = 100):
    """
    List knowledge‑base entries for the authenticated user, paginated.

    • `cursor` – Firestore document ID to start after (optional)
    • `page_size` – max docs to return (default 100)

    Returns: { entries: [...], next_cursor: str | None }
    """
    try:
        col = fs_db.collection("agent_context").where("user_id", "==", uid).order_by(
            "created_at", direction=firestore.Query.DESCENDING
        )

        if cursor:
            # start_after needs a document snapshot
            snap = fs_db.collection("agent_context").document(cursor).get()
            if snap.exists:
                col = col.start_after(snap)

        docs = col.limit(page_size).stream()
        entries = []
        last_id = None
        for d in docs:
            entries.append(d.to_dict() | {"id": d.id})
            last_id = d.id

        next_cursor = last_id if len(entries) == page_size else None
        return {"entries": entries, "next_cursor": next_cursor}
    except Exception as e:
        logging.exception("Knowledge‑base list failed")
        raise HTTPException(status_code=500, detail=str(e))