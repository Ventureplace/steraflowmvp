from fastapi import FastAPI, HTTPException, APIRouter
from google.cloud import bigquery
from google.oauth2 import service_account
from typing import List, Dict, Any
import json
import os
from pydantic import BaseModel
from google.cloud import firestore

# Initialize FastAPI app
app = FastAPI(
    title="BigQuery API",
    description="API for interacting with BigQuery",
    version="1.0.0"
)

# Create router
router = APIRouter(prefix="/api/v1")

# Load credentials from environment or config
try:
    # Load credentials from JSON file or environment
    if os.path.exists('/Users/iyangodwin/steraflowmvp-1/production-build-v1-23b953a13a62.json'):
        credentials = service_account.Credentials.from_service_account_file('/Users/iyangodwin/steraflowmvp-1/production-build-v1-23b953a13a62.json')
    else:
        # You'll need to set this environment variable with the JSON content
        creds_json = json.loads(os.getenv('GCP_CREDENTIALS'))
        credentials = service_account.Credentials.from_service_account_info(creds_json)
    
    # Initialize BigQuery client
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    try:
        firestore_client = firestore.Client(credentials=credentials, project=credentials.project_id)
    except Exception as e:
        raise Exception(f"Failed to initialize Firestore client: {str(e)}")
except Exception as e:
    raise Exception(f"Failed to initialize BigQuery client: {str(e)}")

class QueryRequest(BaseModel):
    query: str

class SaveQueryRequest(BaseModel):
    user_id: str
    name: str
    query: str
    description: str | None = None
    tags: list[str] | None = None

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "BigQuery API",
        "version": "1.0.0",
        "status": "running",
        "docs_url": "/docs",
        "endpoints": {
            "datasets": "/api/v1/datasets",
            "tables": "/api/v1/datasets/{dataset_id}/tables",
            "query_run": "/api/v1/query/run",
            "query_save": "/api/v1/query/save",
            "query_history": "/api/v1/query/history",
            "query_metadata": "/api/v1/query/metadata/{dataset_id}/{table_id}",
            "sample": "/api/v1/table/{dataset_id}/{table_id}/sample"
        }
    }

@router.get("/datasets")
async def list_datasets() -> List[str]:
    """List all available datasets"""
    try:
        datasets = list(client.list_datasets())
        return [dataset.dataset_id for dataset in datasets]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")

@router.get("/datasets/{dataset_id}/tables")
async def list_tables(dataset_id: str) -> List[str]:
    """List all tables in a dataset"""
    try:
        tables = list(client.list_tables(dataset_id))
        return [table.table_id for table in tables]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tables: {str(e)}")

@router.post("/query/run")
async def run_query(request: QueryRequest, max_results: int = 1000) -> List[Dict[str, Any]]:
    """Execute a BigQuery query"""
    try:
        query_job = client.query(request.query)
        rows = query_job.result(max_results=max_results)
        return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

@router.post("/query/save")
async def save_query(payload: SaveQueryRequest):
    """
    Save a user-defined query along with metadata to Firestore.
    Document path: saved_queries/{user_id}/{auto-id}
    """
    try:
        doc_ref = firestore_client.collection("saved_queries").document(payload.user_id).collection("items").document()
        doc_ref.set(payload.dict())
        return {"status": "saved", "id": doc_ref.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save query: {str(e)}")

@router.get("/query/history")
async def query_history(user_id: str):
    """
    Fetch saved queries for a specific user.
    """
    try:
        docs = firestore_client.collection("saved_queries").document(user_id).collection("items").stream()
        return [{**doc.to_dict(), "id": doc.id} for doc in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch query history: {str(e)}")

@router.get("/query/metadata/{dataset_id}/{table_id}")
async def query_metadata(dataset_id: str, table_id: str):
    """
    Expose schema information for a given table.
    """
    try:
        table_ref = f"{dataset_id}.{table_id}"
        table = client.get_table(table_ref)
        return [{
            "name": field.name,
            "type": field.field_type,
            "mode": field.mode,
            "description": field.description
        } for field in table.schema]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch metadata: {str(e)}")

@router.get("/table/{dataset_id}/{table_id}/sample")
async def get_table_sample(dataset_id: str, table_id: str, sample_size: int = 5) -> List[Dict[str, Any]]:
    """Get sample rows from a table"""
    try:
        table_ref = f"{dataset_id}.{table_id}"
        table = client.get_table(table_ref)
        rows = client.list_rows(table, max_results=sample_size)
        return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get table sample: {str(e)}")

# Include router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)