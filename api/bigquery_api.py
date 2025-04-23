from fastapi import FastAPI, HTTPException, APIRouter
from google.cloud import bigquery
from google.oauth2 import service_account
from typing import List, Dict, Any, Optional
import json
import os
from pydantic import BaseModel
from google.cloud import firestore
import datetime
import re
import uuid
from fastapi import Depends

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

# ---------- Advanced Query Models ----------
class ScheduledQueryRequest(BaseModel):
    query: str
    schedule: str  # cron expression
    destination_table: Optional[str] = None
    description: Optional[str] = None

class QueryMetricsRequest(BaseModel):
    query: str
    dry_run: bool = True

class DataProfileRequest(BaseModel):
    dataset_id: str
    table_id: str
    sample_size: Optional[int] = 1000

# ---------- Cache and Template Models ----------
class QueryTemplate(BaseModel):
    name: str
    description: Optional[str] = None
    query_template: str
    parameters: List[Dict[str, str]]  # [{"name": "table", "type": "string"}, ...]
    created_by: Optional[str] = None

class QueryCache:
    def __init__(self):
        self.cache = {}  # Simple in-memory cache
        self.ttl = 300  # 5 minutes TTL

    def get(self, query_hash: str) -> Optional[Dict]:
        if query_hash in self.cache:
            result, timestamp = self.cache[query_hash]
            if (datetime.datetime.utcnow() - timestamp).seconds < self.ttl:
                return result
            del self.cache[query_hash]
        return None

    def set(self, query_hash: str, result: Dict):
        self.cache[query_hash] = (result, datetime.datetime.utcnow())

# Initialize cache
query_cache = QueryCache()

class ExportRequest(BaseModel):
    query: str
    format: str = "csv"  # csv, json, avro
    destination: Optional[str] = None  # GCS bucket path

class LineageRecord(BaseModel):
    query_id: str
    source_tables: List[str]
    destination_table: Optional[str]
    user_id: str
    timestamp: datetime.datetime
    query_text: str

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

@router.post("/query/schedule")
async def schedule_query(request: ScheduledQueryRequest):
    """Schedule a query to run periodically"""
    try:
        job_config = bigquery.QueryJobConfig()
        if request.destination_table:
            job_config.destination = client.dataset(request.destination_table.split('.')[0]).table(request.destination_table.split('.')[1])
        
        query_job = client.query(
            request.query,
            job_config=job_config
        )
        # TODO: Implement actual scheduling logic with Cloud Scheduler
        return {"job_id": query_job.job_id, "schedule": request.schedule}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/analyze")
async def analyze_query(request: QueryMetricsRequest):
    """Analyze query performance and cost before running"""
    try:
        job_config = bigquery.QueryJobConfig(dry_run=request.dry_run)
        query_job = client.query(
            request.query,
            job_config=job_config
        )
        return {
            "total_bytes_processed": query_job.total_bytes_processed,
            "total_bytes_billed": query_job.total_bytes_billed,
            "estimated_cost": (query_job.total_bytes_billed or 0) * 5 / 1e12,  # $5 per TB
            "schema": [field.to_api_repr() for field in query_job.schema] if query_job.schema else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/table/profile")
async def profile_table(request: DataProfileRequest):
    """Generate statistical profile of table data"""
    try:
        profiling_query = f"""
        SELECT
            column_name,
            COUNT(*) as total_rows,
            COUNT(DISTINCT {column_name}) as unique_values,
            COUNTIF({column_name} IS NULL) as null_count,
            MIN({column_name}) as min_value,
            MAX({column_name}) as max_value
        FROM `{request.dataset_id}.{request.table_id}`
        GROUP BY column_name
        LIMIT {request.sample_size}
        """
        
        query_job = client.query(profiling_query)
        results = query_job.result()
        
        return [dict(row) for row in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Template Management ----------
@router.post("/templates/save")
async def save_template(template: QueryTemplate):
    """Save a query template to Firestore"""
    try:
        doc_ref = firestore_client.collection("query_templates").document()
        template_dict = template.dict()
        template_dict["created_at"] = datetime.datetime.utcnow()
        doc_ref.set(template_dict)
        return {"template_id": doc_ref.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates/list")
async def list_templates():
    """List all available query templates"""
    try:
        templates = firestore_client.collection("query_templates").stream()
        return [{"id": t.id, **t.to_dict()} for t in templates]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Query Caching ----------
@router.post("/query/cached")
async def run_cached_query(request: QueryRequest):
    """Run query with caching support"""
    try:
        # Simple hash of the query for cache key
        query_hash = str(hash(request.query))
        
        # Check cache
        cached_result = query_cache.get(query_hash)
        if cached_result:
            return {"cached": True, "results": cached_result}
        
        # Run query if not cached
        query_job = client.query(request.query)
        results = [dict(row) for row in query_job.result()]
        
        # Cache results
        query_cache.set(query_hash, results)
        
        return {"cached": False, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Data Export ----------
@router.post("/query/export")
async def export_query_results(request: ExportRequest):
    """Export query results to GCS in specified format"""
    try:
        job_config = bigquery.QueryJobConfig()
        
        # Set destination format
        format_map = {
            "csv": "CSV",
            "json": "NEWLINE_DELIMITED_JSON",
            "avro": "AVRO"
        }
        
        if request.destination:
            # Export to GCS
            destination_uri = f"gs://{request.destination}"
            job_config.destination = destination_uri
            job_config.destination_format = format_map.get(request.format, "CSV")
        
        query_job = client.query(request.query, job_config=job_config)
        query_job.result()  # Wait for query to complete
        
        return {
            "status": "completed",
            "destination": request.destination,
            "format": request.format,
            "job_id": query_job.job_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Query Optimization ----------
@router.post("/query/optimize")
async def optimize_query(request: QueryRequest):
    """Simple query optimization suggestions"""
    try:
        suggestions = []
        query = request.query.lower()
        
        # Simple pattern-based suggestions
        if "select *" in query:
            suggestions.append("Specify needed columns instead of SELECT *")
        
        if "where" not in query:
            suggestions.append("Consider adding filters to reduce data scanned")
        
        if "order by" in query and "limit" not in query:
            suggestions.append("Add LIMIT clause when using ORDER BY")
        
        if "group by" in query and "having" not in query:
            suggestions.append("Consider using HAVING for grouped data filtering")
        
        # Run EXPLAIN
        job_config = bigquery.QueryJobConfig(dry_run=True)
        query_job = client.query(f"EXPLAIN {request.query}", job_config=job_config)
        
        return {
            "suggestions": suggestions,
            "bytes_processed": query_job.total_bytes_processed,
            "estimated_cost": (query_job.total_bytes_processed or 0) * 5 / 1e12  # $5 per TB
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Data Lineage ----------
@router.post("/query/lineage")
async def track_query_lineage(request: QueryRequest, uid: str = Depends(get_current_uid)):
    """Track data lineage for queries"""
    try:
        # Extract source tables using simple regex
        source_tables = []
        query_lower = request.query.lower()
        
        # Find FROM and JOIN tables
        from_matches = re.findall(r'from\s+`?([^`\s]+)`?', query_lower)
        join_matches = re.findall(r'join\s+`?([^`\s]+)`?', query_lower)
        source_tables = list(set(from_matches + join_matches))
        
        # Find destination table if INSERT/CREATE
        destination_table = None
        if "create table" in query_lower:
            create_matches = re.findall(r'create\s+table\s+`?([^`\s]+)`?', query_lower)
            if create_matches:
                destination_table = create_matches[0]
        elif "insert into" in query_lower:
            insert_matches = re.findall(r'insert\s+into\s+`?([^`\s]+)`?', query_lower)
            if insert_matches:
                destination_table = insert_matches[0]
        
        # Store lineage
        lineage = LineageRecord(
            query_id=str(uuid.uuid4()),
            source_tables=source_tables,
            destination_table=destination_table,
            user_id=uid,
            timestamp=datetime.datetime.utcnow(),
            query_text=request.query
        )
        
        doc_ref = firestore_client.collection("query_lineage").document(lineage.query_id)
        doc_ref.set(lineage.dict())
        
        return {
            "lineage_id": lineage.query_id,
            "source_tables": source_tables,
            "destination_table": destination_table
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)