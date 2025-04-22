from fastapi import FastAPI
from api.bigquery_api import router as bigquery_router
from api.firestore_api import router as firestore_router
from api.upload_api import router as upload_router
from api.knowledgebase_api import router as kb_router


app = FastAPI(title="Steraflow API", version="0.1")

# Register routers
app.include_router(bigquery_router)
app.include_router(firestore_router)
app.include_router(upload_router)
app.include_router(kb_router)



@app.get("/")
def root():
    return {"message": "Steraflow backend is live"}