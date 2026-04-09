from fastapi import FastAPI
from contextlib import asynccontextmanager
from core import loader
from routes import query, train
import torch

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load everything on startup
    loader.load_all_resources()
    yield
    # Cleanup on shutdown
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="Employee Performance AI Backend",
    description="End-to-end API for training, forecasting, and NLP summarization.",
    version="1.0.0",
    lifespan=lifespan
)

# Include both Routers
app.include_router(train.router)
app.include_router(query.router)

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "API is running. Models loaded."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)