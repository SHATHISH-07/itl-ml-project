from fastapi import FastAPI
from contextlib import asynccontextmanager
from core import loader
from routes import query, train
import torch

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading resources...")
    yield
    print("🛑 Shutting down...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="Employee Performance AI Backend",
    version="2.0.0",
    lifespan=lifespan
)

app.include_router(train.router)
app.include_router(query.router)

@app.get("/")
def health():
    return {"status": "running"}