"""
Long-Form Memory API — main entry point.
"""

from fastapi import FastAPI
from src.api.routes import router
from src.storage.db import init_db
from dotenv import load_dotenv

load_dotenv()

# Initialize DB on startup
init_db()

app = FastAPI(
    title="Long-Form Memory API",
    description="Real-time long-form memory system for AI agents — IITG Hackathon",
    version="2.0.0"
)

# Register all routes
app.include_router(router)


@app.get("/")
def root():
    return {"status": "running", "message": "Long-Form Memory API v2.0 is live!"}


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)