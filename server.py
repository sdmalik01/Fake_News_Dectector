# server.py (move static mount below routes)
import asyncio
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict
import uvicorn
from pathlib import Path

# Import checker
from claim_checker import check_claim

app = FastAPI(title="Claim Checker API")

class ClaimIn(BaseModel):
    headline: str
    description: str = ""

async def run_check_in_executor(headline: str, description: str) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    combined = headline.strip()
    if description and description.strip():
        combined = f"{combined}. {description.strip()}"
    result = await loop.run_in_executor(None, check_claim, combined)
    return result

@app.post("/api/check")
async def api_check(payload: ClaimIn):
    headline = payload.headline
    description = payload.description
    try:
        result = await run_check_in_executor(headline, description)
        if isinstance(result, dict) and "json" in result:
            return JSONResponse(content=result["json"])
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Mount static files AFTER defining API routes so /api/* is handled by routes ---
STATIC_DIR = Path(__file__).resolve().parent / "static"
if not STATIC_DIR.exists():
    raise SystemExit(f"Missing static directory: {STATIC_DIR}")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
