import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

import scanner_v1

app = FastAPI(title="SignalDrone Scanner API")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "scanner_output"
JSON_FILE = OUTPUT_DIR / "scanner_app_data.json"


async def scanner_loop():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            await scanner_v1.run_one_scan()
        except Exception as e:
            print(f"Scanner loop error: {e}")

        await asyncio.sleep(scanner_v1.SCAN_EVERY_SECONDS)


@app.on_event("startup")
async def startup_event():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.create_task(scanner_loop())


@app.get("/")
async def home():
    return {
        "app": "SignalDrone AI Scanner",
        "status": "running",
        "json_endpoint": "/scanner_app_data.json",
        "mode": "paper_trading_research_only",
    }


@app.get("/scanner_app_data.json")
async def scanner_app_data():
    if JSON_FILE.exists():
        return FileResponse(JSON_FILE, media_type="application/json")

    return JSONResponse(
        status_code=503,
        content={
            "app": "SignalDrone AI",
            "status": "waiting_for_first_scan",
            "message": "Scanner data is not ready yet. Try again after the first scan finishes.",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
