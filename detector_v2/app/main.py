from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from .scanner import scan_minifigure
from .storage import CloudStorage
import os

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize storage
storage = CloudStorage()

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/scan/")
async def scan_image(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        
        # Upload to Cloud Storage
        filename = f"scans/{image.filename}"
        file_url = storage.upload_file(contents, filename)
        
        # Scan the minifigure
        result = await scan_minifigure(contents)
        
        return {
            "status": "success",
            "file_url": file_url,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
