import os
import uuid
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import edit_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="AuraFarming AI Videographer Engine", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static audio/video/assets and generated outputs
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

@app.get("/")
async def get_index():
    index_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)

@app.get("/index.css")
async def get_css():
    return FileResponse(os.path.join(BASE_DIR, "index.css"), media_type="text/css")

@app.get("/app.js")
async def get_js():
    return FileResponse(os.path.join(BASE_DIR, "app.js"), media_type="application/javascript")

@app.post("/api/generate-edit")
async def generate_edit_endpoint(
    video: UploadFile = File(...),
    style: str = Form("sigma")
):
    valid_styles = ["sigma", "second", "third"]
    if style not in valid_styles:
        style = "sigma"
        
    temp_dir = tempfile.mkdtemp(prefix="aurafarming_")
    try:
        # Save uploaded video
        ext = os.path.splitext(video.filename)[1] or ".webm"
        input_filename = f"user_clip{ext}"
        input_path = os.path.join(temp_dir, input_filename)
        
        with open(input_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
            
        output_filename = f"edit_{style}_{uuid.uuid4().hex[:8]}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Process edit
        edit_engine.generate_edit(
            user_files=[input_path],
            edit_style=style,
            base_dir=BASE_DIR,
            output_path=output_path
        )
        
        video_url = f"/output/{output_filename}"
        return JSONResponse({
            "status": "success",
            "video_url": video_url,
            "style": style,
            "filename": output_filename
        })
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"message": f"Failed to generate edit: {str(e)}", "traceback": tb})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"AuraFarming server launching on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
