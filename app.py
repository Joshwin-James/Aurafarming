import os
import uuid
import shutil
import tempfile
import subprocess
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import edit_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Formats OpenCV can typically read directly without needing conversion.
NATIVE_FORMATS = {".mp4", ".avi", ".mov", ".mkv"}
# Formats accepted from the client, but that may need conversion to mp4
# since OpenCV's build may not have codecs to read them reliably.
SUPPORTED_UPLOAD_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".3gp"}


def convert_video_to_mp4(input_path, output_path):
    """Convert a video file to mp4 (h264/aac) using ffmpeg.

    Raises RuntimeError if ffmpeg is unavailable or the conversion fails.
    """
    ffmpeg_cmd = shutil.which("ffmpeg")
    if not ffmpeg_cmd:
        raise RuntimeError("ffmpeg is not available on this system; cannot convert video.")

    cmd = [
        ffmpeg_cmd, "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        output_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0 or not os.path.exists(output_path):
        stderr = result.stderr.decode("utf-8", errors="ignore") if result.stderr else ""
        raise RuntimeError(f"ffmpeg conversion failed: {stderr[-500:]}")
    return output_path

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

    ext = os.path.splitext(video.filename or "")[1].lower() or ".webm"
    if ext not in SUPPORTED_UPLOAD_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Supported formats: "
                   f"{', '.join(sorted(SUPPORTED_UPLOAD_FORMATS))}."
        )

    temp_dir = tempfile.mkdtemp(prefix="aurafarming_")
    try:
        # Save uploaded video
        input_filename = f"user_clip{ext}"
        input_path = os.path.join(temp_dir, input_filename)
        
        with open(input_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded video format not supported or corrupted."
            )

        # Convert to mp4 if the uploaded format isn't natively OpenCV-friendly,
        # so downstream processing can reliably read it regardless of the
        # original container/codec.
        processed_path = input_path
        if ext not in NATIVE_FORMATS:
            converted_path = os.path.join(temp_dir, "user_clip_converted.mp4")
            if shutil.which("ffmpeg"):
                try:
                    convert_video_to_mp4(input_path, converted_path)
                    processed_path = converted_path
                except Exception as conv_err:
                    print(f"Video conversion error: {conv_err}")
                    raise HTTPException(
                        status_code=400,
                        detail="Video file could not be read (check format: mp4, avi, mov, webm)."
                    )
            else:
                print("ffmpeg not available; attempting to process original file format directly.")

        output_filename = f"edit_{style}_{uuid.uuid4().hex[:8]}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Process edit
        edit_engine.generate_edit(
            user_files=[processed_path],
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
    except HTTPException:
        raise
    except edit_engine.VideoReadError as e:
        print(f"Video read error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Edit generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate edit: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"AuraFarming server launching on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
