import re
import math
import shlex
import asyncio
import pathlib
import subprocess
from typing import AsyncIterator

# add at top with other imports
import tempfile
from fastapi import BackgroundTasks
from fastapi.responses import FileResponse

from fastapi import FastAPI, HTTPException, Query, Response, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
# add this import near the top
import contextlib

app = FastAPI(title="Video Clip Retriever")

VIDEO_DIR = pathlib.Path("./video").resolve()
SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-\.]+$")  # allow foo, foo-bar, foo_bar, foo.mp4


def parse_hms(hms: str) -> int:
    """Parse HH:MM:SS into seconds."""
    if not re.match(r"^\d{1,2}:\d{2}:\d{2}$", hms):
        raise ValueError("time must be HH:MM:SS")
    h, m, s = map(int, hms.split(":"))
    if m >= 60 or s >= 60:
        raise ValueError("minutes/seconds must be < 60")
    return h * 3600 + m * 60 + s


def validate_and_locate(videoname: str) -> pathlib.Path:
    """Ensure safe path and file exists (assumes .mp4 under ./video)."""
    if not SAFE_NAME.match(videoname):
        raise HTTPException(status_code=400, detail="Invalid videoname.")
    # allow either `name` or `name.mp4`
    p = VIDEO_DIR / (videoname if videoname.endswith(".mp4") else f"{videoname}.mp4")
    try:
        p = p.resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Bad path.")
    if not str(p).startswith(str(VIDEO_DIR)):
        raise HTTPException(status_code=400, detail="Path traversal detected.")
    if not p.exists():
        raise HTTPException(status_code=404, detail="Video not found.")
    return p


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Video Clip Retriever</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{font-family:system-ui,Arial,Helvetica,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;}
    form{display:grid;gap:.75rem;grid-template-columns:1fr 1fr 1fr auto}
    input,button{padding:.6rem .8rem;font-size:1rem}
    video{width:100%;max-height:70vh;margin-top:1rem;background:#000;border-radius:12px}
    .row{display:flex;gap:.5rem;align-items:center;margin:.5rem 0}
    label{font-weight:600}
  </style>
</head>
<body>
  <h1>Video Clip Retriever</h1>
  <p>Enter <code>videoname</code> and time range (HH:MM:SS).</p>
  <form action="/clip" method="get">
    <input name="videoname" placeholder="myvideo (or myvideo.mp4)" required>
    <input name="start" placeholder="00:00:05" required pattern="\\d{1,2}:\\d{2}:\\d{2}">
    <input name="end" placeholder="00:00:15" required pattern="\\d{1,2}:\\d{2}:\\d{2}">
    <button type="submit">Show Clip</button>
  </form>
</body>
</html>
    """


@app.get("/clip", response_class=HTMLResponse)
def clip_page(
    videoname: str = Query(..., description="video name (with or without .mp4)"),
    start: str = Query(..., description="HH:MM:SS"),
    end: str = Query(..., description="HH:MM:SS"),
) -> str:
    # Validate early so the page shows a friendly error if needed
    try:
        src = f"/stream_file?videoname={videoname}&start={start}&end={end}"

        _ = validate_and_locate(videoname)
        s = parse_hms(start)
        e = parse_hms(end)
        if e <= s:
            raise ValueError("end must be after start")
    except Exception as exc:
        return f"""
<!doctype html>
<html><body style="font-family:system-ui;max-width:760px;margin:2rem auto">
<h2>Error</h2>
<p>{str(exc)}</p>
<p><a href="/">Back</a></p>
</body></html>
        """

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Clip: {videoname} [{start}–{end}]</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{{font-family:system-ui,Arial,Helvetica,sans-serif;max-width:980px;margin:1.5rem auto;padding:0 1rem}}
    header{{display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap}}
    h1{{font-size:1.15rem;margin:.2rem 0}}
    video{{width:100%;max-height:75vh;margin-top:1rem;background:#000;border-radius:12px}}
    a.button{{text-decoration:none;border:1px solid #ddd;border-radius:10px;padding:.5rem .8rem}}
  </style>
</head>
<body>
  <header>
    <h1>Clip: <code>{videoname}</code> <small>({start} – {end})</small></h1>
    <nav><a class="button" href="/">New clip</a></nav>
  </header>
  <video controls autoplay src="{src}"></video>
  <p><small>Direct stream URL: <code>{src}</code></small></p>
</body>
</html>
    """


async def _ffmpeg_stream(cmd: list[str], proc: subprocess.Popen) -> AsyncIterator[bytes]:
    """
    Async generator to yield ffmpeg stdout in chunks.
    Ensures process is terminated if client disconnects.
    """
    try:
        while True:
            chunk = await asyncio.get_event_loop().run_in_executor(None, proc.stdout.read, 64 * 1024)
            if not chunk:
                break
            yield chunk
    finally:
        # terminate if still running
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass


@app.get("/stream")
async def stream_clip(
    request: Request,
    videoname: str = Query(...),
    start: str = Query(..., description="HH:MM:SS"),
    end: str = Query(..., description="HH:MM:SS"),
):
    src = validate_and_locate(videoname)
    s = parse_hms(start)
    e = parse_hms(end)
    if e <= s:
        raise HTTPException(status_code=400, detail="end must be after start")

    duration = e - s

    # ffmpeg pipeline:
    # -ss after -i = accurate seeking; we re-encode a short segment for reliable playback
    # Use fragmented MP4 so it can start streaming immediately.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-i", str(src),
        "-ss", start,
        "-t", str(duration),
        "-map", "0:v:0?",
        "-map", "0:a:0?",          # best-effort include audio if present
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "frag_keyframe+empty_moov",
        "-f", "mp4",
        "pipe:1",
    ]

    try:
        proc = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg is not installed or not in PATH.")

    # If client disconnects, stop ffmpeg
    async def gen():
        async for chunk in _ffmpeg_stream(cmd, proc):
            # Stop if client disconnected
            if await request.is_disconnected():
                break
            yield chunk

    headers = {
        "Content-Disposition": f'inline; filename="{src.stem}_{start.replace(":","-")}_{end.replace(":","-")}.mp4"',
        "Cache-Control": "no-store",
    }
    return StreamingResponse(gen(), media_type="video/mp4", headers=headers)

@app.get("/stream_file")
def stream_file(
    background_tasks: BackgroundTasks,
    videoname: str = Query(...),
    start: str = Query(..., description="HH:MM:SS"),
    end: str = Query(..., description="HH:MM:SS"),
):
    src = validate_and_locate(videoname)
    s = parse_hms(start)
    e = parse_hms(end)
    if e <= s:
        raise HTTPException(status_code=400, detail="end must be after start")

    duration = e - s

    # Make a unique temp filename
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = pathlib.Path(tmp.name)
    tmp.close()

    cmd = [
        "ffmpeg",
        "-y",                       # <— allow overwrite of the temp path
        "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", start,
        "-t", str(duration),
        "-i", str(src),
        "-map", "0:v:0?",
        "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(tmp_path),
    ]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        with contextlib.suppress(Exception):
            tmp_path.unlink()
        raise HTTPException(status_code=500, detail="ffmpeg is not installed or not in PATH.")
    except subprocess.CalledProcessError:
        with contextlib.suppress(Exception):
            tmp_path.unlink()
        raise HTTPException(status_code=500, detail="Failed to create clip.")

    background_tasks.add_task(lambda p=tmp_path: p.unlink(missing_ok=True))

    filename = f'{src.stem}_{start.replace(":","-")}_{end.replace(":","-")}.mp4'
    headers = {
        "Content-Disposition": f'inline; filename="{filename}"',
        "Cache-Control": "no-store",
    }
    return FileResponse(path=str(tmp_path), media_type="video/mp4", filename=filename, headers=headers)

# A simple health check
@app.get("/healthz")
def healthz():
    return {"ok": True}
