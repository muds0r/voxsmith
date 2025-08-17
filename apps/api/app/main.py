# apps/api/app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from mutagen import File as MutagenFile
import os, tempfile, uuid
import subprocess, contextlib, wave
from openai import OpenAI
import textwrap
from typing import Optional


VOX_NOTES_ENABLED = os.getenv("VOX_NOTES_ENABLED", "true").lower() == "true"
VOX_NOTES_MODEL = os.getenv("VOX_NOTES_MODEL", "gpt-4o-mini")


# load .env from repo root
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Voxsmith API", version="0.1.0")

# ---------- Config ----------
VOX_STT_ENABLED = os.getenv("VOX_STT_ENABLED", "true").lower() == "true"
VOX_CLASSIFY_ENABLED = os.getenv("VOX_CLASSIFY_ENABLED", "true").lower() == "true"
VOX_CLASSIFY_MODEL = os.getenv("VOX_CLASSIFY_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
MAX_UPLOAD_MB = int(os.getenv("VOX_MAX_UPLOAD_MB", "60"))
ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".mp4", ".opus", ".ogg"}
FFPROBE = os.getenv("VOX_FFPROBE", "ffprobe")
FFMPEG = os.getenv("VOX_FFMPEG", "ffmpeg")   # <— hinzufügen



# ---------- Helpers ----------
def _should_run_stt() -> bool:
    return VOX_STT_ENABLED and bool(OPENAI_API_KEY)

def _pre_classify_rule(transcript: str) -> str | None:
    t = (transcript or "").strip()
    if not t:
        return None
    # sehr kurzer, deklarativer Einzeiler -> eher "Presentation / Speech"
    if len(t.split()) <= 12 and t[-1:] in ".?!":
        return "Presentation / Speech"
    return None

def _stt_ready_path(src_path: str) -> tuple[str, list[str], str | None]:
    """
    Returns (path_for_stt, temps_to_delete, error_note).
    Transcodes .opus/.ogg -> WAV (mono,16k). Otherwise returns original.
    """
    ext = os.path.splitext(src_path)[1].lower()
    temps: list[str] = []
    if ext in {".opus", ".ogg"}:
        out_wav = os.path.join(tempfile.gettempdir(), f"vox_{uuid.uuid4().hex}.wav")
        try:
            # -y overwrite, -ac 1 mono, -ar 16000 16kHz (Whisper-freundlich)
            proc = subprocess.run(
                [FFMPEG, "-y", "-i", src_path, "-ac", "1", "-ar", "16000", out_wav],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if proc.returncode != 0 or not os.path.exists(out_wav):
                return src_path, temps, f"ffmpeg transcode failed (code {proc.returncode}): {proc.stderr[:200]}"
            temps.append(out_wav)
            return out_wav, temps, None
        except Exception as e:
            return src_path, temps, f"ffmpeg exception: {e.__class__.__name__}: {str(e)[:200]}"
    return src_path, temps, None


def _transcribe(path: str, model: str = "whisper-1") -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing")
    client = OpenAI(api_key=OPENAI_API_KEY)
    with open(path, "rb") as f:
        r = client.audio.transcriptions.create(model=model, file=f)
    return getattr(r, "text", "").strip()

def _get_duration_ffprobe(path: str) -> int | None:
    try:
        # ffprobe gibt Sekunden als Float aus
        cmd = [FFPROBE, "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", path]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        dur = float(out.decode().strip())
        if dur > 0:
            return int(round(dur))
    except Exception:
        return None
    return None

def _analyze_notes(transcript: str) -> str:
    if not VOX_NOTES_ENABLED:
        return "Notes disabled via VOX_NOTES_ENABLED=false."
    if not OPENAI_API_KEY:
        return "Notes unavailable: missing OPENAI_API_KEY."
    transcript = (transcript or "").strip()
    if not transcript:
        return "Notes unavailable: empty transcript."

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "You are Voxsmith, a concise conversation coach.\n"
        "Analyze the transcript and return:\n"
        "- 2–3 sentence feedback (strengths + one improvement)\n"
        "- 3 short bullet To-Dos (imperative, concrete)\n\n"
        f"Transcript:\n{transcript}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=VOX_NOTES_MODEL,
            messages=[
                {"role": "system", "content": "Be concise, practical, and non-fluffy."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=250,
        )
        content = (resp.choices[0].message.content or "").strip()
        print("DEBUG NOTES RAW:", repr(content))   # <--- hier
        return content if content else "Notes generation returned empty content."
    except Exception as e:
        print("DEBUG NOTES ERROR:", e)            # <--- hier
        return f"Notes generation failed: {e.__class__.__name__}: {str(e)[:200]}"



def _get_duration_sec(path: str) -> int | None:
    # 1) Mutagen versuchen
    try:
        m = MutagenFile(path)
        if m and getattr(m, "info", None):
            dur = getattr(m.info, "length", None)
            if dur:
                return int(round(dur))
    except Exception:
        pass
    # 2) WAV-Fallback (PCM)
    try:
        if os.path.splitext(path)[1].lower() == ".wav":
            with contextlib.closing(wave.open(path, "rb")) as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return int(round(frames / float(rate)))
    except Exception:
        pass
    # 3) ffprobe als universeller Fallback (holt auch MP4)
    return _get_duration_ffprobe(path)

def _classify_conversation(transcript: str) -> dict:
    """
    Returns dict with keys: type (str), confidence (0-1), why (short string).
    """
    if not OPENAI_API_KEY or not transcript.strip():
        return {}

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""
    Classify the conversation transcript into ONE of the following categories:
    - sales_call
    - job_interview
    - coaching_session
    - meeting
    - voice_memo
    Return JSON with:
    {{
      "type": "<category>",
      "confidence": <float 0-1>,
      "why": "<short reason in 1 sentence>"
    }}

    Transcript:
    {transcript}
    """

    try:
        resp = client.responses.create(
            model=VOX_CLASSIFY_MODEL,
            input=[{"role":"user","content":prompt}],
            response_format={"type":"json_object"},
            max_output_tokens=200,
            temperature=0.1,
        )
        return json.loads(resp.output[0].content[0].text)
    except Exception as e:
        return {"type":"unknown","confidence":0.0,"why":f"failed: {e.__class__.__name__}"}





# ---------- Models ----------
class AnalyzeResponse(BaseModel):
    file_name: str
    bytes: int
    duration_sec: int | None = None  # TODO: ffprobe später
    transcript: str | None = None
    notes: str | None = None
    classification: Optional[str] = None

# ---------- Routes ----------
@app.get("/config")
def cfg():
    return {
        "VOX_STT_ENABLED": VOX_STT_ENABLED,
        "VOX_NOTES_ENABLED": VOX_NOTES_ENABLED,
        "VOX_NOTES_MODEL": VOX_NOTES_MODEL,
        "VOX_CLASSIFY_ENABLED": VOX_CLASSIFY_ENABLED,
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "voxsmith-api", "version": "0.1.0"}

@app.get("/")
def root():
    return {"service": "voxsmith-api", "status": "ok", "endpoints": ["/health", "/analyze"]}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    print("DEBUG upload bytes:", len(data))
    notes = None
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix and suffix not in ALLOWED_EXT:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}")

    tmp_path = os.path.join(tempfile.gettempdir(), f"vox_{uuid.uuid4().hex}{suffix}")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_MB} MB)")

    try:
        with open(tmp_path, "wb") as out:
            out.write(data)


        # Dauer bestimmen
        duration_sec = _get_duration_sec(tmp_path)

        # OPUS/OGG -> WAV (falls nötig)
        stt_path, extra_temps, stt_prep_err = _stt_ready_path(tmp_path)
        if stt_prep_err:
            notes = f"{(notes + '\\n') if notes else ''}Prep: {stt_prep_err}"
        print("DEBUG stt_path:", stt_path)

        transcript = None
        classification = None

        if _should_run_stt():
            try:
                transcript = _transcribe(stt_path)   # <— WICHTIG: stt_path, nicht tmp_path!
            except Exception as e:
                notes = f"{(notes + '\\n') if notes else ''}STT failed or unavailable: {e.__class__.__name__}: {str(e)}"


        # Notes-Analyse nur, wenn Transcript vorhanden
        if VOX_NOTES_ENABLED and transcript:
            ai_notes = _analyze_notes(transcript)
            print("DEBUG ai_notes:", repr(ai_notes))   # <--- hier
            if ai_notes and ai_notes.strip():
                notes = f"{(notes + '\\n\\n') if notes else ''}{ai_notes.strip()}"

    
            # --- Classification ---
    
        if VOX_CLASSIFY_ENABLED and transcript:
            # 1) Heuristik zuerst
            label = _pre_classify_rule(transcript)
            if label:
                classification = label
            else:
                # 2) Few-shot Prompt
                try:
                    fewshots = (
                        'Transcript: "So what budget are you working with this quarter?" -> Sales Call\n'
                        'Transcript: "Welcome to the show, today my guest is..." -> Interview\n'
                        'Transcript: "Note to self: call the landlord tomorrow." -> Voice Memo\n'
                        "Transcript: \"Let's list action items: Anna owns landing page by Friday.\" -> Meeting\n"
                    )

                    classify_prompt = f"""You must output ONLY one of:
                        - Sales Call
                        - Interview
                        - Meeting
                        - Voice Memo
                        - Presentation / Speech
                        - Casual Conversation
                        - Other

                        {fewshots}
                        Now classify:
                        {transcript}

                        Answer with the label only."""
                    ai_class = client.chat.completions.create(
                        model=VOX_CLASSIFY_MODEL,  # z.B. gpt-4o-mini
                        messages=[
                            {"role": "system", "content": "You are a strict classifier. Output only the label."},
                            {"role": "user", "content": classify_prompt},
                        ],
                        temperature=0.0,
                        max_tokens=10,
                    )
                    classification = (ai_class.choices[0].message.content or "").strip()
                    print("DEBUG classification:", classification)
                except Exception as e:
                    classification = f"Classification failed: {e.__class__.__name__}: {str(e)[:120]}"



            
        return AnalyzeResponse(
            file_name=file.filename,
            bytes=len(data),
            duration_sec=duration_sec,
            transcript=transcript,
            notes=notes,
            classification= classification,
        )
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass
        for p in extra_temps:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except:
                pass
