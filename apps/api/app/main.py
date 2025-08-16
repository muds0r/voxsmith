from fastapi import FastAPI

app = FastAPI(title="Voxsmith API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok", "service": "voxsmith-api", "version": "0.1.0"}