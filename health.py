try:
    from fastapi import FastAPI
except Exception:
    FastAPI = None  # Optional dependency

app = FastAPI() if FastAPI else None

if app:
    @app.get("/health")
    def health():
        return {"status": "ok"}
