from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ChestTriage API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # temporary, we'll restrict this later
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}