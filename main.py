from fastapi import FastAPI
from app.routes import upload, verify
from fastapi.middleware.cors import CORSMiddleware
from app.routes import revert

app = FastAPI(title="Secure Model Watermarking API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React
        "http://localhost:4200",   # Angular
        "http://127.0.0.1:5500",   # HTML Live Server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(verify.router)
app.include_router(revert.router)

@app.get("/")
def root():
    return {"message": "Backend running successfully"}
