from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os

router = APIRouter()

# Folder where original uploaded models are stored
UPLOAD_DIR = os.path.join(os.getcwd(), "storage", "uploaded_models")


@router.get("/download-original-model")
def download_original_model(filename: str):
    """
    Returns original uploaded model file.
    Used to prove watermarking did not change base model.
    """

    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Original model not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )
