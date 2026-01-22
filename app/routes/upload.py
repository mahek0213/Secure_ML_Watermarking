from fastapi import APIRouter, UploadFile, File
import os
import shutil

from app.utils.model_detector import detect_model_type
from app.services.watermark_linear import watermark_linear_model
from app.services.watermark_tree import watermark_tree_model
from app.services.watermark_knn import watermark_knn_model

router = APIRouter()

UPLOAD_DIR = os.path.join(os.getcwd(), "storage", "uploaded_models")

@router.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        model_type = detect_model_type(file_path)

        response = {
            "message": "Model uploaded successfully",
            "filename": file.filename,
            "detected_model_type": model_type,
            "original_model_path": file_path
        }

        if model_type == "linear":
            response["watermark"] = watermark_linear_model(file_path)

        elif model_type == "tree":
            response["watermark"] = watermark_tree_model(file_path)

        elif model_type == "knn":
            response["watermark"] = watermark_knn_model(file_path)

        else:
            response["note"] = "Watermarking not supported for this model type"

        return response

    except Exception as e:
        return {
            "error": "Upload or watermarking failed",
            "details": str(e)
        }
