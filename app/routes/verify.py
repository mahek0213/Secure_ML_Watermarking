from fastapi import APIRouter, UploadFile, File, Form
import os
import shutil
import json

from app.utils.model_detector import detect_model_type
from app.services.verify_linear import verify_linear_watermark
from app.services.verify_tree import verify_tree_watermark
from app.services.verify_knn import verify_knn_watermark

router = APIRouter()

VERIFY_DIR = os.path.join(os.getcwd(), "storage", "verify_models")

def parse_list(input_str: str):
    input_str = input_str.strip()
    try:
        return json.loads(input_str)
    except Exception:
        input_str = input_str.replace("[", "").replace("]", "")
        return [int(x.strip()) for x in input_str.split(",") if x.strip()]

@router.post("/verify-model")
async def verify_model(
    file: UploadFile = File(...),
    expected_bits: str = Form(...),
    leaf_indices: str = Form(None),
    sample_indices: str = Form(None)
):
    try:
        os.makedirs(VERIFY_DIR, exist_ok=True)

        file_path = os.path.join(VERIFY_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        model_type = detect_model_type(file_path)
        expected_bits_list = parse_list(expected_bits)

        if model_type == "linear":
            result = verify_linear_watermark(file_path, expected_bits_list)

        elif model_type == "tree":
            if not leaf_indices:
                return {"error": "leaf_indices required for tree verification"}

            leaf_indices_list = parse_list(leaf_indices)
            result = verify_tree_watermark(
                file_path,
                expected_bits_list,
                leaf_indices_list
            )

        elif model_type == "knn":
            if not sample_indices:
                return {"error": "sample_indices required for KNN verification"}

            sample_indices_list = parse_list(sample_indices)
            result = verify_knn_watermark(
                file_path,
                expected_bits_list,
                sample_indices_list
            )

        else:
            return {"error": "Unsupported model type"}

        return {
            "detected_model_type": model_type,
            "verification_result": result
        }

    except Exception as e:
        return {
            "error": "Verification failed",
            "details": str(e)
        }
