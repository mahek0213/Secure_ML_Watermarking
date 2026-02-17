from fastapi import APIRouter, UploadFile, File, Form
import os
import shutil
import json

from app.utils.model_detector import detect_model_type
from app.services.verify_linear import verify_linear_watermark
from app.services.verify_tree import verify_tree_watermark
from app.services.verify_knn import verify_knn_watermark

router = APIRouter()

# Directory to store uploaded models for verification
VERIFY_DIR = os.path.join(os.getcwd(), "storage", "verify_models")


# ✅ Improved List Parser Function
def parse_list(input_str: str):
    """
    Converts input string into a list of integers.

    Supports formats:
    - "[1,2,3]"
    - "1,2,3"

    Prevents invalid inputs like "string"
    """

    if not input_str:
        return []

    input_str = input_str.strip()

    # ❌ Prevent Swagger placeholder mistake
    if input_str.lower() == "string":
        raise ValueError(
            "Invalid input: Please provide a list like [1,2,3] instead of 'string'"
        )

    # Try JSON format first
    try:
        data = json.loads(input_str)

        # Ensure it's a list
        if isinstance(data, list):
            return [int(x) for x in data]

        raise ValueError("Input must be a list of integers")

    except Exception:
        # Fallback: comma-separated format
        input_str = input_str.replace("[", "").replace("]", "")
        return [int(x.strip()) for x in input_str.split(",") if x.strip()]


# ✅ Verify Model Endpoint
@router.post("/verify-model")
async def verify_model(
    file: UploadFile = File(...),
    expected_bits: str = Form(...),
    leaf_indices: str = Form(None),
    sample_indices: str = Form(None),
):
    try:
        # Create directory if not exists
        os.makedirs(VERIFY_DIR, exist_ok=True)

        # Save uploaded file
        file_path = os.path.join(VERIFY_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Detect model type automatically
        model_type = detect_model_type(file_path)

        # Parse expected watermark bits
        expected_bits_list = parse_list(expected_bits)

        # -------------------------------
        # Linear Model Verification
        # -------------------------------
        if model_type == "linear":
            result = verify_linear_watermark(file_path, expected_bits_list)

        # -------------------------------
        # Tree Model Verification
        # -------------------------------
        elif model_type == "tree":

            if not leaf_indices:
                return {
                    "error": "leaf_indices required for Tree verification",
                    "example": "[1,2,3]"
                }

            leaf_indices_list = parse_list(leaf_indices)

            result = verify_tree_watermark(
                file_path,
                expected_bits_list,
                leaf_indices_list
            )

        # -------------------------------
        # KNN Model Verification
        # -------------------------------
        elif model_type == "knn":

            if not sample_indices:
                return {
                    "error": "sample_indices required for KNN verification",
                    "example": "[1,2,3]"
                }

            sample_indices_list = parse_list(sample_indices)

            result = verify_knn_watermark(
                file_path,
                expected_bits_list,
                sample_indices_list
            )

        # -------------------------------
        # Unsupported Model Type
        # -------------------------------
        else:
            return {
                "error": "Unsupported model type detected",
                "detected_type": model_type
            }

        # ✅ Successful Response
        return {
            "detected_model_type": model_type,
            "expected_bits": expected_bits_list,
            "verification_result": result,
            "ownership_status": "Verified Ownership" if result.get("match") else "Ownership Not Verified"
        }

    # ❌ Handle Parsing Errors Clearly
    except ValueError as ve:
        return {
            "error": "Invalid Input Format",
            "details": str(ve),
            "hint": "Use format like [1,2,3] instead of 'string'"
        }

    # ❌ Handle Other Errors
    except Exception as e:
        return {
            "error": "Verification failed",
            "details": str(e)
        }
