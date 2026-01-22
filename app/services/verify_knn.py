import joblib

def verify_knn_watermark(model_path: str, expected_bits: list, sample_indices: list):
    model = joblib.load(model_path)

    if not hasattr(model, "_watermark_bits"):
        raise ValueError("No watermark found in KNN model")

    extracted_bits = model._watermark_bits

    return {
        "expected_bits": expected_bits,
        "extracted_bits": extracted_bits,
        "match": extracted_bits == expected_bits
    }
