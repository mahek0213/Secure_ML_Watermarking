import joblib
import numpy as np

def verify_linear_watermark(model_path: str, expected_bits: list):
    model = joblib.load(model_path)

    if not hasattr(model, "coef_"):
        raise ValueError("Model has no coefficients")

    coef = model.coef_.flatten()

    if len(coef) < len(expected_bits):
        raise ValueError("Expected bits length is larger than coefficients")

    extracted_bits = []

    for i in range(len(expected_bits)):
        extracted_bits.append(1 if coef[i] >= 0 else 0)

    return {
        "expected_bits": expected_bits,
        "extracted_bits": extracted_bits,
        "match": extracted_bits == expected_bits
    }
