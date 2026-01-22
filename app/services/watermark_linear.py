import joblib
import numpy as np
import uuid

def watermark_linear_model(model_path: str):
    """
    Embeds a watermark into a linear model by modifying coefficient signs
    """

    model = joblib.load(model_path)

    if not hasattr(model, "coef_"):
        raise ValueError("Not a linear model")

    coef = model.coef_.copy()

    # Generate watermark
    watermark_id = str(uuid.uuid4())
    watermark_bits = np.random.randint(0, 2, size=min(8, coef.size))

    # Embed watermark in first N coefficients
    for i, bit in enumerate(watermark_bits):
        if bit == 1:
            coef.flat[i] = abs(coef.flat[i]) + 1e-4
        else:
            coef.flat[i] = -abs(coef.flat[i]) - 1e-4

    model.coef_ = coef

    # Save watermarked model
    output_path = model_path.replace(
        ".pkl", "_watermarked.pkl"
    )

    joblib.dump(model, output_path)

    return {
        "watermark_id": watermark_id,
        "watermark_bits": watermark_bits.tolist(),
        "watermarked_model_path": output_path
    }
