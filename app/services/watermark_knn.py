import joblib
import numpy as np
import uuid

def watermark_knn_model(model_path: str):
    model = joblib.load(model_path)

    if not hasattr(model, "_fit_X"):
        raise ValueError("Not a KNN model")

   
    watermark_bits = np.random.randint(0, 2, size=4).tolist()
    sample_indices = list(range(len(watermark_bits)))

    
    model._watermark_bits = watermark_bits
    model._watermark_indices = sample_indices

    output_path = model_path.replace(".pkl", "_watermarked.pkl")
    joblib.dump(model, output_path)

    return {
        "watermark_id": str(uuid.uuid4()),
        "watermark_bits": watermark_bits,
        "sample_indices": sample_indices,
        "watermarked_model_path": output_path
    }
