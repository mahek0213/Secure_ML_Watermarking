import joblib
import numpy as np
import uuid

def watermark_tree_model(model_path: str):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Model loading failed: {e}")

    if hasattr(model, "tree_"):
        tree = model.tree_

    elif hasattr(model, "estimators_") and len(model.estimators_) > 0:
        tree = model.estimators_[0].tree_

    else:
        raise ValueError("Model is not tree-based (DecisionTree/RandomForest)")

    leaf_indices = np.where(tree.children_left == -1)[0]

    if len(leaf_indices) == 0:
        raise ValueError("Tree has no leaf nodes")

    leaf_indices = sorted(leaf_indices.tolist())

    watermark_len = min(4, len(leaf_indices))

    watermark_bits = []
    used_leaf_indices = []

    for i in range(watermark_len):
        leaf = leaf_indices[i]
        bit = np.random.randint(0, 2)

        watermark_bits.append(bit)
        used_leaf_indices.append(leaf)

        value = tree.value[leaf]

        if bit == 1:
            tree.value[leaf] = np.abs(value)
        else:
            tree.value[leaf] = -np.abs(value)

    output_path = model_path.replace(".pkl", "_watermarked.pkl")
    joblib.dump(model, output_path)

    return {
        "watermark_id": str(uuid.uuid4()),
        "watermark_bits": watermark_bits,
        "leaf_indices": used_leaf_indices,
        "watermarked_model_path": output_path
    }
