import joblib

def verify_tree_watermark(model_path: str, expected_bits: list, leaf_indices: list):
    model = joblib.load(model_path)

    extracted_bits = []

    # -------- Decision Tree --------
    if hasattr(model, "tree_"):
        tree = model.tree_

        for idx in leaf_indices:
            value = tree.value[idx]
            bit = 1 if value.mean() >= 0 else 0
            extracted_bits.append(bit)

    # -------- Random Forest --------
    elif hasattr(model, "estimators_"):
        tree = model.estimators_[0].tree_

        for idx in leaf_indices:
            value = tree.value[idx]
            bit = 1 if value.mean() >= 0 else 0
            extracted_bits.append(bit)

    else:
        raise ValueError("Not a tree-based model")

    return {
        "expected_bits": expected_bits,
        "extracted_bits": extracted_bits,
        "match": extracted_bits == expected_bits
    }
