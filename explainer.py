import config
from core.classifier import rf_confidence


def get_feature_importance(model):
    if model is None or not hasattr(model, "feature_importances_"):
        return {}
    return {
        name: float(value)
        for name, value in zip(config.FEATURE_NAMES, model.feature_importances_)
    }


def explain_cell(cell_features, prediction, ground_truth, crop_type, model=None):
    ndvi, red, green, texture, moisture = cell_features
    thresholds = config.NDVI_THRESHOLDS.get(crop_type, config.NDVI_THRESHOLDS["Wheat"])
    confidence = rf_confidence(model, cell_features)

    lines = [
        f"Prediction: {config.CLASS_LABELS[prediction]}",
        f"Ground truth: {config.CLASS_LABELS[ground_truth]}",
        f"NDVI: {ndvi:.3f} (healthy ≥ {thresholds['healthy']:.2f}, severe < {thresholds['severe']:.2f})",
        f"Red intensity: {red:.3f} — higher values suggest stress",
        f"Green intensity: {green:.3f}",
        f"Texture variance: {texture:.3f}",
        f"Moisture index: {moisture:.3f}",
    ]

    if ndvi >= thresholds["healthy"]:
        lines.append("NDVI rule: classified as Healthy based on crop threshold.")
    elif ndvi >= thresholds["severe"]:
        lines.append("NDVI rule: classified as Early Disease due to reduced vegetation index.")
    else:
        lines.append("NDVI rule: classified as Severe Disease due to very low NDVI.")

    if confidence is not None:
        lines.append(f"Random Forest confidence: {confidence * 100:.1f}%")

    if prediction == ground_truth:
        lines.append("Result matches simulated ground truth.")
    else:
        lines.append("Result differs from ground truth — review features above.")

    return "\n".join(lines)


def explain_path_choice(path_comparison):
    boustro = path_comparison["Boustrophedon"]
    rw = path_comparison["Random Walk"]
    return (
        f"Boustrophedon coverage: {boustro['coverage']:.1f}% with {boustro['gaps']} gaps "
        f"and {boustro['astar_nodes']} A* nodes explored.\n"
        f"Random Walk coverage: {rw['coverage']:.1f}% with {rw['gaps']} gaps.\n"
        "Systematic Boustrophedon scanning is preferred when full field coverage is required."
    )
