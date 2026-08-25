from PIL import Image, ImageChops
import numpy as np
import os


def run_ela(image_path: str, output_dir: str = "data/converted", quality: int = 90) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    original = Image.open(image_path).convert("RGB")
    temp_path = os.path.join(output_dir, f"{base_name}_resaved.jpg")
    original.save(temp_path, "JPEG", quality=quality)
    resaved = Image.open(temp_path)

    diff = ImageChops.difference(original, resaved)
    diff_array = np.array(diff).astype(np.float64)

    max_diff = diff_array.max()
    scale = 255.0 / max_diff if max_diff != 0 else 1.0
    amplified = (diff_array * scale).astype(np.uint8)

    ela_image = Image.fromarray(amplified)
    ela_output_path = os.path.join(output_dir, f"{base_name}_ela.png")
    ela_image.save(ela_output_path)

    mean_error = float(diff_array.mean())
    std_error = float(diff_array.std())
    p99 = float(np.percentile(diff_array, 99))

    # Spatial uniformity: ratio of std to mean.
    # High ratio = error is spread uniformly (typical JPEG artifact on clean docs).
    # Low ratio = error is concentrated in specific regions (possible edit).
    uniformity_ratio = std_error / mean_error if mean_error > 0 else 0.0

    # Anomaly strength: how far the tail extends beyond the bulk
    tail_extension = (p99 - mean_error) / std_error if std_error > 0 else 0.0

    # Confidence score (0.0 to 1.0):
    # - Uniform compression artifacts (high uniformity_ratio) → low confidence
    # - Localized spikes (low uniformity_ratio, high tail_extension) → high confidence
    # - Very low mean_error (clean image) → low confidence regardless
    if mean_error < 0.5:
        # Very clean image — ELA is unreliable here
        anomaly_confidence = 0.05
        spatial_type = "clean"
    elif uniformity_ratio > 3.0:
        # Error is spread uniformly — likely compression artifact, not edit
        anomaly_confidence = min(0.2, tail_extension / 30.0)
        spatial_type = "uniform"
    elif tail_extension > 4.0 and uniformity_ratio <= 3.0:
        # Localized spike with non-uniform distribution — more suspicious
        anomaly_confidence = min(0.9, 0.3 + (tail_extension - 4.0) / 10.0)
        spatial_type = "localized"
    else:
        # Moderate, inconclusive
        anomaly_confidence = 0.15
        spatial_type = "moderate"

    # Human-readable confidence label
    if anomaly_confidence >= 0.6:
        confidence_label = "HIGH"
    elif anomaly_confidence >= 0.3:
        confidence_label = "MODERATE"
    else:
        confidence_label = "LOW"

    summary = (
        f"ELA anomaly confidence: {confidence_label} ({anomaly_confidence:.2f}/1.0). "
        f"Spatial pattern: {spatial_type}. "
        f"Mean error: {mean_error:.2f}, std: {std_error:.2f}, p99: {p99:.1f}. "
        + (
            "Error is concentrated in specific regions — may indicate localized edits."
            if spatial_type == "localized" else
            "Error is spread uniformly — likely compression artifacts, not indicative of tampering."
            if spatial_type == "uniform" else
            "Error levels are very low — document appears clean."
            if spatial_type == "clean" else
            "Error pattern is inconclusive."
        )
    )

    findings = {
        "ela_image_path": ela_output_path,
        "mean_error": round(mean_error, 3),
        "std_error": round(std_error, 3),
        "p99_error": round(p99, 3),
        "anomaly_confidence": round(anomaly_confidence, 3),
        "confidence_label": confidence_label,
        "spatial_type": spatial_type,
        "summary": summary,
    }
    return findings


if __name__ == "__main__":
    test_image = "data/converted/Forgery-test_page1.png"
    result = run_ela(test_image)
    for k, v in result.items():
        print(f"  {k}: {v}")
