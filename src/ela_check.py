from PIL import Image, ImageChops
import numpy as np
import os

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

    # A region is "anomalous" if its error is far outside the overall spread,
    # not just far from a possibly-tiny mean.
    flagged = (p99 - mean_error) > (5 * std_error) and std_error > 0.5

    findings = {
        "ela_image_path": ela_output_path,
        "mean_error": round(mean_error, 3),
        "std_error": round(std_error, 3),
        "p99_error": round(p99, 3),
        "flagged": flagged,
        "summary": (
            f"ELA found localized high-error pixels (99th percentile {p99:.1f}) "
            f"well outside the image's typical error spread (mean {mean_error:.2f}, "
            f"std {std_error:.2f}) — consistent with a possibly edited region."
            if flagged else
            f"ELA found error levels fairly uniform across the image "
            f"(mean {mean_error:.2f}, std {std_error:.2f}) — no strong localized anomaly."
        )
    }
    return findings


if __name__ == "__main__":
    test_image = "data/converted/Forgery-test_page1.png"
    result = run_ela(test_image)
    print(result)