import requests
import base64
from ela_check import run_ela

FORENSIC_PROMPT_TEMPLATE = """You are a document forensic analyst. You are reviewing a document image
for signs of forgery, tampering, or inconsistency, to help a human reviewer decide whether
this document needs deeper investigation.

Automated compression analysis (Error Level Analysis) results:
- Mean error level: {mean_error}
- 99th percentile error level: {p99_error}
- Anomaly flagged by automated check: {flagged}
- Automated summary: {ela_summary}

Now examine the document image itself and provide:
1. A brief description of what type of document this appears to be.
2. Any visual inconsistencies you notice (font mismatches, alignment issues, signature
   irregularities, unnatural edges around any element, formatting inconsistencies).
3. Your interpretation of the automated compression analysis result above, in context —
   is it meaningful for this type of document, or likely a limitation of the technique?
4. An overall risk level: Low, Medium, or High — with a brief justification.

Keep the report professional, concise, and clearly organized under these four headings.
"""


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_full_pipeline(image_path: str, model: str = "gemma3:4b") -> str:
    print("Step 1: running ELA...")
    ela_result = run_ela(image_path)
    print("ELA result:", ela_result)

    print("Step 2: building prompt...")
    prompt = FORENSIC_PROMPT_TEMPLATE.format(
        mean_error=ela_result["mean_error"],
        p99_error=ela_result["p99_error"],
        flagged=ela_result["flagged"],
        ela_summary=ela_result["summary"]
    )

    print("Step 3: calling Gemma via ollama...")
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": [encode_image(image_path)],
            "stream": False
        }
    )
    response.raise_for_status()
    report = response.json()["response"]
    return report


if __name__ == "__main__":
    image = "../data/converted/Forgery-test_page1.png"
    print("Running pipeline on:", image)
    report = run_full_pipeline(image)
    print("\n=== FINAL REPORT ===\n")
    print(report)