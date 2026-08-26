import sys
import os
import json
import time
import base64
import io
import requests
from PIL import Image
import re

from document_preprocessing import prepare_document_for_model
from concurrent.futures import ThreadPoolExecutor
from ocr_check import run_ocr
from deterministic_check import run_deterministic_checks
from embedding_check import compute_embedding_similarity
from ela_check import run_ela

OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_IMAGE_WIDTH = 768
STREAM_TIMEOUT = 600


def extract_risk_level(report_text: str) -> str:
    """Pull out the labeled risk level line, defaulting to 'Unknown' if not found."""
    match = re.search(r'Risk Level:\s*(Low|Medium|High)', report_text, re.IGNORECASE)
    return match.group(1).capitalize() if match else "Unknown"


def extract_reason(report_text: str) -> str:
    """Pull out the labeled justification, defaulting to a fallback message."""
    match = re.search(r'Justification:\s*(.+?)(?:\n\n|\Z)', report_text, re.DOTALL)
    return match.group(1).strip() if match else "See full report for details."


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def resize_for_model(image_path: str, max_width: int = MAX_IMAGE_WIDTH) -> str:
    img = Image.open(image_path)
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


FORENSIC_PROMPT_TEMPLATE = """You are a document forensic analyst. Your job is to help a human reviewer decide whether this document needs deeper investigation.

**Automated compression analysis (ELA):**
- Anomaly confidence: {anomaly_confidence} ({confidence_label})
- Spatial pattern: {spatial_type}
- {ela_summary}

**Deterministic structural validation:**
- Completeness score: {completeness_score} (fields found / fields expected)
- {validation_summary}

**Embedding similarity to a known-authentic reference document:**
- Similarity score: {similarity_score}
- {similarity_interpretation}

**Important context:** ELA measures JPEG compression artifacts and frequently produces false positives on digitally-native (typed/printed) documents — weight it accordingly. The structural validation and similarity scores are deterministic/reproducible and generally more reliable indicators of completeness and content-pattern match than ELA alone.

**Your task — examine the document image and provide:**

1. **Document Type**: What kind of document is this?

2. **Visual Findings**: List each observation as a separate bullet — what you observed, where, and whether it's normal or suspicious.

3. **Synthesis**: Weigh the ELA signal, the structural validation result, and the embedding similarity together with your own visual inspection. Which pieces of evidence agree, which conflict, and how does that affect your confidence?

4. **Risk Level**: State exactly one word — Low, Medium, or High — on its own line, formatted exactly as: "Risk Level: <word>"

5. **Justification**: A short 2-3 sentence explanation of WHY, referencing specific findings above. Start with exactly: "Justification:"

Be precise. Every conclusion must trace back to a specific observation."""

FALLBACK_REPORT_TEMPLATE = """DocLens Forensic Analysis Report
================================

Document: {filename}
Analysis Date: {timestamp}

--- Automated ELA Results ---
Anomaly Confidence: {anomaly_confidence} ({confidence_label})
Spatial Pattern: {spatial_type}
Mean Error: {mean_error} | Std: {std_error} | P99: {p99_error}

--- Assessment ---
{ela_summary}

Risk Level: {risk_level}

Justification: {reason}

--- Note ---
The AI vision model was unavailable. This report contains only automated ELA analysis.
A full forensic assessment requires visual inspection by the AI model or a human reviewer."""

def run_pipeline_on_image(image_path: str, reference_image_path: str = None, model: str = "gemma3:4b") -> dict:
    if reference_image_path is None:
        reference_image_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "converted", "Authentic_page1.png"
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        ela_future = executor.submit(run_ela, image_path)
        ocr_future = executor.submit(run_ocr, image_path)
        ref_ocr_future = executor.submit(run_ocr, reference_image_path)

        ela_result = ela_future.result()
        ocr_result = ocr_future.result()
        reference_text = ref_ocr_future.result()["text"]

    # these depend on OCR text, so they run after OCR resolves,
    # but OCR itself already ran in parallel with ELA above
    validation_result = run_deterministic_checks(ocr_result["text"])
    similarity_result = compute_embedding_similarity(ocr_result["text"], reference_text)

    image_b64 = resize_for_model(image_path)
    prompt = FORENSIC_PROMPT_TEMPLATE.format(
        anomaly_confidence=f"{ela_result['anomaly_confidence']:.2f}",
        confidence_label=ela_result["confidence_label"],
        spatial_type=ela_result["spatial_type"],
        ela_summary=ela_result["summary"],
        completeness_score=validation_result["completeness_score"],
        validation_summary=validation_result["summary"],
        similarity_score=similarity_result["similarity_score"],
        similarity_interpretation=similarity_result["interpretation"],
    )

    print("Calling Gemma 3 4B via Ollama...")
    start = time.time()
    try:
        report = call_ollama(prompt, image_b64, model)
        print(f"  Model responded in {time.time() - start:.1f}s")
    except requests.exceptions.RequestException as e:
        print(f"  Ollama unavailable: {e}")
        report = build_fallback_report(ela_result, image_path)

    return {
        "report": report,
        "risk_level": extract_risk_level(report),
        "reason": extract_reason(report),
        "ela": ela_result,
        "validation": validation_result,
        "similarity": similarity_result,
    }
    
def call_ollama(prompt: str, image_b64: str, model: str = "gemma3:4b") -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": True,
        },
        timeout=STREAM_TIMEOUT,
    )
    response.raise_for_status()

    full_response = ""
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            if "response" in chunk:
                full_response += chunk["response"]
    return full_response


def build_fallback_report(ela_result: dict, image_path: str) -> str:
    filename = os.path.basename(image_path)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    confidence = ela_result["anomaly_confidence"]
    if confidence >= 0.6:
        risk = "High"
        reason = "Based on ELA only — high anomaly confidence detected, requires visual confirmation by a human reviewer since the AI model was unavailable."
    elif confidence >= 0.3:
        risk = "Medium"
        reason = "ELA results are inconclusive — requires visual confirmation by a human reviewer since the AI model was unavailable."
    else:
        risk = "Low"
        reason = "ELA shows no significant compression anomaly. Note: this is an automated-only assessment since the AI model was unavailable."

    return FALLBACK_REPORT_TEMPLATE.format(
        filename=filename,
        timestamp=timestamp,
        anomaly_confidence=f"{confidence:.2f}",
        confidence_label=ela_result["confidence_label"],
        spatial_type=ela_result["spatial_type"],
        mean_error=ela_result["mean_error"],
        std_error=ela_result["std_error"],
        p99_error=ela_result["p99_error"],
        ela_summary=ela_result["summary"],
        risk_level=risk,
        reason=reason,
    )


def run_full_pipeline(file_path: str, model: str = "gemma3:4b") -> str:
    # Step 1: Preprocess document to images
    print("Step 1: Preprocessing document...")
    image_paths = prepare_document_for_model(file_path)
    print(f"  Converted to {len(image_paths)} image(s).")

    results = []
    for idx, image_path in enumerate(image_paths):
        print(f"\n--- Analyzing page {idx + 1}/{len(image_paths)}: {os.path.basename(image_path)} ---")

        # Step 2: ELA analysis
        print("Step 2: Running ELA...")
        ela_result = run_ela(image_path)
        print(f"  Anomaly confidence: {ela_result['anomaly_confidence']:.2f} ({ela_result['confidence_label']})")
        print(f"  Spatial type: {ela_result['spatial_type']}")

        # Step 3: Resize image for model
        print("Step 3: Preparing image for model...")
        image_b64 = resize_for_model(image_path)

        # Step 4: Build prompt with ELA context
        prompt = FORENSIC_PROMPT_TEMPLATE.format(
            anomaly_confidence=f"{ela_result['anomaly_confidence']:.2f}",
            confidence_label=ela_result["confidence_label"],
            spatial_type=ela_result["spatial_type"],
            mean_error=ela_result["mean_error"],
            std_error=ela_result["std_error"],
            p99_error=ela_result["p99_error"],
            ela_summary=ela_result["summary"],
        )

        # Step 5: Call vision model
        print("Step 4: Calling Gemma 3 4B via Ollama (streaming)...")
        start = time.time()
        try:
            report = call_ollama(prompt, image_b64, model)
            elapsed = time.time() - start
            print(f"  Model responded in {elapsed:.1f}s")
        except requests.exceptions.RequestException as e:
            elapsed = time.time() - start
            print(f"  Ollama unavailable after {elapsed:.1f}s: {e}")
            print("  Generating fallback report from ELA data...")
            report = build_fallback_report(ela_result, image_path)

        page_result = {
            "page": idx + 1,
            "image_path": image_path,
            "ela": ela_result,
            "report": report,
            "risk_level": extract_risk_level(report),
            "reason": extract_reason(report),
        }
        results.append(page_result)

    # Step 6: Assemble final output
    if len(results) == 1:
        return results[0]["report"]

    # Multi-page: combine reports
    # NOTE: for gRPC responses, use results[0]["risk_level"] / results[0]["reason"]
    # as the top-level fields — a single risk_level can't represent multiple pages.
    # This is a known scope gap for multi-page documents; flagged for Phase 2.
    combined = []
    for r in results:
        combined.append(f"{'='*60}")
        combined.append(f"PAGE {r['page']}: {os.path.basename(r['image_path'])}")
        combined.append(f"{'='*60}")
        combined.append(r["report"])
        combined.append("")
    return "\n".join(combined)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = os.path.join(os.path.dirname(__file__), "..", "data", "test", "Forgery-test.pdf")

    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    print(f"DocLens Pipeline — Analyzing: {input_path}\n")
    report = run_full_pipeline(input_path)
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(report)
    print("\n" + "=" * 60)
    print(f"Extracted Risk Level: {extract_risk_level(report)}")
    print(f"Extracted Reason: {extract_reason(report)}")