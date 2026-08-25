import re


REQUIRED_PATTERNS = {
    "date": r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
    "seller_field": r'\bSeller:?\s*[A-Z][a-z]+\s+[A-Z][a-z]+',
    "buyer_field": r'\bBuyer:?\s*[A-Z][a-z]+\s+[A-Z][a-z]+',
    "price": r'\bRWF\s*[\d,]+|\b[\d,]+\s*(Rwandan Francs|USD|Francs)',
    "area_measurement": r'\b\d+\s*square\s*meters?\b',
}


def run_deterministic_checks(ocr_text: str) -> dict:
    """
    Runs rule-based structural checks against OCR'd document text.
    Returns which expected fields were found/missing, with no AI involved —
    fully deterministic and reproducible.
    """
    findings = {}
    missing = []

    for field_name, pattern in REQUIRED_PATTERNS.items():
        match = re.search(pattern, ocr_text, re.IGNORECASE)
        found = match is not None
        findings[field_name] = {
            "found": found,
            "matched_text": match.group(0) if found else None
        }
        if not found:
            missing.append(field_name)

    completeness_score = (len(REQUIRED_PATTERNS) - len(missing)) / len(REQUIRED_PATTERNS)

    if completeness_score == 1.0:
        summary = "All expected structural fields were found in the document."
    elif completeness_score >= 0.6:
        summary = f"Most expected fields were found; missing: {', '.join(missing)}."
    else:
        summary = f"Multiple expected fields are missing: {', '.join(missing)}. This document may be incomplete or a non-standard format."

    return {
        "fields": findings,
        "missing_fields": missing,
        "completeness_score": round(completeness_score, 2),
        "summary": summary
    }


if __name__ == "__main__":
    from ocr_check import run_ocr

    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_image = os.path.join(script_dir, "..", "data", "converted", "Forgery-test_page1.png")
    ocr_result = run_ocr(test_image)
    result = run_deterministic_checks(ocr_result["text"])

    print("--- Deterministic Validation ---")
    for field, info in result["fields"].items():
        status = "✓" if info["found"] else "✗"
        print(f"{status} {field}: {info['matched_text']}")
    print(f"\nCompleteness score: {result['completeness_score']}")
    print(f"Summary: {result['summary']}")