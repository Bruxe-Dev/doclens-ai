import pytesseract
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def run_ocr(image_path: str) -> dict:
    """
    Extracts raw text from a document image.
    Returns a dict with the text and basic metadata about the extraction.
    """
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)

    return {
        "text": text.strip(),
        "char_count": len(text.strip()),
        "line_count": len([l for l in text.split("\n") if l.strip()])
    }


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_image = os.path.join(script_dir, "..", "data", "converted", "Forgery-test_page1.png")
    result = run_ocr(test_image)
    print(f"Extracted {result['char_count']} characters, {result['line_count']} lines")
    print("--- TEXT ---")
    print(result["text"])