import os
from pathlib import Path
import pymupdf
from docx2pdf import convert as docx_to_pdf_convert

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
PDF_DPI = 200


def identify_document_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext == ".pdf":
        return "pdf"
    elif ext in (".docx", ".doc"):
        return "docx"
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def convert_pdf_to_images(pdf_path: str, output_dir: str) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    base_name = Path(pdf_path).stem
    image_paths = []

    zoom = PDF_DPI / 72
    matrix = pymupdf.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=matrix)
        out_path = os.path.join(output_dir, f"{base_name}_page{page_num + 1}.png")
        pix.save(out_path)
        image_paths.append(out_path)

    doc.close()
    return image_paths


def convert_docx_to_images(docx_path: str, output_dir: str) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, Path(docx_path).stem + ".pdf")
    docx_to_pdf_convert(docx_path, pdf_path)
    return convert_pdf_to_images(pdf_path, output_dir)


def prepare_document_for_model(file_path: str, output_dir: str = "data/converted") -> list[str]:
    doc_type = identify_document_type(file_path)
    print(f"Detected type: {doc_type} - {file_path}")

    if doc_type == "image":
        return [file_path]
    elif doc_type == "pdf":
        return convert_pdf_to_images(file_path, output_dir)
    elif doc_type == "docx":
        return convert_docx_to_images(file_path, output_dir)


if __name__ == "__main__":
    test_file = "data/test/Authentic.pdf"
    images = prepare_document_for_model(test_file)
    print("Ready-to-use image(s):", images)
