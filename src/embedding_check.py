import requests
import numpy as np
import os 

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "embeddinggemma:300m"


def get_embedding(text: str) -> np.ndarray:
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "input": text}
    )
    response.raise_for_status()
    embedding = response.json()["embeddings"][0]
    return np.array(embedding)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def compute_embedding_similarity(submitted_text: str, reference_text: str) -> dict:
    submitted_vec = get_embedding(submitted_text)
    reference_vec = get_embedding(reference_text)
    similarity = cosine_similarity(submitted_vec, reference_vec)

    if similarity >= 0.85:
        interpretation = "High similarity to the authentic reference document."
    elif similarity >= 0.6:
        interpretation = "Moderate similarity — some content/structure differs from the reference."
    else:
        interpretation = "Low similarity — this document's content diverges significantly from the authentic reference."

    return {
        "similarity_score": round(similarity, 3),
        "interpretation": interpretation
    }


if __name__ == "__main__":
    from ocr_check import run_ocr

    submitted_image = "../data/converted/Forgery-test_page1.png"
    reference_image = "../data/converted/Forgery-test_page1.png"  

    submitted_text = run_ocr(submitted_image)["text"]
    reference_text = run_ocr(reference_image)["text"]

    result = compute_embedding_similarity(submitted_text, reference_text)
    print("--- Embedding Cross-Check ---")
    print(f"Similarity score: {result['similarity_score']}")
    print(f"Interpretation: {result['interpretation']}")