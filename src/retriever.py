import faiss
import numpy as np

from pdf_processor import extract_text_from_pdf
from chunker import create_chunks
from embeddings import create_embeddings


def retrieve_chunks(question, chunks, index, top_k=3):
    """
    Retrieve the most relevant chunks for a question.
    """

    # Convert the question into an embedding
    question_embedding = create_embeddings([question])

    # Convert to FAISS-compatible format
    question_embedding = np.array(question_embedding).astype("float32")

    # Search the vector database
    distances, indices = index.search(question_embedding, top_k)

    results = []

    for distance, index_number in zip(distances[0], indices[0]):

        if index_number != -1:

            results.append({
                "text": chunks[index_number]["text"],
                "page_number": chunks[index_number]["page_number"],
                "distance": float(distance)
            })

    return results


if __name__ == "__main__":

    pdf_path = "data/papers/researchpaper.pdf"

    # Extract PDF text
    pages = extract_text_from_pdf(pdf_path)

    # Create chunks
    chunks = create_chunks(pages)

    # Create embeddings for all chunks
    chunk_embeddings = create_embeddings(
        [chunk["text"] for chunk in chunks]
    )

    chunk_embeddings = np.array(chunk_embeddings).astype("float32")

    # Create FAISS index
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(chunk_embeddings)

    # Test question
    question = "What is the Transformer architecture?"

    print("\nQuestion:")
    print(question)

    # Retrieve relevant chunks
    results = retrieve_chunks(
        question,
        chunks,
        index,
        top_k=3
    )

    print("\nRetrieved chunks:")

    for i, result in enumerate(results, start=1):

        print("\n" + "=" * 60)
        print(f"RESULT {i}")
        print(f"PAGE: {result['page_number']}")
        print(f"DISTANCE: {result['distance']:.4f}")
        print("=" * 60)

        print(result["text"])