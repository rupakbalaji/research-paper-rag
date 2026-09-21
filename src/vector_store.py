import faiss
import numpy as np

from pdf_processor import extract_text_from_pdf
from chunker import create_chunks
from embeddings import create_embeddings


def build_vector_store(chunks):
    """
    Create a FAISS vector index from document chunks.
    """

    texts = [chunk["text"] for chunk in chunks]

    # Create embeddings
    embeddings = create_embeddings(texts)

    # Convert to float32 for FAISS
    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to the index
    index.add(embeddings)

    return index, chunks


if __name__ == "__main__":

    pdf_path = "data/papers/researchpaper.pdf"

    # Step 1: Extract text
    pages = extract_text_from_pdf(pdf_path)

    print(f"Pages extracted: {len(pages)}")

    # Step 2: Create chunks
    chunks = create_chunks(pages)

    print(f"Chunks created: {len(chunks)}")

    # Step 3: Build FAISS vector store
    index, stored_chunks = build_vector_store(chunks)

    print(f"Vectors stored in FAISS: {index.ntotal}")
    print(f"Vector dimension: {index.d}")