import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
import os

from pdf_processor import extract_text_from_pdf
from chunker import create_chunks
from embeddings import create_embeddings


# Load environment variables
load_dotenv()


def build_vector_store(chunks):
    """
    Create a FAISS vector index from document chunks.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = create_embeddings(texts)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def retrieve_chunks(question, chunks, index, top_k=3):
    """
    Retrieve the most relevant chunks for a question.
    """

    question_embedding = create_embeddings([question])

    question_embedding = np.array(question_embedding).astype("float32")

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for distance, index_number in zip(
        distances[0],
        indices[0]
    ):

        if index_number != -1:

            results.append({
                "text": chunks[index_number]["text"],
                "page_number": chunks[index_number]["page_number"],
                "distance": float(distance)
            })

    return results


def generate_answer(question, retrieved_chunks):
    """
    Generate an answer using only the retrieved paper context.
    """

    context_parts = []

    for result in retrieved_chunks:

        context_parts.append(
            f"[Page {result['page_number']}]\n"
            f"{result['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a research paper question-answering assistant.

Answer the user's question using ONLY the provided research paper context.

If the answer cannot be found in the provided context, say:

"I could not find this information in the research paper."

Do not invent information.

Include page citations in your answer using this format:
(Page X)

Research paper context:

{context}

User question:
{question}
"""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    return interaction.output_text


if __name__ == "__main__":

    pdf_path = "data/papers/researchpaper.pdf"

    print("Loading research paper...")

    # Extract text
    pages = extract_text_from_pdf(pdf_path)

    print(f"Pages extracted: {len(pages)}")

    # Create chunks
    chunks = create_chunks(pages)

    print(f"Chunks created: {len(chunks)}")

    # Build vector store
    print("Building FAISS vector store...")

    index = build_vector_store(chunks)

    print(f"Vectors stored: {index.ntotal}")

    # Ask a question
    question = "What is the Transformer architecture?"

    print("\nQuestion:")
    print(question)

    # Retrieve relevant chunks
    retrieved_chunks = retrieve_chunks(
        question,
        chunks,
        index,
        top_k=3
    )

    print("\nRetrieved pages:")

    for result in retrieved_chunks:
        print(
            f"Page {result['page_number']} "
            f"(distance: {result['distance']:.4f})"
        )

    # Generate answer
    print("\nGenerating answer...")

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(answer)