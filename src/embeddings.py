from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(texts):
    """
    Convert a list of text chunks into numerical vectors.
    """

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings


if __name__ == "__main__":
    print("Embedding model loaded successfully.")

    test_texts = [
        "The Transformer is based on attention mechanisms.",
        "The model uses self-attention."
    ]

    embeddings = create_embeddings(test_texts)

    print("Number of embeddings:", len(embeddings))
    print("Embedding dimension:", embeddings.shape[1])