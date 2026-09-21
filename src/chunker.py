from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from src.pdf_processor import extract_text_from_pdf
except ModuleNotFoundError:
    from pdf_processor import extract_text_from_pdf

def create_chunks(pages):
    """
    Split extracted PDF pages into smaller chunks.

    Each chunk keeps its original page number.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for page in pages:

        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "page_number": page["page_number"],
                "text": chunk
            })

    return chunks


if __name__ == "__main__":

    pdf_path = "data/papers/researchpaper.pdf"

    # Step 1: Extract text from PDF
    pages = extract_text_from_pdf(pdf_path)

    print(f"Total pages extracted: {len(pages)}")

    # Step 2: Create chunks
    chunks = create_chunks(pages)

    print(f"Total chunks created: {len(chunks)}")

    # Show first 3 chunks
    for i, chunk in enumerate(chunks[:3], start=1):

        print("\n" + "=" * 60)
        print(f"CHUNK {i}")
        print(f"PAGE: {chunk['page_number']}")
        print("=" * 60)

        print(chunk["text"])