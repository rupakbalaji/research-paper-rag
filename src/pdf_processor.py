import pymupdf


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF while preserving page numbers.

    Returns:
        list of dictionaries containing page number and page text.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()

        if text:
            pages.append({
                "page_number": page_number,
                "text": text
            })

    document.close()

    return pages


if __name__ == "__main__":

    pdf_path = "data/papers/researchpaper.pdf"

    pages = extract_text_from_pdf(pdf_path)

    print(f"\nTotal pages with text: {len(pages)}")

    for page in pages[:2]:
        print("\n" + "=" * 60)
        print(f"PAGE {page['page_number']}")
        print("=" * 60)
        print(page["text"][:1000])