import streamlit as st
import tempfile
import os
import faiss
import numpy as np

from dotenv import load_dotenv
from google import genai

from src.pdf_processor import extract_text_from_pdf
from src.chunker import create_chunks
from src.embeddings import create_embeddings


# Load environment variables
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="📚",
    layout="wide"
)


st.title("📚 Research Paper Question Answering System")

st.write(
    "Upload a research paper and ask questions about its content."
)


# Check API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY was not found in the .env file.")
    st.stop()


# Upload PDF
uploaded_file = st.file_uploader(
    "Upload a research paper PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())

        pdf_path = temp_file.name


    with st.spinner("Processing research paper..."):

        # Extract text
        pages = extract_text_from_pdf(pdf_path)

        # Create chunks
        chunks = create_chunks(pages)

        # Create embeddings
        embeddings = create_embeddings(
            [chunk["text"] for chunk in chunks]
        )

        embeddings = np.array(
            embeddings
        ).astype("float32")

        # Create FAISS index
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)


    st.success(
        f"Paper processed successfully: "
        f"{len(pages)} pages and {len(chunks)} chunks."
    )


    # Question input
    question = st.text_input(
        "Ask a question about the paper:"
    )


    if st.button("🔎 Ask Question"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner("Searching the paper..."):

                # Create question embedding
                question_embedding = create_embeddings(
                    [question]
                )

                question_embedding = np.array(
                    question_embedding
                ).astype("float32")


                # Retrieve top 3 chunks
                distances, indices = index.search(
                    question_embedding,
                    3
                )


                retrieved_chunks = []

                for distance, index_number in zip(
                    distances[0],
                    indices[0]
                ):

                    if index_number != -1:

                        retrieved_chunks.append({
                            "text": chunks[index_number]["text"],
                            "page_number": chunks[index_number]["page_number"],
                            "distance": float(distance)
                        })


            # Build context
            context_parts = []

            for result in retrieved_chunks:

                context_parts.append(
                    f"[Page {result['page_number']}]\n"
                    f"{result['text']}"
                )


            context = "\n\n".join(context_parts)


            # Gemini prompt
            prompt = f"""
You are a research paper question-answering assistant.

Answer the user's question using ONLY the provided research paper context.

If the answer cannot be found in the provided context, say:

"I could not find this information in the research paper."

Do not invent information.

Include page citations in your answer using:
(Page X)

Research paper context:

{context}

User question:

{question}
"""


            with st.spinner("Generating answer..."):

                client = genai.Client(
                    api_key=api_key
                )

                interaction = client.interactions.create(
                    model="gemini-3.6-flash",
                    input=prompt
                )

                answer = interaction.output_text


            # Display answer
            st.subheader("Answer")

            st.write(answer)


            # Display sources
            st.subheader("Sources")

            source_pages = sorted(
                set(
                    result["page_number"]
                    for result in retrieved_chunks
                )
            )

            for page in source_pages:

                st.write(
                    f"📄 Page {page}"
                )


    # Clean up temporary PDF
    try:
        os.remove(pdf_path)
    except:
        pass