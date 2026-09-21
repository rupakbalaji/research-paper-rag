\# Research Paper Question Answering System



A Generative AI application that allows users to upload a research paper in PDF format and ask questions about its content.



The project uses Retrieval-Augmented Generation (RAG) to retrieve relevant sections from the uploaded research paper and generate answers grounded in those sections.



\## Features



\- Upload a research paper PDF

\- Extract text from the PDF

\- Split the document into smaller chunks

\- Generate semantic embeddings

\- Store embeddings using FAISS

\- Retrieve the most relevant chunks for a question

\- Generate answers using Gemini

\- Provide page-number citations

\- Simple Streamlit web interface



\## Project Architecture



```text

Research Paper PDF

&#x20;       ↓

&#x20;  Text Extraction

&#x20;       ↓

&#x20;     Chunking

&#x20;       ↓

&#x20;  Embeddings

&#x20;       ↓

&#x20;   FAISS Index

&#x20;       ↓

&#x20;  User Question

&#x20;       ↓

&#x20;Semantic Retrieval

&#x20;       ↓

&#x20;Relevant Context

&#x20;       ↓

&#x20;Gemini 3.6 Flash

&#x20;       ↓

&#x20;Answer + Page Citation

Technologies Used
Python
Streamlit
PyMuPDF
LangChain Text Splitters
Sentence Transformers
FAISS
Google Gemini API
Embedding Model

The project uses:

all-MiniLM-L6-v2

The embedding dimension is 384.

RAG Configuration
Parameter	Value
Chunk size	1000 characters
Chunk overlap	200 characters
Retrieval depth	Top 3 chunks
Vector database	FAISS
Distance metric	L2
Generation model	Gemini 3.6 Flash
How It Works
The user uploads a research paper.
PyMuPDF extracts the text while preserving page numbers.
The text is divided into smaller chunks.
Sentence Transformers converts the chunks into numerical embeddings.
FAISS stores the embeddings for similarity search.
The user's question is converted into an embedding.
The system retrieves the three most relevant chunks.
The retrieved context is sent to Gemini.
Gemini generates an answer using only the retrieved research-paper context.
Page citations are included in the answer.
Why RAG?

A normal language model may answer a question using its general knowledge.

RAG first retrieves relevant information from the uploaded research paper and provides that information to the language model as context.

This helps the system produce answers grounded in the selected document and provides source-page references.

Example

Example question:

What is the Transformer architecture?

The system retrieves relevant sections from the research paper and generates an answer with page citations.

Project Structure
research-paper-rag/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── data/
│   └── papers/
│       └── researchpaper.pdf
│
└── src/
    ├── chunker.py
    ├── embeddings.py
    ├── gemini_test.py
    ├── pdf_processor.py
    ├── rag_pipeline.py
    ├── retriever.py
    └── vector_store.py

Running the Application Locally

Create and activate the virtual environment:

python -m venv venv
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Create a .env file:

GEMINI_API_KEY=your_api_key_here

Run the application:

streamlit run app.py

The application will open in your browser.

Security

The Gemini API key is stored in .env.

The .env file is excluded from Git using .gitignore and should never be uploaded to GitHub.

Limitations
The application currently processes the uploaded document during the session.
Retrieval quality depends on chunking and embedding quality.
Scanned PDFs may require OCR.
An internet connection and Gemini API access are required for answer generation.
Future Enhancements
Support multiple research papers
Persistent vector databases
Better document management
Configurable chunk size and retrieval depth
Display retrieved source excerpts
OCR support for scanned documents
Evaluation of retrieval and answer quality
Author

Rupak Balaji

GitHub: https://github.com/rupakbalaji
