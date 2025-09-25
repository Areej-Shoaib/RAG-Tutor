import os
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
chroma_client = chromadb.Client()

# Find where "Page 1" or "1" appears in PDF to check page number
def find_first_page_number(pdf_file, max_pages_to_check=20):
    reader = PdfReader(pdf_file)
    for i in range(min(max_pages_to_check, len(reader.pages))):
        text = reader.pages[i].extract_text()
        if text:
            lines = text.strip().split("\n")
            top = lines[:3]
            bottom = lines[-3:]
            if any("Page 1" in line for line in top + bottom) or any(line.strip() == "1" for line in top + bottom):
                return i  # where real Page 1 starts
    return 0  # fallback if not found


# Process PDF into chunks with correct page numbers
def process_book(pdf_file):
    reader = PdfReader(pdf_file)
    total_pages = len(reader.pages)

    # Find starting page index for actual Page 1
    start_page_index = find_first_page_number(pdf_file)

    collection = chroma_client.get_or_create_collection("book_chunks")
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            embeddings = [embedder.encode(chunk) for chunk in chunks]

            # Book page = PDF page index - start_page_index + 1
            book_page_num = page_num - start_page_index + 1
            if book_page_num < 1:  # pages before start are 0
                book_page_num = 0

            for i, chunk in enumerate(chunks):
                collection.add(
                    documents=[chunk],
                    metadatas=[{"page_num": book_page_num, "total_pages": total_pages}],
                    ids=[f"{page_num+1}_{i}"],
                    embeddings=[embeddings[i]]
                )
    return collection, total_pages


# Ask question with correct page validation
def ask_question(question, collection, total_pages):
    question_embedding = embedder.encode(question)
    results = collection.query(query_embeddings=[question_embedding], n_results=5)

    model = genai.GenerativeModel("gemini-1.5-flash")

    if not results["documents"] or not results["documents"][0]:
        return "❌ This topic is not discussed in the book."

    context = " ".join(results["documents"][0])

    # Extract valid book page numbers only
    page_citations = sorted(
        set(meta["page_num"] for meta in results["metadatas"][0] if 1 <= meta["page_num"] <= total_pages)
    )

    prompt = f"Answer the question using ONLY the following text:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    response = model.generate_content(prompt)

    citation_text = f"\n\n**Source: Pages {', '.join(map(str, page_citations))}**" if page_citations else ""
    return response.text + citation_text


# Streamlit UI
st.title("RAG Tutor - Learn from your Book")

uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_pdf:
    st.success("PDF uploaded successfully!")
    collection, total_pages = process_book(uploaded_pdf)
    st.info(f"PDF processed with {total_pages} pages. You can now ask questions.")

    question = st.text_input("Ask a question based on the PDF:")
    if st.button("Get Answer"):
        if question.strip() == "":
            st.warning("Please enter a question before clicking 'Get Answer'")
        else:
            answer = ask_question(question, collection, total_pages)
            st.markdown(f"**Answer:**\n{answer}")







