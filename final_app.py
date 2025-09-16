
import os
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# Loading API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initializing Models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()

# Processing PDF
def process_book(pdf_file):
    reader = PdfReader(pdf_file)
    collection = chroma_client.get_or_create_collection("book_chunks")
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            # Split page text into chunks
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            embeddings = [embedder.encode(chunk) for chunk in chunks]
            
            for i, chunk in enumerate(chunks):
                collection.add(
                    documents=[chunk],
                    metadatas=[{"chunk_id": i, "page_num": page_num + 1}],  
                    ids=[f"{page_num+1}_{i}"],  
                    embeddings=[embeddings[i]]
                )
    return collection

# Ask Question
def ask_question(question, collection):
    question_embedding = embedder.encode(question)
    results = collection.query(query_embeddings=[question_embedding], n_results=3)

    if not results['documents'][0]:
        return "Insufficient evidence"

    context = " ".join(results['documents'][0])
    
    page_citations = sorted(set([meta.get("page_num") for meta in results["metadatas"][0] if "page_num" in meta]))

    prompt = f"Answer the question using ONLY the following text:\n\n{context}\n\nQuestion: {question}\nAnswer:"

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    citation_text = f"\n\n **Source: Pages {', '.join(map(str, page_citations))}**"
    return response.text + citation_text

# Streamlit UI 
st.title("RAG Tutor - Learn from your Book")

uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_pdf:
    st.success("PDF uploaded successfully!")
    collection = process_book(uploaded_pdf)
    st.info("PDF processed. You can now ask questions.")

    question = st.text_input("Ask a question based on the PDF:")
    if st.button("Get Answer"):
        if question.strip() == "":
            st.warning("Please enter a question before clicking 'Get Answer'")
        else:
            answer = ask_question(question, collection)
            st.markdown(f"Answer:\n{answer}")




