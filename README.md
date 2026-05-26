# RAG Tutor – Retrieval-Augmented Generation System

A document-based question answering system that allows users to query 
custom PDF documents and receive contextually grounded responses using 
a combination of vector retrieval and large language models.

---

## Overview

RAG Tutor implements a full Retrieval-Augmented Generation pipeline. 
Documents are parsed, chunked, and converted into dense vector embeddings 
stored in a ChromaDB vector database. On receiving a user query, the system 
retrieves the most semantically relevant context and passes it to a generative 
model to produce accurate, document-grounded responses — reducing hallucination 
and improving answer relevance.

---

## Tech Stack

- **Language:** Python
- **Framework:** Streamlit
- **ML Libraries:** PyTorch, Transformers, Sentence-Transformers
- **Vector Database:** ChromaDB
- **LLM:** Google Generative AI
- **Document Parsing:** PyPDF

---

## Features

- Semantic question answering over custom PDF documents
- Full RAG pipeline with vector-based retrieval
- Context-aware response generation using LLMs
- Support for multiple PDF inputs
- Interactive real-time web interface
