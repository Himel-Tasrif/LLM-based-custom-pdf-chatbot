# LLM Based Custom PDF Chatbot

This repository contains the code for a PDF chatbot application that leverages advanced language models and vector databases to provide intelligent conversational responses based on the content of uploaded PDF documents.

Features:
1. PDF Text Extraction: Uses PyMuPDF or any Text Extraction Library to extract text from uploaded PDF files.

2. Document Processing: Splits extracted text into manageable chunks for efficient processing.

3. Embeddings: Utilizes HuggingFace embeddings or any other embeddings to transform text into numerical vectors.

4. Vector Database: Integrates with Pinecone to store and retrieve document vectors for fast and accurate responses.

5. Conversational AI: Employs a Replicate Llama2 model to generate context-aware responses to user queries.

6. Streamlit Interface: Provides an easy-to-use web interface for uploading PDFs and interacting with the chatbot.