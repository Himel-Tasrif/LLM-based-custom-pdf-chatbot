# LLM Based Custom PDF Chatbot

<figure style="text-align: center;">
  <img src="https://media.licdn.com/dms/image/D5612AQHWdL-lVswI-Q/article-cover_image-shrink_600_2000/0/1722599085907?e=1727913600&v=beta&t=2RDSWjO6RJEynFUvAq-F8flaCLZkLqM4SuRUaUlvL2c" alt="Llama-PDF-Chatbot" width="300" height="200">
  <figcaption>This repository contains the code for a PDF chatbot application that leverages advanced large language models and vector databases to provide intelligent conversational responses based on the content of uploaded PDF documents. For Large Language Model i am using Llama here.
</figcaption>
</figure>
Features:
1. PDF Text Extraction: Uses PyMuPDF or any Text Extraction Library to extract text from uploaded PDF files.

2. Document Processing: Splits extracted text into manageable chunks for efficient processing.

3. Embeddings: Utilizes HuggingFace embeddings or any other embeddings to transform text into numerical vectors.

4. Vector Database: Integrates with Pinecone to store and retrieve document vectors for fast and accurate responses.

5. Conversational AI: Employs a Replicate Llama2 model to generate context-aware responses to user queries.

6. Streamlit Interface: Provides an easy-to-use web interface for uploading PDFs and interacting with the chatbot.