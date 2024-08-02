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


## First of all, we need to create accounts:
### Pinecone Account:

Pinecone is a vector database designed to manage, store, and query high-dimensional vectors efficiently. It is particularly useful in machine learning applications for tasks such as similarity search, natural language processing, and recommendation systems. In our project, Pinecone is used to store and retrieve the vectorized text data from the PDF, enabling fast and accurate query responses.

### Setup:

Visit Pinecone’s website: Go to Pinecone.
Sign up: Click on the sign-up button and fill out the required information to create an account.
Access API Keys: Once logged in, go to the left side navigation menu and click on “API Keys”.
Copy API Key: Copy the API key displayed on the screen. We will use this key later in the code.
Create an Index: Go to the “Indexes” tab and create a new index.
Name the index as you prefer.
Set the dimensions to 768, which corresponds to the dimensionality of the Hugging Face embeddings used in the project.

### Replicate Account:

Replicate is a platform that provides access to machine learning models via a simple API. It allows developers to integrate powerful models into their applications without needing extensive machine learning expertise. In this project, Replicate is used to access the Llama AI model, which powers the natural language understanding capabilities of the chatbot, enabling it to comprehend and respond to user queries effectively.

### Setup:

Visit Replicate’s website: Go to Replicate.
Sign up: Create an account by providing the necessary information.
Get the API token: After logging in, go to the API section to get your Replicate API token. API Key: https://replicate.com/account/api-tokens