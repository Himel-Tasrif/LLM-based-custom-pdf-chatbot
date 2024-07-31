import os
import sys
import streamlit as st
from io import BytesIO
from uuid import uuid4
import PyPDF2  # PyPDF2 for PDF text extraction
from pinecone import Pinecone, ServerlessSpec
from langchain_community.llms import Replicate
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

# Replicate API token
os.environ['REPLICATE_API_TOKEN'] = "Your REPLICATE_API_TOKEN"
# Pinecone API key
os.environ['PINECONE_API_KEY'] = 'Your PINECONE_API_KEY'

# Initialize Pinecone
api_key = os.environ['PINECONE_API_KEY']
pc = Pinecone(api_key=api_key)

# Ensure the index exists
index_name = "himel06"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Based on your dimension
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

def extract_text_with_pypdf2(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Streamlit interface
st.title("PDF Chatbot by *** Tasrif Nur Himel ***")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    # Extract text from the PDF using PyPDF2
    extracted_text = extract_text_with_pypdf2(uploaded_file)

    # Convert the extracted text into Document format for Langchain
    documents = [Document(page_content=extracted_text)]

    # Split the documents into smaller chunks for processing
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)  # Adjust chunk size if needed
    texts = text_splitter.split_documents(documents)

    # Use HuggingFace embeddings for transforming text into numerical vectors
    embeddings = HuggingFaceEmbeddings()

    # Set up the Pinecone vector database
    index = pc.Index(index_name)
    vectordb = LangchainPinecone.from_documents(texts, embeddings, index_name=index_name)

    # Initialize Replicate Llama2 Model
    llm = Replicate(
        model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        model_kwargs={"temperature": 0.75, "max_length": 3000}
    )

    # Set up the Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectordb.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True
    )

    st.write("PDF successfully uploaded and processed. You can now ask questions about its content.")

    chat_history = st.session_state.get('chat_history', [])

    if 'init' not in st.session_state:
        st.session_state['init'] = True
        st.session_state['chat_history'] = []

    query = st.text_input("Prompt:")
    if st.button("Send"):
        if query.lower() in ["exit", "quit", "q"]:
            st.write('Exiting')
            sys.exit()
        result = qa_chain.invoke({'question': query, 'chat_history': st.session_state['chat_history']})
        st.write('Answer: ' + result['answer'])
        st.session_state['chat_history'].append((query, result['answer']))

    # Display the chat history
    st.write("### Chat History")
    for query, answer in st.session_state['chat_history']:
        st.write(f"**You:** {query}")
        st.write(f"**Bot:** {answer}")
