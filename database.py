import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec

# Replicate API token
os.environ['REPLICATE_API_TOKEN'] = "YOUR REPLICATE_API_TOKEN"
# Pinecone API key
os.environ['PINECONE_API_KEY'] = "YOUR PINECONE_API_KEY"

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

# Set up the Pinecone vector database
index = pc.Index(index_name)

# Streamlit interface
st.title("Pinecone Data Viewer")

# Function to fetch data from Pinecone
def fetch_data_from_pinecone(index, top_k=100):
    results = index.query(
        vector=[0]*768,  # Dummy query vector
        top_k=top_k,
        include_values=True,
        include_metadata=True
    )
    return results['matches']

# Function to delete data from Pinecone
def delete_data_from_pinecone(index, id):
    index.delete(ids=[id])
    st.success(f"Data with ID {id} has been deleted.")

# Function to delete all data from Pinecone
def delete_all_data_from_pinecone(index):
    data = fetch_data_from_pinecone(index, top_k=1000)  # Increase top_k if you have more data
    ids_to_delete = [match['id'] for match in data]
    if ids_to_delete:
        index.delete(ids=ids_to_delete)
        st.success("All data has been deleted from the Pinecone index.")
    else:
        st.write("No data found to delete.")

# Button to delete all data
if st.button("Delete All Data"):
    delete_all_data_from_pinecone(index)

# Fetch data from Pinecone
data = fetch_data_from_pinecone(index)

# Display data in a tabular format with delete option
if data:
    st.write("### Retrieved Data from Pinecone")
    for match in data:
        st.write(f"ID: {match['id']}")
        st.write(f"Score: {match['score']}")
        st.write(f"Metadata: {match['metadata']}")
        if st.button(f"Delete {match['id']}"):
            delete_data_from_pinecone(index, match['id'])
        st.write("-----")
else:
    st.write("No data found in Pinecone index.")
