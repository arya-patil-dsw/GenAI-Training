# Import libraries
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBEr-PgOwhKnceTbgvRBICDDpCD6amRVVM"  # Replace with your API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to local PostgreSQL
conn = psycopg2.connect(
    dbname="ragdb",
    user="postgres",
    password="123456",
    host="localhost",
    port="5432"
)
register_vector(conn)
cur = conn.cursor()

# Function to clean text by removing null characters
def clean_text(text):
    return text.replace('\x00', '')  # Remove null bytes

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""  # Handle cases where text extraction fails
            text += clean_text(page_text)  # Clean text per page
    return text

# Load and split PDF content
def load_and_split_pdf(pdf_path):
    # Extract and clean text
    raw_text = extract_text_from_pdf(pdf_path)
    
    # Use LangChain's text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    documents = text_splitter.split_text(raw_text)
    # Clean each chunk again to be safe
    documents = [clean_text(doc) for doc in documents if doc.strip()]  # Remove empty chunks
    return documents

# Ingest documents into pgvector
def ingest_documents(documents):
    embeddings = embedding_model.encode(documents)
    cur.execute("TRUNCATE TABLE documents RESTART IDENTITY;")  # Clear existing data
    for doc, emb in zip(documents, embeddings):
        try:
            cur.execute(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                (doc, emb)
            )
        except psycopg2.Error as e:
            print(f"Failed to insert document: {e}")
            print(f"Problematic content: {repr(doc)}")
            conn.rollback()  # Roll back on error
            continue
    conn.commit()
    print(f"Ingested {len(documents)} document chunks into pgvector!")

# Retrieval function
def retrieve_documents(query, k=2):
    query_embedding = embedding_model.encode([query])[0]
    cur.execute(
        "SELECT content, embedding <=> %s AS distance "
        "FROM documents "
        "ORDER BY distance LIMIT %s",
        (query_embedding, k)
    )
    results = cur.fetchall()
    return [row[0] for row in results]

# Generation function
def generate_response(query, context):
    prompt = f"Query: {query}\nContext: {' '.join(context)}\nAnswer:"
    response = model.generate_content(prompt)
    return response.text

# Main execution
if __name__ == "__main__":
    # Path to your PDF file
    pdf_path = "document.pdf"  # Replace with your PDF file path
    
    # Load and split PDF
    try:
        documents = load_and_split_pdf(pdf_path)
        print(f"Extracted {len(documents)} chunks from PDF.")
        
        # Ingest into pgvector
        ingest_documents(documents)
        
        # Interactive chat loop
        while True:
            query = input("Ask a question about the PDF (or type 'exit' to quit): ")
            if query.lower() == "exit":
                break
            context = retrieve_documents(query)
            response = generate_response(query, context)
            print("Query:", query)
            print("Context:", context)
            print("Response:", response)
            print("-" * 50)
    
    except Exception as e:
        print(f"Error: {e}")
    
    # Cleanup
    cur.close()
    conn.close()