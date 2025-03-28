# Import libraries
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import chainlit as cl

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
    return text.replace('\x00', '')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += clean_text(page_text)
    return text

# Load and split PDF content
def load_and_split_pdf(pdf_path):
    raw_text = extract_text_from_pdf(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    documents = text_splitter.split_text(raw_text)
    documents = [clean_text(doc) for doc in documents if doc.strip()]
    return documents

# Check if PDF is already ingested
def is_pdf_ingested(pdf_path):
    cur.execute("CREATE TABLE IF NOT EXISTS pdf_metadata (pdf_name VARCHAR(255) PRIMARY KEY, ingested BOOLEAN)")
    conn.commit()
    
    pdf_name = os.path.basename(pdf_path)
    cur.execute("SELECT ingested FROM pdf_metadata WHERE pdf_name = %s", (pdf_name,))
    result = cur.fetchone()
    return result is not None and result[0]

# Mark PDF as ingested
def mark_pdf_ingested(pdf_path):
    pdf_name = os.path.basename(pdf_path)
    cur.execute("INSERT INTO pdf_metadata (pdf_name, ingested) VALUES (%s, %s) ON CONFLICT (pdf_name) DO UPDATE SET ingested = %s",
                (pdf_name, True, True))
    conn.commit()

# Ingest documents into pgvector
def ingest_documents(documents, pdf_path):
    if is_pdf_ingested(pdf_path):
        print(f"PDF {pdf_path} already ingested, skipping ingestion.")
        return
    
    embeddings = embedding_model.encode(documents)
    cur.execute("TRUNCATE TABLE documents RESTART IDENTITY;")
    for doc, emb in zip(documents, embeddings):
        try:
            cur.execute(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                (doc, emb)
            )
        except psycopg2.Error as e:
            print(f"Failed to insert document: {e}")
            print(f"Problematic content: {repr(doc)}")
            conn.rollback()
            continue
    conn.commit()
    mark_pdf_ingested(pdf_path)
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

# Chainlit setup
@cl.on_chat_start
async def start():
    # Set a title for the chatbot
    await cl.Message(content="Welcome to the PDF Chatbot! Ask me anything about your document.").send()
    
    # Path to your PDF file
    pdf_path = "document.pdf"  # Replace with your PDF file path
    
    # Load and ingest PDF if not already done
    try:
        documents = load_and_split_pdf(pdf_path)
        print(f"Extracted {len(documents)} chunks from PDF.")
        ingest_documents(documents, pdf_path)
    except Exception as e:
        await cl.Message(content=f"Error loading PDF: {e}").send()

@cl.on_message
async def main(message: cl.Message):
    query = message.content
    context = retrieve_documents(query)
    response = generate_response(query, context)
    
    # Send response with context for transparency
    await cl.Message(content=f"**Response:** {response}\n\n**Context Used:** {context}").send()

# Cleanup on script exit (optional)
def cleanup():
    cur.close()
    conn.close()

if __name__ == "__main__":
    # Run Chainlit (this is handled by the 'chainlit run' command)
    pass