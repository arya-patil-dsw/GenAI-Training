# Import libraries
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

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
    host="localhost",  # Local connection, no ngrok needed
    port="5432"        # Default port from docker-compose
)
register_vector(conn)
cur = conn.cursor()

# Ingest documents
documents = [
    "The sky is blue because of Rayleigh scattering.",
    "Python is a popular programming language.",
    "The Earth orbits the Sun in 365.25 days."
]
embeddings = embedding_model.encode(documents)

# Clear existing data (optional)
cur.execute("TRUNCATE TABLE documents RESTART IDENTITY;")
for doc, emb in zip(documents, embeddings):
    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        (doc, emb)
    )
conn.commit()
print("Documents ingested into pgvector!")

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

# Test the RAG system
query = "Why is the sky blue?"
context = retrieve_documents(query)
response = generate_response(query, context)
print("Query:", query)
print("Context:", context)
print("Response:", response)

# Cleanup
cur.close()
conn.close()