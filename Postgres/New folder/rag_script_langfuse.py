# Import libraries
import asyncio
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import chainlit as cl
from langfuse import Langfuse

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBEr-PgOwhKnceTbgvRBICDDpCD6amRVVM"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize Langfuse
langfuse = Langfuse(
    public_key="pk-lf-82509dc0-acf0-4e1e-9169-6ada712cd6c5",
    secret_key="sk-lf-6b209a5a-409e-464d-9ba7-8d5df4efa0f8",
    host="https://cloud.langfuse.com"
)

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

# Async ingest documents into pgvector with Langfuse tracing
async def ingest_documents(documents, pdf_path):
    trace = langfuse.trace(
        name="ingest_documents",
        input={"pdf_path": pdf_path, "chunk_count": len(documents)},
        tags=["ingestion"]
    )
    
    if is_pdf_ingested(pdf_path):
        print(f"PDF {pdf_path} already ingested, skipping ingestion.")
        trace.update(output={"status": "skipped"})
        return
    
    embeddings = await asyncio.to_thread(embedding_model.encode, documents)
    cur.execute("TRUNCATE TABLE documents RESTART IDENTITY;")
    for i, (doc, emb) in enumerate(zip(documents, embeddings)):
        try:
            with trace.span(name=f"insert_chunk_{i}", input=doc):
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
    trace.update(output={"status": "success"})
    langfuse.flush()

# Async retrieval function with Langfuse tracing
async def retrieve_documents(query, k=2):
    trace = langfuse.trace(
        name="retrieve_documents",
        input={"query": query, "k": k},
        tags=["retrieval"]
    )
    query_embedding = await asyncio.to_thread(embedding_model.encode, [query])
    query_embedding = query_embedding[0]
    cur.execute(
        "SELECT content, embedding <=> %s AS distance "
        "FROM documents "
        "ORDER BY distance LIMIT %s",
        (query_embedding, k)
    )
    results = cur.fetchall()
    context = [row[0] for row in results]
    trace.update(output={"context": context})
    langfuse.flush()
    return context

# Async generation function with Langfuse tracing, scores, and cost tracking
async def generate_response(query, context):
    trace = langfuse.trace(
        name="generate_response",
        input={"query": query, "context": context},
        tags=["generation"]
    )
    
    prompt = f"Query: {query}\nContext: {' '.join(context)}\nAnswer:"
    response = await asyncio.to_thread(model.generate_content, prompt)
    
    # Capture token usage (Gemini API doesn't return this directly, so we estimate)
    # For accurate usage, you'd need to integrate a tokenizer or use a model that provides usage data
    input_tokens = len(prompt.split())  # Rough estimate
    output_tokens = len(response.text.split())  # Rough estimate
    total_tokens = input_tokens + output_tokens
    
    # Log usage and cost to Langfuse
    trace.generation(
        name="gemini_call",
        model="gemini-2.0-flash",
        input=prompt,
        output=response.text,
        usage={
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
            "unit": "TOKENS"
        }
    )
    
    # Add a simple evaluation score (e.g., response length as a heuristic)
    length_score = min(1.0, len(response.text) / 1000.0)  # Normalize to 0-1
    langfuse.score(
        trace_id=trace.id,
        name="response_length",
        value=length_score,
        comment="Normalized length of response"
    )
    
    trace.update(output={"response": response.text})
    langfuse.flush()
    return response.text

# Chainlit setup
@cl.on_chat_start
async def start():
    try:
        await cl.Message(content="Welcome to the PDF Chatbot! Initializing...").send()
        pdf_path = "document.pdf"  # Replace with your PDF file path
        documents = await asyncio.to_thread(load_and_split_pdf, pdf_path)
        print(f"Extracted {len(documents)} chunks from PDF.")
        await ingest_documents(documents, pdf_path)
        await cl.Message(content="Initialization complete! Ask me anything about your document.").send()
    except Exception as e:
        await cl.Message(content=f"Error during initialization: {e}").send()

@cl.on_message
async def main(message: cl.Message):
    try:
        query = message.content
        context = await retrieve_documents(query)
        response = await generate_response(query, context)
        
        # Send response with feedback buttons for user scoring
        msg = cl.Message(content=f"**Response:** {response}\n\n**Context Used:** {context}")
        msg.actions = [
            cl.Action(name="thumbs_up", value="positive", label="👍"),
            cl.Action(name="thumbs_down", value="negative", label="👎")
        ]
        await msg.send()
    except Exception as e:
        await cl.Message(content=f"Error processing query: {e}").send()

# Capture user feedback as scores
@cl.action_callback("thumbs_up")
async def on_thumbs_up(action):
    trace_id = langfuse.get_current_trace_id()
    if trace_id:
        langfuse.score(
            trace_id=trace_id,
            name="user_feedback",
            value=1.0,
            comment="User gave a thumbs up"
        )
        langfuse.flush()

@cl.action_callback("thumbs_down")
async def on_thumbs_down(action):
    trace_id = langfuse.get_current_trace_id()
    if trace_id:
        langfuse.score(
            trace_id=trace_id,
            name="user_feedback",
            value=0.0,
            comment="User gave a thumbs down"
        )
        langfuse.flush()

# Cleanup on script exit
def cleanup():
    cur.close()
    conn.close()
    langfuse.flush()

if __name__ == "__main__":
    import atexit
    atexit.register(cleanup)