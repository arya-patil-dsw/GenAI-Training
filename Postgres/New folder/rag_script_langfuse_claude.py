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
import tiktoken  # Add tiktoken for more accurate token counting
from datetime import datetime

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

# Define model costs per 1000 tokens (update these with actual prices)
MODEL_COSTS = {
    "gemini-2.0-flash": {
        "input": 0.00025,  # $0.00025 per 1K input tokens
        "output": 0.0005,  # $0.0005 per 1K output tokens
    },
    "all-MiniLM-L6-v2": {
        "embedding": 0.0001  # Cost per 1K tokens for embeddings (approximate)
    }
}

# Initialize tokenizer for better token counting (GPT tokenizer as approximation)
tokenizer = tiktoken.get_encoding("cl100k_base")

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

# Calculate token count more accurately
def count_tokens(text):
    return len(tokenizer.encode(text))

# Calculate cost based on token count and model
def calculate_cost(token_count, rate_per_1k):
    return (token_count / 1000) * rate_per_1k

# Async ingest documents into pgvector with Langfuse tracing and cost tracking
async def ingest_documents(documents, pdf_path):
    # Start ingest trace
    trace = langfuse.trace(
        name="ingest_documents",
        input={"pdf_path": pdf_path, "chunk_count": len(documents)},
        tags=["ingestion"]
    )
    
    if is_pdf_ingested(pdf_path):
        print(f"PDF {pdf_path} already ingested, skipping ingestion.")
        trace.update(output={"status": "skipped"})
        return
    
    # Calculate total tokens in documents for embedding
    total_tokens = sum(count_tokens(doc) for doc in documents)
    
    # Track embedding process with cost
    embedding_span = trace.span(
        name="embedding_generation",
        input={"document_count": len(documents), "total_tokens": total_tokens}
    )
    
    # Generate embeddings
    start_time = datetime.now()
    embeddings = await asyncio.to_thread(embedding_model.encode, documents)
    end_time = datetime.now()
    
    # Calculate embedding cost
    embedding_cost = calculate_cost(
        total_tokens, 
        MODEL_COSTS["all-MiniLM-L6-v2"]["embedding"]
    )
    
    # Update embedding span with metrics
    embedding_span.update(
        output={
            "embedding_count": len(embeddings),
            "processing_time_ms": (end_time - start_time).total_seconds() * 1000
        },
        metrics={
            "tokens": total_tokens,
            "cost": embedding_cost,
            "latency": (end_time - start_time).total_seconds()
        }
    )
    embedding_span.end()
    
    # Clear existing documents
    cur.execute("CREATE TABLE IF NOT EXISTS documents (id SERIAL PRIMARY KEY, content TEXT, embedding VECTOR(384))")
    cur.execute("TRUNCATE TABLE documents RESTART IDENTITY;")
    
    # Insert new documents
    db_span = trace.span(name="database_insertion")
    insert_start = datetime.now()
    
    for i, (doc, emb) in enumerate(zip(documents, embeddings)):
        try:
            with trace.span(name=f"insert_chunk_{i}", input={"chunk_length": len(doc)}) as chunk_span:
                cur.execute(
                    "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                    (doc, emb)
                )
                chunk_span.update(output={"status": "success"})
        except psycopg2.Error as e:
            print(f"Failed to insert document: {e}")
            print(f"Problematic content: {repr(doc)}")
            conn.rollback()
            trace.span(name=f"insert_chunk_{i}").update(output={"status": "error", "error": str(e)})
            continue
    
    conn.commit()
    insert_end = datetime.now()
    db_span.update(metrics={"latency": (insert_end - insert_start).total_seconds()})
    db_span.end()
    
    mark_pdf_ingested(pdf_path)
    print(f"Ingested {len(documents)} document chunks into pgvector!")
    
    # Update the main trace with complete metrics
    trace.update(
        output={
            "status": "success", 
            "document_count": len(documents),
            "total_tokens": total_tokens
        },
        metrics={
            "embedding_cost": embedding_cost,
            "total_cost": embedding_cost,  # For ingest, total cost is just embedding cost
            "total_tokens": total_tokens
        }
    )
    langfuse.flush()

# Async retrieval function with enhanced Langfuse tracing and metrics
async def retrieve_documents(query, k=2):
    # Start retrieval trace
    trace = langfuse.trace(
        name="retrieve_documents",
        input={"query": query, "k": k},
        tags=["retrieval"]
    )
    
    # Count tokens in query for cost calculation
    query_tokens = count_tokens(query)
    
    # Calculate embedding cost
    embedding_cost = calculate_cost(
        query_tokens,
        MODEL_COSTS["all-MiniLM-L6-v2"]["embedding"]
    )
    
    # Generate query embedding
    embedding_span = trace.span(
        name="query_embedding",
        input={"query": query, "token_count": query_tokens}
    )
    
    start_time = datetime.now()
    query_embedding = await asyncio.to_thread(embedding_model.encode, [query])
    end_time = datetime.now()
    
    embedding_span.update(
        output={"status": "success"},
        metrics={
            "tokens": query_tokens,
            "cost": embedding_cost,
            "latency": (end_time - start_time).total_seconds()
        }
    )
    embedding_span.end()
    
    query_embedding = query_embedding[0]
    
    # Perform vector search
    search_span = trace.span(name="vector_search", input={"k": k})
    search_start = datetime.now()
    
    cur.execute(
        "SELECT content, embedding <=> %s AS distance "
        "FROM documents "
        "ORDER BY distance LIMIT %s",
        (query_embedding, k)
    )
    
    results = cur.fetchall()
    search_end = datetime.now()
    
    # Extract content and calculate similarity scores
    context = [row[0] for row in results]
    distances = [float(row[1]) for row in results]
    
    # Convert distances to similarity scores (1 - distance)
    similarity_scores = [1 - dist for dist in distances]
    
    # Calculate retrieval quality score (average similarity)
    retrieval_quality = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    
    search_span.update(
        output={
            "found_documents": len(results),
            "similarity_scores": similarity_scores
        },
        metrics={
            "latency": (search_end - search_start).total_seconds(),
            "retrieval_quality": retrieval_quality
        }
    )
    search_span.end()
    
    # Log retrieval quality score
    langfuse.score(
        trace_id=trace.id,
        name="retrieval_quality",
        value=retrieval_quality,
        comment="Average similarity score of retrieved documents"
    )
    
    # Update main trace with final metrics
    trace.update(
        output={
            "context": context,
            "similarity_scores": similarity_scores
        },
        metrics={
            "embedding_cost": embedding_cost,
            "total_cost": embedding_cost  # For retrieval, total cost is just the embedding cost
        }
    )
    
    langfuse.flush()
    return context, similarity_scores

# Enhanced async generation function with comprehensive Langfuse tracing and cost calculation
async def generate_response(query, context, similarity_scores=None):
    # Start generation trace
    trace = langfuse.trace(
        name="generate_response",
        input={
            "query": query, 
            "context": context,
            "similarity_scores": similarity_scores
        },
        tags=["generation"]
    )
    
    # Construct the prompt
    context_text = ' '.join(context)
    prompt = f"Query: {query}\nContext: {context_text}\nAnswer:"
    
    # Count tokens for cost calculation
    input_tokens = count_tokens(prompt)
    
    # Calculate input cost
    input_cost = calculate_cost(
        input_tokens,
        MODEL_COSTS["gemini-2.0-flash"]["input"]
    )
    
    # Generate response
    generation_span = trace.span(
        name="gemini_generation",
        input={"prompt": prompt, "token_count": input_tokens}
    )
    
    start_time = datetime.now()
    response = await asyncio.to_thread(model.generate_content, prompt)
    response_text = response.text
    end_time = datetime.now()
    
    # Count output tokens
    output_tokens = count_tokens(response_text)
    
    # Calculate output cost
    output_cost = calculate_cost(
        output_tokens,
        MODEL_COSTS["gemini-2.0-flash"]["output"]
    )
    
    # Calculate total cost
    total_cost = input_cost + output_cost
    
    # Log generation details
    generation_span.update(
        output={"response": response_text},
        metrics={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "latency": (end_time - start_time).total_seconds()
        }
    )
    generation_span.end()
    
    # Create a generation observation with accurate token counts and costs
    trace.generation(
        name="gemini_call",
        model="gemini-2.0-flash",
        modelParameters={
            "temperature": 0.7,  # Add actual parameters if available
            "maxOutputTokens": 1024
        },
        input=prompt,
        output=response_text,
        usage={
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
            "unit": "TOKENS"
        },
        promptTokens=input_tokens,
        completionTokens=output_tokens,
        totalTokens=input_tokens + output_tokens,
        startTime=start_time.isoformat(),
        endTime=end_time.isoformat()
    )
    
    # Calculate and log multiple quality scores
    
    # 1. Response length score (normalized)
    length_score = min(1.0, len(response_text) / 1000.0)
    langfuse.score(
        trace_id=trace.id,
        name="response_length",
        value=length_score,
        comment="Normalized length of response"
    )
    
    # 2. Response/query relevance score (approximated by average similarity score)
    if similarity_scores:
        relevance_score = sum(similarity_scores) / len(similarity_scores)
        langfuse.score(
            trace_id=trace.id,
            name="context_relevance",
            value=relevance_score,
            comment="Average similarity score of retrieved context"
        )
    
    # 3. Helpfulness score (to be rated by user)
    # This will be set by the thumbs up/down feedback
    
    # Update main trace with final cost metrics
    trace.update(
        output={"response": response_text},
        metrics={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
    )
    
    langfuse.flush()
    return response_text

# Chainlit setup
@cl.on_chat_start
async def start():
    session = {"trace_ids": {}}  # Store trace IDs for the session
    await cl.Message(content="Welcome to the PDF Chatbot! Initializing...").send()
    
    try:
        # Initialize PDF processing
        pdf_path = "document.pdf"  # Replace with your PDF file path
        
        # Create a session trace to track overall usage
        session_trace = langfuse.trace(
            name="user_session",
            tags=["session"]
        )
        cl.user_session.set("session_trace_id", session_trace.id)
        
        # Process PDF
        documents = await asyncio.to_thread(load_and_split_pdf, pdf_path)
        print(f"Extracted {len(documents)} chunks from PDF.")
        
        # Ingest documents
        await ingest_documents(documents, pdf_path)
        
        # Update session trace
        session_trace.update(
            output={"status": "initialized", "document_chunks": len(documents)}
        )
        
        await cl.Message(content="Initialization complete! Ask me anything about your document.").send()
    except Exception as e:
        await cl.Message(content=f"Error during initialization: {e}").send()
        if 'session_trace_id' in cl.user_session:
            session_trace_id = cl.user_session.get("session_trace_id")
            langfuse.trace(trace_id=session_trace_id).update(
                output={"status": "error", "error": str(e)}
            )
        langfuse.flush()

@cl.on_message
async def main(message: cl.Message):
    try:
        query = message.content
        
        # Track this query in the session trace
        session_trace_id = cl.user_session.get("session_trace_id")
        interaction_trace = langfuse.trace(
            name="user_interaction",
            tags=["interaction"],
            parentTraceId=session_trace_id
        )
        
        # Store interaction trace ID for feedback
        cl.user_session.set("current_interaction_id", interaction_trace.id)
        
        # Retrieve documents with similarity scores
        context, similarity_scores = await retrieve_documents(query)
        
        # Generate response with context and similarity scores
        response = await generate_response(query, context, similarity_scores)
        
        # Format the response with context
        formatted_response = f"**Response:** {response}\n\n**Context Used:**"
        for i, (ctx, score) in enumerate(zip(context, similarity_scores)):
            # Show truncated context with similarity score
            truncated_ctx = ctx[:100] + "..." if len(ctx) > 100 else ctx
            formatted_response += f"\n\n{i+1}. (Relevance: {score:.2f}) {truncated_ctx}"
        
        # Send response with feedback buttons for user scoring
        msg = cl.Message(content=formatted_response)
        msg.actions = [
            cl.Action(name="thumbs_up", value="positive", label="👍"),
            cl.Action(name="thumbs_down", value="negative", label="👎"),
            cl.Action(name="accurate", value="accurate", label="✓ Accurate"),
            cl.Action(name="inaccurate", value="inaccurate", label="✗ Inaccurate")
        ]
        await msg.send()
        
        # Update interaction trace
        interaction_trace.update(
            output={
                "query": query,
                "response": response,
                "context_count": len(context)
            }
        )
        
    except Exception as e:
        await cl.Message(content=f"Error processing query: {e}").send()
        if 'current_interaction_id' in cl.user_session:
            trace_id = cl.user_session.get("current_interaction_id")
            langfuse.trace(trace_id=trace_id).update(
                output={"status": "error", "error": str(e)}
            )
        langfuse.flush()

# Enhanced feedback callbacks with multiple scoring dimensions
@cl.action_callback("thumbs_up")
async def on_thumbs_up(action):
    trace_id = cl.user_session.get("current_interaction_id")
    if trace_id:
        langfuse.score(
            trace_id=trace_id,
            name="user_satisfaction",
            value=1.0,
            comment="User gave a thumbs up"
        )
        langfuse.flush()
    await cl.Message(content="Thanks for your positive feedback!").send()

@cl.action_callback("thumbs_down")
async def on_thumbs_down(action):
    trace_id = cl.user_session.get("current_interaction_id")
    if trace_id:
        langfuse.score(
            trace_id=trace_id,
            name="user_satisfaction",
            value=0.0,
            comment="User gave a thumbs down"
        )
        langfuse.flush()
    await cl.Message(content="Thanks for your feedback. Could you tell us what could be improved?").send()

@cl.action_callback("accurate")
async def on_accurate(action):
    trace_id = cl.user_session.get("current_interaction_id")
    if trace_id:
        langfuse.score(
            trace_id=trace_id,
            name="factual_accuracy",
            value=1.0,
            comment="User marked response as accurate"
        )
        langfuse.flush()
    await cl.Message(content="Thanks for confirming the accuracy!").send()

@cl.action_callback("inaccurate")
async def on_inaccurate(action):
    trace_id = cl.user_session.get("current_interaction_id")
    if trace_id:
        langfuse.score(
            trace_id=trace_id,
            name="factual_accuracy",
            value=0.0,
            comment="User marked response as inaccurate"
        )
        langfuse.flush()
    await cl.Message(content="Thanks for reporting inaccuracy. Could you specify what was incorrect?").send()

# Enhanced cleanup function
def cleanup():
    # Get session trace ID if available
    session_trace_id = cl.user_session.get("session_trace_id") if hasattr(cl, 'user_session') else None
    
    # If we have a session, finalize it with metrics
    if session_trace_id:
        try:
            # Calculate total costs and metrics for the session
            # This would require fetching all child traces and summing their costs
            # For simplicity, we just mark it as completed
            langfuse.trace(trace_id=session_trace_id).update(
                output={"status": "completed"}
            )
        except Exception as e:
            print(f"Error finalizing session: {e}")
    
    # Close database connection
    cur.close()
    conn.close()
    
    # Flush remaining telemetry
    langfuse.flush()
    print("Cleanup completed successfully")

if __name__ == "__main__":
    import atexit
    atexit.register(cleanup)