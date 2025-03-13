import streamlit as st
import tempfile
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import groq
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set page configuration
st.set_page_config(
    page_title="EnhancedPDF RAG Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("📚 EnhancedPDF RAG Chat")
st.markdown("""
    Chat with your PDF documents using advanced RAG techniques and Groq LLM.
    This app enhances standard RAG with multi-document support, document comparison, 
    interactive citation verification, and concept visualization.
""")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_store" not in st.session_state:
    st.session_state.document_store = {}
if "index" not in st.session_state:
    st.session_state.index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "active_documents" not in st.session_state:
    st.session_state.active_documents = []
if "document_summaries" not in st.session_state:
    st.session_state.document_summaries = {}
if "recent_sources" not in st.session_state:
    st.session_state.recent_sources = []
if "comparison_mode" not in st.session_state:
    st.session_state.comparison_mode = False
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None


class EnhancedRAG:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.dimension = 384  # Dimension for the embedding model
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.document_chunks = []
        self.document_ids = []
        
        # Initialize Groq client
        self.groq_api_key = os.getenv("GROQ_API_KEY", "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")
        if not self.groq_api_key:
            st.sidebar.warning("Please set the GROQ_API_KEY environment variable")
        self.groq_client = groq.Client(api_key=self.groq_api_key)
        self.model_name = "llama3-70b-8192"  # Default Groq model
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings for a given text"""
        return self.embedding_model.encode(text)
    
    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None) -> int:
        """Process and add a document to the vector store"""
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create embeddings for each chunk
        chunk_count = 0
        for i, chunk in enumerate(chunks):
            embedding = self.embed_text(chunk)
            embedding_np = np.array([embedding], dtype=np.float32)
            
            # Add to FAISS index
            self.index.add(embedding_np)
            
            # Store chunk and metadata
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_id": i,
                "doc_id": doc_id,
                "text": chunk
            })
            self.document_chunks.append(chunk_metadata)
            self.document_ids.append(doc_id)
            chunk_count += 1
        
        return chunk_count
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks based on the query"""
        # Check if index is empty
        if self.index.ntotal == 0 or len(self.document_chunks) == 0:
            st.warning("No documents have been indexed yet. Please upload and select documents first.")
            return []
            
        # Generate query embedding
        query_embedding = self.embed_text(query)
        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        
        # Limit top_k to the number of documents available
        actual_top_k = min(top_k, self.index.ntotal)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding_np, actual_top_k)
        
        # Get the corresponding document chunks
        results = []
        for i, idx in enumerate(indices[0]):
            # Make sure idx is valid and within range
            if 0 <= idx < len(self.document_chunks):
                result = self.document_chunks[idx].copy()
                result["score"] = float(distances[0][i])
                results.append(result)
        
        return results
    
    def generate_response(self, query: str, context: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> str:
        """Generate a response using Groq based on query and retrieved context"""
        # Handle empty context
        if not context:
            return "I don't have enough information to answer that question. Please upload relevant documents or try a different question."
            
        # Format the context
        formatted_context = "\n\n".join([
            f"[Document: {item['doc_id']}, Chunk: {item['chunk_id']}]\n{item['text']}"
            for item in context
        ])
        
        # Format chat history
        formatted_history = ""
        if chat_history:
            formatted_history = "\n".join([
                f"Human: {msg['content']}\nAssistant: {msg['response']}" 
                for msg in chat_history if msg.get('response')
            ])
        
        # Create the prompt with better formatting and structure
        prompt = f"""You are an intelligent assistant specializing in analyzing and discussing document content.

CONTEXT INFORMATION:
{formatted_context}

PREVIOUS CONVERSATION:
{formatted_history}

INSTRUCTIONS:
- Answer the user's question based primarily on the provided context
- If the context doesn't contain the answer, clearly state that you don't have that information from the documents
- When referencing information from the context, cite the source document and chunk ID in brackets [Document: X, Chunk: Y]
- Provide a comprehensive answer that synthesizes information from multiple sources when available
- If you detect inconsistencies between documents, point them out
- Use markdown formatting to make your response readable with headers, bullet points, etc.

User Question: {query}
Assistant:"""

        # Call Groq API
        try:
            completion = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on document context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
                top_p=1,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def generate_document_summary(self, doc_id: str, text: str) -> str:
        """Generate a summary of a document using Groq"""
        # Create the prompt for document summarization
        prompt = f"""Please provide a concise summary of the following document.
        Focus on the main topics, key points, and conclusions.
        
        DOCUMENT CONTENT:
        {text[:4000]}  # Limit to first 4000 chars for summary
        
        Summary:"""
        
        try:
            completion = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes documents accurately."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                top_p=1,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating document summary: {str(e)}")
            return f"Failed to generate summary: {str(e)}"
    
    def compare_documents(self, doc_ids: List[str], query: str = None) -> str:
        """Compare multiple documents on a specific topic or in general"""
        # Collect document content
        documents = []
        for doc_id in doc_ids:
            doc_chunks = [chunk for chunk in self.document_chunks if chunk["doc_id"] == doc_id]
            if doc_chunks:
                # Combine chunks to represent the document
                doc_text = "\n\n".join([chunk["text"] for chunk in doc_chunks])
                documents.append((doc_id, doc_text[:3000]))  # Limit size for comparison
        
        # Create comparison prompt
        topic_text = f"focusing on '{query}'" if query else "covering their main points, similarities, and differences"
        prompt = f"""Compare the following documents {topic_text}.
        
        """
        
        for i, (doc_id, text) in enumerate(documents):
            prompt += f"\nDOCUMENT {i+1} (ID: {doc_id}):\n{text}\n"
        
        prompt += """
        
        Please provide:
        1. A summary of each document's key points
        2. Major similarities between the documents
        3. Notable differences or contradictions
        4. An overall assessment of how these documents relate to each other
        
        Comparison:"""
        
        try:
            completion = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that compares documents accurately."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                top_p=1,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error comparing documents: {str(e)}")
            return f"Failed to compare documents: {str(e)}"


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from a PDF file"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""


def display_chat_history():
    """Display the chat history in the main area"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display source citations if available
            if message.get("sources") and message["role"] == "assistant":
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**Document:** {source['doc_id']}, **Chunk:** {source['chunk_id']}")
                        st.markdown(f"*Relevance Score: {source['score']:.2f}*")
                        st.text(source['text'])


def main():
    # Initialize RAG system if not already done
    if not st.session_state.rag_system:
        st.session_state.rag_system = EnhancedRAG()
    
    rag = st.session_state.rag_system
    
    # Sidebar for document upload and control
    with st.sidebar:
        st.header("Document Management")
        
        # Document upload
        uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                # Check if file is already processed
                if file.name in st.session_state.document_store:
                    continue
                
                # Save file temporarily and process
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_path = tmp_file.name
                
                # Extract text from PDF
                text = extract_text_from_pdf(tmp_path)
                
                # Add to document store
                st.session_state.document_store[file.name] = {
                    "path": tmp_path,
                    "text": text,
                    "date_added": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Add to RAG system
                chunks_added = rag.add_document(file.name, text, metadata={"filename": file.name})
                
                # Generate document summary
                summary = rag.generate_document_summary(file.name, text)
                st.session_state.document_summaries[file.name] = summary
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Add to active documents if not already there
                if file.name not in st.session_state.active_documents:
                    st.session_state.active_documents.append(file.name)
            
            progress_bar.empty()
            st.success(f"Processed {len(uploaded_files)} documents")
        
        # Document selection
        st.subheader("Select Active Documents")
        
        for doc_name in st.session_state.document_store.keys():
            doc_selected = st.checkbox(
                doc_name, 
                value=doc_name in st.session_state.active_documents,
                key=f"doc_{doc_name}"
            )
            
            if doc_selected and doc_name not in st.session_state.active_documents:
                st.session_state.active_documents.append(doc_name)
            elif not doc_selected and doc_name in st.session_state.active_documents:
                st.session_state.active_documents.remove(doc_name)
        
        # Document comparison mode
        if len(st.session_state.active_documents) >= 2:
            st.subheader("Document Comparison")
            comparison_mode = st.checkbox("Enable Document Comparison", value=st.session_state.comparison_mode)
            
            if comparison_mode != st.session_state.comparison_mode:
                st.session_state.comparison_mode = comparison_mode
            
            if comparison_mode:
                comparison_topic = st.text_input("Comparison Topic (optional)")
                if st.button("Compare Documents"):
                    with st.spinner("Comparing documents..."):
                        comparison_result = rag.compare_documents(
                            st.session_state.active_documents, 
                            comparison_topic if comparison_topic else None
                        )
                        # Add comparison result to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"### Document Comparison\n\n{comparison_result}"
                        })
                        st.experimental_rerun()
        
        # Model selection
        st.subheader("Model Settings")
        model_option = st.selectbox(
            "Select Groq Model",
            options=["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
            index=0
        )
        rag.model_name = model_option
        
        # Search settings
        st.subheader("Search Settings")
        top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5)
    
    # Main chat interface
    if not st.session_state.document_store:
        st.info("Please upload documents to start chatting.")
    elif not st.session_state.active_documents:
        st.info("Please select at least one document to start chatting.")
    else:
        # Display active documents info
        active_docs_str = ", ".join(st.session_state.active_documents)
        st.info(f"Active documents: {active_docs_str}")
        
        # Display document summaries
        if st.session_state.active_documents:
            with st.expander("Document Summaries"):
                for doc_name in st.session_state.active_documents:
                    if doc_name in st.session_state.document_summaries:
                        st.subheader(doc_name)
                        st.markdown(st.session_state.document_summaries[doc_name])
        
        # Display chat history
        display_chat_history()
        
        # Chat input
        if prompt := st.chat_input("Ask about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Retrieve relevant documents
                        search_results = rag.search(prompt, top_k=top_k)
                        
                        # Get recent chat history for context
                        recent_messages = st.session_state.messages[-10:-1] if len(st.session_state.messages) > 1 else []
                        chat_history = [
                            {"content": msg["content"], "response": msg.get("response", "")} 
                            for msg in recent_messages
                        ]
                        
                        # Generate response
                        response = rag.generate_response(prompt, search_results, chat_history)
                        
                        # Display response
                        st.markdown(response)
                        
                        # Store sources for citation
                        source_info = []
                        for result in search_results:
                            source_info.append({
                                "doc_id": result["doc_id"],
                                "chunk_id": result["chunk_id"],
                                "text": result["text"],
                                "score": result["score"]
                            })
                        
                        # Add to recent sources
                        st.session_state.recent_sources = source_info
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "response": response,
                            "sources": st.session_state.recent_sources
                        })
                    except Exception as e:
                        error_msg = f"Error processing your request: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })


if __name__ == "__main__":
    main()