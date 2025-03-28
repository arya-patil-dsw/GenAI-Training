import os
import logging
from typing import List, Dict, Any
from datetime import datetime

import litellm
from litellm import Router
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file (optional)
load_dotenv()

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Output to console only
)
logger = logging.getLogger("litellm-proxy")

# Initialize FastAPI app
app = FastAPI(title="LiteLLM Proxy Demo")

# Allow all origins for simplicity (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model configurations (including Gemini)
model_list = [
    # Gemini model (requires API key)
    {
        "model_name": "gemini-2.0-flash",
        "litellm_params": {
            "model": "gemini/gemini-2.0-flash",
            "api_key": os.getenv("GEMINI_API_KEY", "AIzaSyBKEkK1iNCe7S3rKAhh0emhNkRdP2C2XOk")
        },
    },
    # Groq model (requires API key)
    {
        "model_name": "llama3-70b-8192",
        "litellm_params": {
            "model": "groq/llama3-70b-8192",
            "api_key": os.getenv("GROQ_API_KEY", "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")
        },
    },
    # Ollama model (local, no API key needed if running locally)
    {
        "model_name": "llama3.2:1b",
        "litellm_params": {
            "model": "ollama/llama3.2:1b",
            "api_base": "http://localhost:11434"  # Assumes Ollama is running locally
        },
    },
]

# Initialize LiteLLM Router (minimal parameters)
router = Router(
    model_list=model_list,
    routing_strategy="least-busy",  # Load balancing across models
)
logger.info("LiteLLM Router initialized with least-busy routing")

# Simple API key verification
async def verify_api_key(request: Request, api_key: str = Header(None, alias="Authorization")):
    """Basic API key check (optional for demo)"""
    expected_key = os.getenv("PROXY_API_KEY", "sk-demo-key")
    if api_key is None:
        logger.warning("No API key provided, allowing request for demo purposes")
        return None
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]
    if api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "LiteLLM Proxy is running", "timestamp": datetime.now().isoformat()}

@app.get("/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """List available models"""
    models = [
        {"id": model["model_name"], "object": "model", "provider": model["litellm_params"]["model"].split("/")[0]}
        for model in model_list
    ]
    return {"object": "list", "data": models}

@app.post("/chat/completions")
async def chat_completions(request: Request, api_key: str = Depends(verify_api_key)):
    """Handle chat completion requests with streaming support"""
    try:
        body = await request.json()
        model = body.get("model", None)  # Default to None for router to pick
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        temperature = body.get("temperature", 0.7)
        max_tokens = body.get("max_tokens", 100)

        logger.info(f"Request: model={model or 'router default'}, messages={messages}, stream={stream}")

        if stream:
            completion_stream = await router.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            return StreamingResponse(completion_stream, media_type="text/event-stream")
        else:
            completion = await router.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            logger.info(f"Response: {completion}")
            return completion

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/embeddings")
async def create_embeddings(request: Request, api_key: str = Depends(verify_api_key)):
    """Generate embeddings (demo with a single model)"""
    try:
        body = await request.json()
        model = body.get("model", "llama3.2:1b")  # Default to local model (Gemini may not support embeddings)
        input_text = body.get("input", ["Hello, world!"])
        
        embedding = await router.aembedding(model=model, input=input_text)
        logger.info(f"Embedding generated for input: {input_text}")
        return embedding
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

# Run the server
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting LiteLLM Proxy Server...")
    logger.info("To test, use: curl http://localhost:8000/ or the client script")
    logger.info("Models available: gemini-2.0-flash (Gemini), llama3-70b-8192 (Groq), llama3.2:1b (Ollama)")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )