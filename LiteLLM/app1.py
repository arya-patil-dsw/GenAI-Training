# LiteLLM Proxy Client Application
# This client demonstrates how to interact with the LiteLLM proxy server

import os
import json
import time
import argparse
import requests
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure client settings
PROXY_BASE_URL = os.getenv("LITELLM_PROXY_URL", "http://localhost:8000")
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "sk-demo-key")  # Match app.py default

class LiteLLMClient:
    def __init__(self, base_url: str, api_key: str):
        """Initialize the LiteLLM proxy client"""
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def list_models(self) -> Dict[str, Any]:
        """List all available models from the proxy"""
        url = f"{self.base_url}/models"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            raise Exception(f"Error listing models: {response.text}")
        
        return response.json()
    
    def chat_completion(
        self, 
        model: Optional[str],  # Allow None for router default
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        user: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a chat completion request to the proxy"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        
        if model:
            payload["model"] = model
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if user:
            payload["user"] = user
        
        if stream:
            # Handle streaming responses
            with requests.post(url, json=payload, headers=self.headers, stream=True) as response:
                if response.status_code != 200:
                    raise Exception(f"Error in streaming completion: {response.text}")
                
                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]  # Remove 'data: ' prefix
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                if content:
                                    print(content, end='', flush=True)
                            except json.JSONDecodeError:
                                print(f"Error decoding: {data}")
                print()  # Add newline at the end
                return {"status": "Stream completed"}
        else:
            # Handle regular responses
            response = requests.post(url, json=payload, headers=self.headers)
            
            if response.status_code != 200:
                raise Exception(f"Error in completion: {response.text}")
            
            return response.json()
    
    def create_embedding(self, model: str, input_text: List[str]) -> Dict[str, Any]:
        """Create embeddings for the given texts"""
        url = f"{self.base_url}/embeddings"
        payload = {
            "model": model,
            "input": input_text
        }
        
        response = requests.post(url, json=payload, headers=self.headers)
        
        if response.status_code != 200:
            raise Exception(f"Error creating embeddings: {response.text}")
        
        return response.json()

    def benchmark_models(self, models: List[str], prompt: str, runs: int = 3):
        """Benchmark different models with the same prompt"""
        results = {}
        
        for model in models:
            print(f"\nBenchmarking model: {model}")
            model_times = []
            
            for i in range(runs):
                messages = [{"role": "user", "content": prompt}]
                
                start_time = time.time()
                response = self.chat_completion(model, messages)
                end_time = time.time()
                
                elapsed = end_time - start_time
                model_times.append(elapsed)
                
                # Extract response text
                response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "No response")
                token_count = response.get("usage", {}).get("total_tokens", 0)
                
                print(f"Run {i+1}: {elapsed:.2f}s, {token_count} tokens")
            
            avg_time = sum(model_times) / len(model_times)
            results[model] = {
                "average_time": avg_time,
                "runs": model_times
            }
            
            print(f"Average response time: {avg_time:.2f}s")
        
        return results

    def test_load_balancing(self, prompt: str, requests: int = 5):
        """Test load balancing by sending multiple requests without specifying a model"""
        print(f"\nTesting load balancing across available models")
        
        messages = [{"role": "user", "content": prompt}]
        
        for i in range(requests):
            try:
                print(f"Request {i+1}: ", end="", flush=True)
                response = self.chat_completion(None, messages)  # No model specified, uses router
                
                # Extract response info
                model_used = response.get("model", "unknown")
                print(f"Used model: {model_used}")
            except Exception as e:
                print(f"Error: {str(e)}")

def main():
    """Main function to demonstrate the LiteLLM proxy client"""
    parser = argparse.ArgumentParser(description="LiteLLM Proxy Client")
    
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--chat", action="store_true", help="Send a chat completion request")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Model to use (optional)")
    parser.add_argument("--prompt", type=str, default="Explain quantum computing in simple terms", help="Prompt to send")
    parser.add_argument("--stream", action="store_true", help="Enable streaming response")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark models")
    parser.add_argument("--load-balance", action="store_true", help="Test load balancing")
    
    args = parser.parse_args()
    
    # Initialize the client
    client = LiteLLMClient(PROXY_BASE_URL, PROXY_API_KEY)
    
    # Handle command line arguments
    try:
        if args.list_models:
            # List available models
            models = client.list_models()
            print("\nAvailable Models:")
            for model in models["data"]:
                print(f"- {model['id']} ({model.get('object', 'model')})")
        
        elif args.chat:
            # Send a chat completion request
            messages = [{"role": "user", "content": args.prompt}]
            print(f"\nSending prompt to {args.model if args.model else 'router default'}: \"{args.prompt}\"")
            
            if args.stream:
                print("\nResponse (streaming):")
                client.chat_completion(args.model, messages, stream=True)
            else:
                print("\nResponse:")
                response = client.chat_completion(args.model, messages)
                content = response["choices"][0]["message"]["content"]
                print(content)
                print(f"\nTokens used: {response.get('usage', {}).get('total_tokens', 'unknown')}")
        
        elif args.benchmark:
            # Benchmark models
            models_to_test = ["gemini-2.0-flash", "llama3-70b-8192", "llama3.2:1b"]
            results = client.benchmark_models(models_to_test, args.prompt)
            
            print("\nBenchmark Summary:")
            for model, stats in results.items():
                print(f"- {model}: {stats['average_time']:.2f}s average")
        
        elif args.load_balance:
            # Test load balancing
            client.test_load_balancing(args.prompt, 5)
        
        else:
            print("No command specified. Use --help for available commands.")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()