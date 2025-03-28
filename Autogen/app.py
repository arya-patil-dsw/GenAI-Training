import os
import autogen
from typing import Dict, Optional, Union
import json
import argparse

# Set up argument parser for command line inputs
def get_task_from_user():
    parser = argparse.ArgumentParser(description='Run AutoGen with Coder and Tester agents.')
    parser.add_argument('--task', type=str, help='The coding task description', default=None)
    parser.add_argument('--api_key', type=str, help='Your Groq API key', default=None)
    parser.add_argument('--iterations', type=int, help='Maximum number of feedback-revision cycles', default=5)
    
    args = parser.parse_args()
    
    # Interactive mode if task not provided through command line
    if args.task is None:
        print("\n=== AutoGen Coder-Tester System ===")
        print("Please describe the coding task you want to solve:\n")
        lines = []
        print("(Enter your task description, type 'END' on a new line when finished)")
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        task = "\n".join(lines)
    else:
        task = args.task
    
    # Get API key if not provided
    api_key = args.api_key
    if api_key is None:
        api_key = input("\nEnter your Groq API key: ")
    
    return task, api_key, args.iterations

# Define configurations for the agents
def create_groq_config(model: str, api_key: str, temperature: float = 0.1) -> Dict:
    """Create a configuration for the Groq LLM."""
    return {
        "config_list": [
            {
                "model": model,
                "api_key": api_key,
                # Fixed API endpoint URL
                "base_url": "https://api.groq.com/openai/v1",
            }
        ],
        "temperature": temperature,
    }

# Function to initiate a development task
def solve_coding_task(task_description: str, groq_api_key: str, max_iterations: int = 5) -> None:
    """
    Coordinate the multi-agent system to solve a coding task.
    
    Args:
        task_description: Detailed requirements for the coding task
        groq_api_key: API key for Groq
        max_iterations: Maximum number of feedback-revision cycles
    """
    # Configure the agents with different Groq models
    # Using Llama 3 70B for the coder (better reasoning capabilities)
    coder_config = create_groq_config("llama3-70b-8192", groq_api_key)

    # Using Mixtral for the tester (good for evaluation)
    tester_config = create_groq_config("mixtral-8x7b-32768", groq_api_key)

    # Create the coder agent
    coder_agent = autogen.AssistantAgent(
        name="Coder",
        system_message="""You are an expert Python programmer. Your job is to write clean, efficient, 
        and well-documented code based on requirements. Always include docstrings and comments
        to explain your implementation. When you're asked to modify code based on feedback,
        carefully analyze the issues and make appropriate changes. Think step-by-step and
        create robust solutions that handle edge cases.""",
        llm_config=coder_config,
    )

    # Create the tester agent
    tester_agent = autogen.AssistantAgent(
        name="Tester",
        system_message="""You are a software QA specialist focused on Python testing. Your job is to:
        1. Review code for potential bugs and edge cases
        2. Create comprehensive test cases that validate functionality
        3. Provide specific, actionable feedback on code quality and robustness
        4. Think about error handling, input validation, and performance issues
        5. Always provide specific examples when pointing out problems
        
        After testing, clearly state whether the code PASSES or FAILS and why.""",
        llm_config=tester_config,
    )

    # Create a user proxy agent that can execute code
    user_proxy = autogen.UserProxyAgent(
        name="DevLead",
        human_input_mode="NEVER",  # Set to "ALWAYS" if you want to interact manually
        code_execution_config={
            "work_dir": "coding_workspace",
            "use_docker": False,  # Set to True for isolation
            "last_n_messages": 3,
        },
        system_message="""You are the development team lead coordinating between the coder and tester.
        Your role is to:
        1. Execute code provided by the coder
        2. Share execution results with both agents
        3. Track progress and ensure requirements are met
        4. Make final decisions on implementation
        
        Think carefully about the results of code execution and provide helpful context."""
    )
    
    # Start the conversation with the coding task
    user_proxy.initiate_chat(
        coder_agent,
        message=f"""
        # Coding Task
        
        {task_description}
        
        Please implement this functionality. Once you provide the code, 
        our tester will evaluate it and provide feedback if needed.
        """
    )
    
    # Track iterations
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
        
        # Get the last message from the coder
        last_coder_message = user_proxy.chat_messages[coder_agent.name][-1]["content"]
        
        # Send the code to the tester for review
        user_proxy.initiate_chat(
            tester_agent,
            message=f"""
            # Code Review Request
            
            Please review the following code provided by our coder:
            
            {last_coder_message}
            
            Test this code thoroughly and provide specific feedback on:
            1. Correctness - Does it fulfill the requirements?
            2. Edge cases - How does it handle unexpected inputs?
            3. Code quality - Is it well-structured and documented?
            4. Performance - Any inefficiencies?
            
            Execute the code with various test cases and report your findings.
            """
        )
        
        # Get the tester's feedback
        tester_feedback = user_proxy.chat_messages[tester_agent.name][-1]["content"]
        
        # Check if the tester found issues
        if "PASSES" in tester_feedback.upper() and "FAILS" not in tester_feedback.upper():
            # Code passed testing, we can finish
            user_proxy.send(
                f"The implementation has passed all tests after {iterations} iterations! Here's the final solution:\n\n" + 
                last_coder_message,
                user_proxy.chat_messages.keys()
            )
            break
        
        # Send the feedback to the coder for improvements
        user_proxy.initiate_chat(
            coder_agent,
            message=f"""
            # Feedback from Testing
            
            Our tester has provided the following feedback on your implementation:
            
            {tester_feedback}
            
            Please revise your code to address these issues and provide an improved implementation.
            """
        )
    
    if iterations >= max_iterations:
        user_proxy.send(
            f"We've reached the maximum number of iterations ({max_iterations}). The current solution is:\n\n" +
            user_proxy.chat_messages[coder_agent.name][-1]["content"],
            user_proxy.chat_messages.keys()
        )

# Add option to use open-source models locally
def setup_local_models():
    """
    Set up local open-source models as an alternative to Groq.
    This requires additional setup with packages like transformers, llama.cpp, or exllama.
    """
    print("\n=== Local Model Setup ===")
    print("This feature allows you to use open-source models locally instead of Groq API.")
    print("Available options:")
    print("1. Use Hugging Face Transformers")
    print("2. Use llama.cpp")
    print("3. Use ExLlama")
    choice = input("\nEnter your choice (1-3) or press Enter to skip: ")
    
    if not choice:
        return None
    
    if choice == "1":
        print("\nSetting up Hugging Face Transformers...")
        model_name = input("Enter the model name (e.g., 'meta-llama/Llama-2-7b-hf'): ")
        return {
            "type": "huggingface",
            "model": model_name,
            "max_tokens": 2048,
            "temperature": 0.1
        }
    elif choice == "2":
        print("\nSetting up llama.cpp...")
        model_path = input("Enter the path to your model file: ")
        return {
            "type": "llamacpp",
            "model_path": model_path,
            "max_tokens": 2048,
            "temperature": 0.1
        }
    elif choice == "3":
        print("\nSetting up ExLlama...")
        model_path = input("Enter the path to your model directory: ")
        return {
            "type": "exllama",
            "model_path": model_path,
            "max_tokens": 2048,
            "temperature": 0.1
        }
    else:
        print("Invalid choice. Using Groq API.")
        return None

# Main execution block
if __name__ == "__main__":
    print("\n=== AutoGen Coder-Tester System ===")
    print("This system uses the AutoGen framework to create a multi-agent coding system.")
    print("Choose your LLM provider:")
    print("1. Groq API (recommended for performance)")
    print("2. Local open-source models")
    
    provider_choice = input("\nEnter your choice (1-2): ")
    
    if provider_choice == "2":
        local_model_config = setup_local_models()
        if local_model_config:
            print("\nUsing local models. Setup requirements:")
            print("1. Install required packages:")
            if local_model_config["type"] == "huggingface":
                print("   pip install transformers torch")
            elif local_model_config["type"] == "llamacpp":
                print("   pip install llama-cpp-python")
            elif local_model_config["type"] == "exllama":
                print("   pip install exllama")
            
            print("\n2. Download model files as needed")
            print("\n3. Modify the script to use the local_model_config")
            print("\nExiting now. Please modify the script to use local models.")
            exit()
    
    # Get task description from user
    task_description, groq_api_key, max_iterations = get_task_from_user()
    
    # Print some information about the process
    print(f"\nStarting AutoGen with Coder and Tester agents...")
    print(f"Maximum iterations: {max_iterations}")
    print(f"Task: {task_description[:50]}..." if len(task_description) > 50 else f"Task: {task_description}")
    print("\nProcessing... This may take a few minutes depending on the complexity of the task.\n")
    
    # Execute the task
    solve_coding_task(task_description, groq_api_key, max_iterations)