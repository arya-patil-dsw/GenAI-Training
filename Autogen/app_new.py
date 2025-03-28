import gradio as gr
from autogen import AssistantAgent, UserProxyAgent
from autogen import Cache

def run_assistant(message):
    # Define the configuration for the Code Llama model
    config_list = [
        {
            "model": "mistral",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "seed": None
        }
    ]

    # Create an AssistantAgent with the Code Llama model
    assistant = AssistantAgent("assistant", llm_config={"config_list": config_list, "seed": None})

    # Create a UserProxyAgent that can execute code
    user_proxy = UserProxyAgent(
        "user_proxy",
        is_termination_msg = lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        max_consecutive_auto_reply=10,
        code_execution_config={"work_dir": "coding", "use_docker": False},
        human_input_mode="NEVER"
        )

    # Initiate a chat between the user proxy and the assistant
    response = user_proxy.initiate_chat(assistant, message=message)
    summary = response.summary if hasattr(response, 'summary') else "No summary available."
    
    return summary

iface = gr.Interface(fn=run_assistant, inputs="text", outputs="text", title="AutoGen Assistant")
iface.launch()