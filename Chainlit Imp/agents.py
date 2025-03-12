# agents.py
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen_ext.models.grok import GrokClient
from utils import format_expense_data

# Load GROQ configuration
config = {
    "config_list": [{
        "model": "llama3-8b-8192",  # Will be overridden by .env
        "api_key": None,           # Will be overridden by .env
        "base_url": "https://api.grok.xai.com/v1",  # Assuming GROQ uses this
        "api_type": "grok"
    }]
}

def initialize_agents():
    """Initialize the multi-agent system."""
    groq_config = load_config()
    config["config_list"][0]["model"] = groq_config["model"]
    config["config_list"][0]["api_key"] = groq_config["api_key"]

    # Model client for GROQ
    model_client = GrokClient(model=groq_config["model"], api_key=groq_config["api_key"])

    # User Proxy Agent (interacts with the user via Chainlit)
    user_proxy = UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",  # Chainlit will handle input
        max_consecutive_auto_reply=5,
        code_execution_config=False,
        description="Represents the user and initiates tasks."
    )

    # Data Collector Agent
    data_collector = AssistantAgent(
        name="DataCollector",
        model_client=model_client,
        system_message="You collect and validate financial data from the user. Ask for income and expenses if not provided.",
        description="Collects and structures financial data."
    )

    # Analyst Agent
    analyst = AssistantAgent(
        name="Analyst",
        model_client=model_client,
        system_message="You analyze financial data, calculate spending patterns, and identify savings opportunities.",
        description="Analyzes the collected data."
    )

    # Advisor Agent
    advisor = AssistantAgent(
        name="Advisor",
        model_client=model_client,
        system_message="You provide budgeting advice and suggestions based on the analysis.",
        description="Gives actionable financial advice."
    )

    return user_proxy, data_collector, analyst, advisor

def process_financial_data(user_proxy, data_collector, analyst, advisor, income, expenses):
    """Run the multi-agent workflow."""
    formatted_data = format_expense_data(income, expenses)
    
    # Step 1: Data Collector validates and passes data
    user_proxy.initiate_chat(
        data_collector,
        message=f"Here’s the financial data:\n{formatted_data}\nPlease validate and forward it."
    )
    
    # Step 2: Analyst processes the data
    data_collector.send(
        message=f"Validated data:\n{formatted_data}\nPlease analyze it.",
        recipient=analyst
    )
    
    # Step 3: Advisor provides suggestions
    analyst.send(
        message="Analysis complete. Please provide budgeting advice.",
        recipient=advisor
    )
    
    # Get the final response from the advisor
    final_response = advisor.last_message()["content"]
    return final_response