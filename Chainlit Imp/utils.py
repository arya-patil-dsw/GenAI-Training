# utils.py
import os
from dotenv import load_dotenv

def load_config():
    """Load configuration from .env file."""
    load_dotenv()
    config = {
        "api_key": os.getenv("GROQ_API_KEY"),
        "model": os.getenv("GROQ_MODEL")
    }
    if not config["api_key"] or not config["model"]:
        raise ValueError("GROQ_API_KEY or GROQ_MODEL not found in .env")
    return config

def format_expense_data(income, expenses):
    """Format user input into a structured string for agents."""
    expense_str = "\n".join([f"- {cat}: ${amt}" for cat, amt in expenses.items()])
    return f"Monthly Income: ${income}\nExpenses:\n{expense_str}"