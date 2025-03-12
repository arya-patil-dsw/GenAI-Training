import os
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import io
import requests
import chainlit as cl

# --------------------------
# Database Initialization
# --------------------------
def init_db():
    conn = sqlite3.connect("finance.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            amount REAL,
            date TEXT
        )
    """)
    conn.commit()
    conn.close()

# --------------------------
# Expense Data Functions
# --------------------------
def insert_expense(category, amount, date):
    conn = sqlite3.connect("finance.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO expenses (category, amount, date) VALUES (?, ?, ?)",
        (category, amount, date)
    )
    conn.commit()
    conn.close()

def get_expense_summary():
    conn = sqlite3.connect("finance.db")
    df = pd.read_sql_query(
        "SELECT category, SUM(amount) as total FROM expenses GROUP BY category",
        conn
    )
    conn.close()
    return df

def plot_expense_summary():
    """Generate a bar chart of expenses by category and return image bytes."""
    df = get_expense_summary()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(df['category'], df['total'], color='skyblue')
    ax.set_title("Expenses by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Total Amount")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

# --------------------------
# AI Agent: Groq API via HTTP Request
# --------------------------
def finance_advice(user_input):
    groq_api_key = os.environ.get("GROQ_API_KEY")
    groq_model = os.environ.get("GROQ_MODEL")
    if not groq_api_key or not groq_model:
        return "Groq API key or model not configured in environment variables."
    
    # NOTE: Adjust the URL based on the official Groq API documentation.
    url = "https://api.groq.ai/v1/chat"  
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": groq_model,
        "prompt": user_input,
        "max_tokens": 150
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            # Adjust based on the API’s JSON structure.
            return result.get("response", "No response from API.")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"

# --------------------------
# CSV File Upload Processing (Optional)
# --------------------------
def process_csv(file_bytes):
    """
    Process a CSV file containing expense data.
    Expected CSV columns: category, amount, date
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    for _, row in df.iterrows():
        insert_expense(row['category'], row['amount'], row['date'])
    return f"Processed {len(df)} expense records successfully."

# --------------------------
# Chainlit UI Integration using Decorators
# --------------------------
@cl.on_chat_start
async def start():
    init_db()
    await cl.Message(content="Welcome to your Personal Finance Advisor!").send()
    await cl.Message(content="Type your finance question or 'show summary' to view your expenses.").send()

@cl.on_message
async def handle_message(message):
    # Extract text from the Chainlit Message object
    user_text = message.content.strip()
    command = user_text.lower()

    if command in ["show summary", "show expenses", "summary"]:
        df = get_expense_summary()
        summary = df.to_string(index=False)
        await cl.Message(content=f"Expense Summary:\n{summary}").send()
        img_bytes = plot_expense_summary().read()
        await cl.Message(content="Expense Chart:", image=img_bytes).send()
    else:
        # Forward any other query to the Groq API for financial advice.
        advice = finance_advice(user_text)
        await cl.Message(content=advice).send()
