# travel_planner.py
import chainlit as cl
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, Tool
from langchain.agents import initialize_agent, AgentType
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import requests
from datetime import datetime, timedelta
import json

# Set up environment variables (add these to your .env file)
GROQ_API_KEY = "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"
WEATHER_API_KEY = "d0f81d43815845a9a34122823251203"  # Get from weatherapi.com

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="mixtral-8x7b-32768",
    temperature=0.5
)

# Weather API Tool (Using WeatherAPI.com)
def get_weather_forecast(location, days=3):
    """
    Fetches a 3-day weather forecast from WeatherAPI.com for a given location.
    Returns formatted string with date, avg temperature, and condition.
    """
    base_url = "http://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": WEATHER_API_KEY,
        "q": location,  # Location can be city name, lat/lon, etc.
        "days": days,   # Number of forecast days
        "aqi": "no",    # Exclude air quality data
        "alerts": "no"  # Exclude weather alerts
    }
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        # Check for API errors
        if "error" in data:
            return f"Error fetching weather: {data['error']['message']}"
        
        # Parse forecast data
        forecast = []
        for day in data["forecast"]["forecastday"]:
            date = day["date"]
            temp = day["day"]["avgtemp_c"]  # Average temperature in Celsius
            weather = day["day"]["condition"]["text"]  # Weather description
            forecast.append(f"{date}: {temp}°C, {weather}")
        return "\n".join(forecast)
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

# PDF Processing Tool
async def process_travel_brochure(file_path):
    """
    Processes a travel brochure PDF and returns its content as a string.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)
        
        content = "\n".join([doc.page_content for doc in docs])
        return content
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

# Define Tools
tools = [
    Tool(
        name="WeatherForecast",
        func=get_weather_forecast,
        description="Get weather forecast for a location for the next 3 days using WeatherAPI.com"
    )
]

# Prompt Template
travel_planner_prompt = PromptTemplate(
    input_variables=["input", "weather", "brochure_content"],
    template="""
    You are an expert travel planner assistant. Create a detailed travel plan based on:
    1. User request: {input}
    2. Weather forecast: {weather}
    3. Travel brochure information: {brochure_content}
    
    Provide a day-by-day itinerary including:
    - Recommended activities considering the weather
    - Travel tips
    - Suggested accommodations
    - Any relevant information from the brochure
    
    If any information is missing, make reasonable assumptions and explain them.
    Return the plan in markdown format.
    """
)

# Chainlit Setup
@cl.on_chat_start
async def start():
    """Initializes the agent and sends a welcome message."""
    # Initialize agent and chain
    weather_chain = LLMChain(llm=llm, prompt=travel_planner_prompt)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # Store in session
    cl.user_session.set("agent", agent)
    cl.user_session.set("weather_chain", weather_chain)
    
    # Welcome message
    await cl.Message(
        content="Welcome to your Travel Planner Assistant! I can help you plan your trip using WeatherAPI.com forecasts and travel brochures. Upload a travel brochure (PDF) or tell me where you're planning to go!"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handles user messages and generates travel plans."""
    agent = cl.user_session.get("agent")
    weather_chain = cl.user_session.get("weather_chain")
    brochure_content = cl.user_session.get("brochure_content", "No brochure provided")
    
    # Handle file upload
    if message.elements:
        for element in message.elements:
            if element.mime == "application/pdf":
                brochure_content = await process_travel_brochure(element.path)
                cl.user_session.set("brochure_content", brochure_content)
                await cl.Message(content="Brochure processed successfully! Now tell me about your trip.").send()
                return
    
    # Extract location from message (simple approach)
    location = message.content.strip()
    weather = get_weather_forecast(location)
    
    # Generate travel plan
    response = await weather_chain.arun(
        input=message.content,
        weather=weather,
        brochure_content=brochure_content
    )
    
    await cl.Message(content=response).send()

# Command to run: chainlit run travel_planner.py -w