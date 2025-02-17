import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.utilities import SerpAPIWrapper
import os

# Streamlit UI
st.title("AI Agent for Business Data Analysis & Web Search")
st.write("Upload business data or search the web for insights!")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def analyze_business_data(file, query):
    """Analyzes business data based on user query."""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    if "summary" in query.lower():
        return df.describe().to_string()
    elif "correlation" in query.lower():
        return df.corr().to_string()
    elif "missing" in query.lower() or "null" in query.lower():
        return df.isnull().sum().to_string()
    elif "trend" in query.lower() or "growth" in query.lower():
        return df.diff().mean().to_string()
    else:
        return "Please ask about summary, correlation, missing values, or trends in the dataset."


# Web Search Tool
serp_api = SerpAPIWrapper(serpapi_api_key="9c4a1b8038c7df869f3ccd1c712d7deec10f9da794950a49075c6568d1956936")

def search_and_analyze(query):
    """Performs a web search and analyzes the retrieved data."""
    search_results = serp_api.run(query)
    return f"Web Search Results:\n{search_results}\n\nFurther analysis is required for deeper insights."
# LLM Setup
llm = ChatGroq(model_name="llama-3.2-3b-preview", groq_api_key="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")

# Agent with Business Data Analysis & Web Search
tools = [
    Tool(name="Web Search", func=serp_api.run, description="Search the web for real-time information"),
    Tool(name="Business Data Analysis", func=analyze_business_data, description="Analyze the buisiness data using pandas and give the insights correctly,always try to give me the most proper answer")
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Display chat history with proper chat formatting
for message in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(message[0])
    with st.chat_message("assistant"):
        st.write(message[1])

# User Input
user_input = st.chat_input("Enter your message")
if user_input:
    if uploaded_file:
        response = analyze_business_data(uploaded_file, user_input)
    else:
        response = agent.run(user_input)
    
    # Append interaction to chat history
    st.session_state.chat_history.append((user_input, response))
    
    # Display latest user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Display AI response
    with st.chat_message("assistant"):
        st.write(response)



