import streamlit as st
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import SerpAPIWrapper
from langchain_experimental.sql import SQLDatabaseChain
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase  # Correct import
from sqlalchemy import create_engine

# Streamlit UI
st.title("AI Agent for Business Data Analysis & Web Search")
st.write("Query business databases or search the web for insights!")

# Initialize session state for chat history if not already initialized
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Corrected database connection
db_uri = "sqlite:///business.db"  # Change to your database connection
engine = create_engine(db_uri)
database = SQLDatabase(engine)  # Use SQLDatabase, not SQLDatabaseChain

# LLM Setup
llm = ChatGroq(model_name="llama-3.2-3b-preview", groq_api_key="vPL6s1r7zAymzbbHeJptTU9BEnDALzNOItX7sGha")

# Web Search Tool
serp_api = SerpAPIWrapper(serpapi_api_key="9c4a1b8038c7df869f3ccd1c712d7deec10f9da794950a49075c6568d1956936")

def query_database(query):
    """Uses SQLDatabaseChain to query structured business data."""
    sql_chain = SQLDatabaseChain.from_llm(llm=llm, db=database, verbose=True)
    return sql_chain.run(query)

def search_and_analyze(query):
    """Performs a web search and analyzes the retrieved data."""
    search_results = serp_api.run(query)
    return f"Web Search Results:\n{search_results}\n\nFurther analysis is required for deeper insights."

# Agent with SQL & Web Search Tools
tools = [
    Tool(name="Web Search", func=search_and_analyze, description="Search the web and analyze information"),
    Tool(name="SQL Database Query", func=query_database, description="Query structured business data")
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
    if "database" in user_input.lower() or "query" in user_input.lower():
        response = query_database(user_input)
    else:
        response = search_and_analyze(user_input)
    
    # Append interaction to chat history
    st.session_state.chat_history.append((user_input, response))
    
    # Display latest user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Display AI response
    with st.chat_message("assistant"):
        st.write(response)
