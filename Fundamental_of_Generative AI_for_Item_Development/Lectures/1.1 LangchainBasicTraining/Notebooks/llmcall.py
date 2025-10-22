# File: src/examples/01_basic_llm.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def basic_llm_example():
    """Basic LLM interaction example"""
    
    # Initialize the LLM
    # You can change the llm type by replacing the ChatOpenAI with ChatGroq, Chat...
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY") 
    )
    
    # Create a message
    message = HumanMessage(content="What is LangChain?")
    
    # Get response
    response = llm.invoke([message])
    
    print("Response:", response.content)
    return response

if __name__ == "__main__":
    basic_llm_example()