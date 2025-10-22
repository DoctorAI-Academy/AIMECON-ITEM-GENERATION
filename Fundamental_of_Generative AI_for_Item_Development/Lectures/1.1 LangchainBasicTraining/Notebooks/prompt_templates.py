from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def prompt_template_example():
    """Demonstrate prompt templates"""
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that explains {topic} to {audience}."),
        ("human", "{question}")
    ])
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Create chain
    chain = prompt | llm
    
    # Use the chain
    response = chain.invoke({
        "topic": "machine learning",
        "audience": "beginners",
        "question": "What is supervised learning?"
    })
    
    print("Response:", response.content)
    return response

if __name__ == "__main__":
    prompt_template_example()