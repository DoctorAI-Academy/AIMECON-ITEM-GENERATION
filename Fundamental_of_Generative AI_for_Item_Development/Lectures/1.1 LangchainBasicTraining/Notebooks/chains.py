from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def simple_chain_example():
    """Demonstrate a simple chain with output parsing"""
    
    # Create components
    prompt = ChatPromptTemplate.from_template(
        "Create a {question_type} question about {subject} for {grade_level} students."
    )
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
    output_parser = StrOutputParser()
    
    # Chain components together
    chain = prompt | llm | output_parser
    
    # Use the chain
    result = chain.invoke({
        "question_type": "multiple choice",
        "subject": "algebra",
        "grade_level": "8th grade"
    })
    
    print("Generated Question:")
    print(result)
    return result

if __name__ == "__main__":
    simple_chain_example()