from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

class EducationalQuestion(BaseModel):
    """Structure for educational questions"""
    question: str = Field(description="The main question text")
    options: List[str] = Field(description="List of answer options")
    correct_answer: str = Field(description="The correct answer")
    explanation: str = Field(description="Explanation of why the answer is correct")
    difficulty: str = Field(description="Difficulty level: easy, medium, hard")

def custom_educational_chain():
    """Create custom chain for generating educational content"""
    
    # Set up output parser
    parser = PydanticOutputParser(pydantic_object=EducationalQuestion)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Create an educational question about {subject} for {grade_level} students.
        The question should be at {difficulty} difficulty level.
        
        Focus on: {learning_objective}
        
        {format_instructions}
        """
    )
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create chain
    chain = prompt | llm | parser
    
    # Generate question
    result = chain.invoke({
        "subject": "8th grade algebra",
        "grade_level": "8th grade",
        "difficulty": "medium",
        "learning_objective": "solving one-step linear equations with addition and subtraction",
        "format_instructions": parser.get_format_instructions()
    })
    
    print("Generated Educational Question:")
    print(f"Question: {result.question}")
    print(f"Options: {result.options}")
    print(f"Correct Answer: {result.correct_answer}")
    print(f"Explanation: {result.explanation}")
    print(f"Difficulty: {result.difficulty}")
    
    return result

if __name__ == "__main__":
    custom_educational_chain()