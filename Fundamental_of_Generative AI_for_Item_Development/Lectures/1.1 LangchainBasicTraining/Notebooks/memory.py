from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()

def memory_example():
    """Demonstrate conversation memory"""
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Create memory
    memory = ConversationBufferMemory(return_messages=True)
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Have a conversation
    response1 = conversation.predict(input="My name is John and I'm learning about AI.")
    print("Response 1:", response1)
    
    response2 = conversation.predict(input="What did I say my name was?")
    print("Response 2:", response2)
    
    # Check memory
    print("\nMemory contents:")
    print(memory.buffer)

if __name__ == "__main__":
    memory_example()