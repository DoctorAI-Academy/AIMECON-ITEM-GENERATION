from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

def document_processing_example():
    """Demonstrate document loading and Q&A"""
    
    # Create sample document
    sample_text = """
    LangChain is a framework for developing applications powered by language models.
    It provides several key components including LLMs, prompts, chains, memory, and agents.
    LangChain makes it easy to build complex applications that can reason, remember, and act.
    """
    
    # Save sample text to file
    os.makedirs("data", exist_ok=True)
    with open("data/sample.txt", "w") as f:
        f.write(sample_text)
    
    # Load document
    loader = TextLoader("data/sample.txt")
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(splits, embeddings)
    
    # Create Q&A chain
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    # Ask questions
    question = "What are the key components of LangChain?"
    answer = qa_chain.run(question)
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    document_processing_example()