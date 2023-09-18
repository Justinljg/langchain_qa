# from langchain.document_loaders import PyPDFDirectoryLoader
from pypdf import PdfReader
from langchain.vectorstores.redis import Redis
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from io import BytesIO

from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool


def get_pdf_text(pdf_docs: list[BytesIO]) -> str:
    """Extracts string from streamlit UploadedFile.

    Args:
        pdf_docs (list[BytesIO]): List containing multiple streamlit UploadedFiles

    Returns:
        str: Text data
    """

    text_data = ""
    for pdf in pdf_docs:
        doc_loader = PdfReader(pdf)
        for page in doc_loader.pages:
            text_data += page.extract_text()

    return text_data

def read(dirpath: str) -> list:
    """
    Read data from PDF files in a directory.

    Args:
        dirpath (str): The path to the directory containing PDF files.

    Returns:
        list: A list of PDF data.

    This function reads data from the specified directory, which contains PDF files.
    It uses the PyPDFDirectoryLoader to load the PDF data.
    """
    loader = PyPDFDirectoryLoader(dirpath)
    data = loader.load()
    return data

def text_chunks(data: list) -> list:
    """
    Split PDF data into text chunks.

    Args:
        data (list): A list of PDF data.

    Returns:
        list: A list of text chunks.

    This function takes PDF data and splits it into smaller text chunks.
    It uses the RecursiveCharacterTextSplitter to perform the splitting.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(data)
    return chunks

def content_extract(docs: list) -> tuple:
    """
    Extract text and metadata from text chunks.

    Args:
        docs (list): A list of text chunks.

    Returns:
        tuple: A tuple containing lists of texts and metadatas.

    This function extracts text and metadata from a list of text chunks.
    It assumes that each chunk is an object with 'page_content' and 'metadata' attributes.
    """
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    return texts, metadatas

def load_vector(texts: list) -> tuple:
    """
    Load vectors and return relevant data.

    Args:
        texts (list): A list of texts.
        metadatas (list): A list of metadata.
        api_key (str): An API key for loading vectors.

    Returns:
        tuple: A tuple containing loaded vectors and keys.

    This function loads vectors based on the provided texts, metadata, and API key.
    It uses the OpenAIEmbeddings and Redis modules to perform the loading.
    """
    embeddings = OpenAIEmbeddings()
    rds, keys = Redis.from_texts_return_keys(texts, embeddings, redis_url="redis://redis-db:6379")
    return rds

def chain(vectorstore, temperature: float = 0.1, 
          model_name: str = "gpt-3.5-turbo") -> AgentExecutor:
    """
    Create and configure an agent for a conversational task.

    Args:
        model_name (str): The name of the GPT-3.5 model to use.
        temperature (float): The temperature parameter for text generation.
        openai_organization (str): The organization ID for OpenAI (if applicable).

    Returns:
        AgentExecutor: An executor for the configured agent.

    This function sets up an agent with the specified model, temperature, and organization.
    It creates a retriever tool and memory for conversation history.
    The agent is configured to answer questions and use available tools for reference.
    The executor allows you to interact with the agent.
    """
    
    # Create the language model (llm)
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    # Create a retriever for document retrieval
    retriever = vectorstore.as_retriever(k=3)

    # Create a tool for document retrieval
    tool = create_retriever_tool(
        retriever, 
        'doc_retrieve',
        'This tool retrieves relevant documents.'
    )

    # Define the memory for conversation history
    memory_key = "history"
    memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

    # Create a system message
    system_message = SystemMessage(
        content=(
            "Do your best to answer the questions from the input delimited by triple backticks"
            "Feel free to use any tools available to look up "
            "relevant information, only if necessary."
        )
    )

    # Create a prompt for the agent
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )

    # Create the agent with tools and prompt
    agent = OpenAIFunctionsAgent(llm=llm, tools=[tool], prompt=prompt)

    # Create and configure the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=[tool], memory=memory, verbose=True,
                                    return_intermediate_steps=True)
    
    return agent_executor


