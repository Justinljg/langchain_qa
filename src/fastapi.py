from .langchain_utils import (get_pdf_text, text_chunks, content_extract,load_vector, chain)

import json
import logging
import os
from typing import Annotated
from typing import List
from langchain.embeddings import OpenAIEmbeddings

import uvicorn
from fastapi import FastAPI, File, Header, HTTPException, UploadFile, Form
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel
import os
import tempfile
import io

from .general_utils import setup_logging

# Pydantic models
class Data(BaseModel):
    query: str
    temperature: float

# Initial Variables
rds = None
agent_executor = None

# Initialise fastapi app
app = FastAPI()

# Setup logger
logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")
setup_logging(logging_config_path="conf/base/logging.yaml")

@app.get("/")
def redirect_swagger():
    response = RedirectResponse("/docs")
    return response

@app.post("/upload")
async def upload(files: Annotated[List[UploadFile], File()],
                 openai_api_key: Annotated[str, Header()]
) -> None:

    logger.info("Loading API key...")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    logger.info("Loading pdf files...")
    byte_files = [io.BytesIO(await f.read()) for f in files]

    logger.info("Reading pdf files...")
    text = get_pdf_text(byte_files)
    # Create a temporary directory to store uploaded files
    # with tempfile.TemporaryDirectory() as temp_dir:
    #     temp_data_dir = os.path.join(temp_dir, "temp_data")
    #     os.makedirs(temp_data_dir, exist_ok=True)

    #     for f in files:
    #         file_bytes = await f.read()
    #         file_name = f.filename
    #         file_path = os.path.join(temp_data_dir, file_name)

    #         # Save the uploaded file to the temporary directory
    #         with open(file_path, "wb") as temp_file:
    #             temp_file.write(file_bytes)

        # logger.info("Reading PDF files...")
        # data =read(temp_data_dir)

    logger.info("Converting PDF files into Chunks...")
    chunks = text_chunks(text)
        # logger.info("Extracting Text & Metadatas...")
        # texts, metadatas = content_extract(docs)
    logger.info("Saving to Vectorstore")
    # Create vector store
    global rds
    if rds is None:
        rds = load_vector(chunks)
    else:
        embeddings = OpenAIEmbeddings()
        keys = rds.add_texts(texts=chunks)

    # # Temp_dir removed after exiting with
    # logger.info("Temporary Files Removed...")

# Chat
@app.post("/chat")
async def chat(data: Data) -> None:

    logger.info("Checking for existing vector store...")
    if rds is None:
        raise HTTPException(
            status_code=400,
            detail="Documents not uploaded or API key is not working.",
        )

    logger.info("Checking for Agent Executor...")
    global agent_executor
    if agent_executor is None:
        agent_executor = chain(
            vectorstore=rds, temperature=data.temperature
        )

    logger.info("Performing QA...")
    result = agent_executor({"input": f"```{data.query}```"})
    json_payload = json.dumps(result['output'])

    return Response(content=json_payload, media_type="application/json")


if __name__ == "__main__":
    uvicorn.run(app, debug=True)