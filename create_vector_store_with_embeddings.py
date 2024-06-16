# Chatbot to chat with the LangChain docs!
# Uses ChatGPT, LangChain, Chroma, FastAPI, Streamlit

# This file populates the vector store with the extracted 
# LangChain docs and their embeddings.

import chromadb
from chromadb import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma
import json
from typing import List, Dict, Tuple

def read_docs_from_file(filename: str) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
    """
    Read LangChain document texts, IDs, and source URL metadata from a file.

    Args:
        filename (str): Filename to read from.
    
    Returns:
        Tuple[List[str], List[str], List[Dict[str, str]]]: A tuple containing a list of document texts, IDs, and metadata source URLs.
    """
    documents = list()
    docs = list()
    doc_ids = list()
    metadata = list()
    # Open a file, process each line as JSON, and add the line to a list
    with open(filename, "r") as f:
        for line in f:
            documents.append(json.loads(line))

    # Parse each line of JSON into three lists: docs (list str), metadata (list dict), and doc IDs (list str)
    for document in documents:
        for uuid, inner_dict in document.items():
            docs.append(inner_dict["text"])
            metadata.append({"url": inner_dict["url"]})
            doc_ids.append(uuid)

    return docs, doc_ids, metadata

def add_docs_to_chroma_db(docs: List[str], doc_ids: List[str], metadata: List[Dict[str, str]]) -> Chroma:
    """
    Load a tuple of doc texts, doc IDs, and metadata source URLs into a Chroma database with OpenAI embeddings.

    Args:
        docs (List[str]): A list of text snippets.
        doc_ids (List[str]): A list of IDs for the text snippets.
        metadata (List[Dict[str, str]]): A list of dictionaries where the key is "url" and the value is the source URL for the text snippet.
    
    Returns:
        Chroma: A Chroma database containing text snippets, IDs, metadata, and embeddings.
    """
    # Initialize embeddings and Chroma client
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    chroma_client = chromadb.PersistentClient(path="chromadb", settings=Settings(allow_reset=True))

    # # Uncomment the following line to reset the database before loading the texts
    # chroma_client.reset()

    # Create persistent vector store
    langchain_chroma = Chroma(
        client=chroma_client,
        collection_name="langchain_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    # Populate vector store with text, IDs, and metadata
    langchain_chroma.add_texts(
        texts=docs,
        ids=doc_ids,
        metadatas=metadata
    )

    return langchain_chroma