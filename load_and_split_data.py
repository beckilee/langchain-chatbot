# Chatbot to chat with the LangChain docs!
# Uses ChatGPT, LangChain, Chroma, FastAPI, Streamlit

# This file reads downloaded LangChain doc files, splits them into chunks,
# and writes the chunks to a JSONL file.

from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
import uuid
import json
from langchain_core.documents.base import Document
from typing import List, Dict

def calculate_token_length(text: str) -> int:
    """
    Calculate the number of tokens in a given text using the initialized tokenizer.

    Args:
        text (str): Text to tokenize.
    
    Returns:
        Int: Number of tokens in the provided text.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokenized_text = encoding.encode(text, disallowed_special=())
    return len(tokenized_text)

def load_directory(directory: str) -> List[Document]:
    """
    Load a directory of HTML files and return a list of Documents using LangChain.

    Args:
        directory (str): Directory of HTML files to load.

    Returns:
        List[Document]: A list of LangChain Documents.
    """
    loader = ReadTheDocsLoader(directory)
    docs = loader.load()
    num_docs = len(docs)
    # Output number of processed documents
    print(f"Processed {num_docs} documents.")
    return docs

def split_text_into_chunks(documents: List[Document]) -> List[Document]:
    """
    Split Documents into chunks using `calculate_token_length()` as a length function.

    Args:
        documents (List[Document]): Pre-loaded list of LangChain Documents.

    Returns:
        List[Document]: A list of Documents split into chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20, 
        length_function=calculate_token_length, 
        separators=['\n\n', '\n', ' ', '']
        )
    split_text = list()
    split_text = splitter.split_documents(documents)
    return split_text

def create_docs_dict(documents: List[Document]) -> Dict[str, Dict[str, str]]:
    """
    Generate one dictionary per chunk from a list of Documents, where each key is a UUID string representing a chunk, and each value is another dictionary containing the chunk text and source URL.
    
    Example:
    
    ```
    {"bd86b9c0-9e6b-5b15-8628-5a9497b2651b": {"url": "apidocs/api.python.langchain.com/en/latest/smith/filename.html", "text": "Some chunk of text..."}}
    ```

    Args:
        documents (List[Document]): A list of Documents split into chunks.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary where each key is a UUID string representing a chunk, and each value is another dictionary containing the chunk text and source URL.
    """
    # Initialize variables
    counter = 0
    doc_dict = dict()

    # Process each chunk into a dictionary
    for doc in documents:
        # Set `url` to chunk source URL
        url = doc.metadata["source"]
        # Generate UUID for each chunk based on the chunk URL concatenated with a counter
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(doc.metadata["source"]) + "_" + str(counter)))
        # Set `text` to chunk content
        text = doc.page_content
        # Increase counter to prevent collisions
        counter = counter + 1
        # Create dictionary
        doc_dict[doc_id] = {"url": url, "text": text}
    
    return doc_dict

def docs_dict_to_json(doc_dict: Dict[str, Dict[str, str]], output_file: str) -> None:
    """
    Write a dictionary of dictionaries containing text chunks, source URLs, and UUIDs to a JSONL file, one dictionary per line.

    Args:
        doc_dict (Dict[str, Dict[str, str]]): A dictionary where each key is a UUID string representing a chunk, and each value is another dictionary containing the chunk text and source URL.
        output_file (str): The JSONL file to create.
    """
    with open(output_file, "w") as f:
        for uuid, inner_dict in doc_dict.items():
            json_line = json.dumps({uuid: inner_dict})
            f.write(json_line + "\n")