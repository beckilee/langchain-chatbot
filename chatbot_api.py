# Chatbot to chat with the LangChain docs!
# Uses ChatGPT, LangChain, Chroma, FastAPI, Streamlit

# This file contains the code for the chatbot backend.
# Use ngrok to make the API endpoints available on the internet.

import chromadb
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain.memory.buffer import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from fastapi import FastAPI
from langserve import add_routes

apiKey = os.environ.get("OPENAI_API_KEY")

# Initialize embeddings and Chroma client
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
chroma_client = chromadb.PersistentClient(path="chromadb")

# Instantiate persistent vector store
langchain_chroma = Chroma(
    client=chroma_client,
    collection_name="langchain_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Instantiate retriever using similarity search
retriever = langchain_chroma.as_retriever(search_type="similarity")

# Instantiate OpenAI chat model
llm = ChatOpenAI(model="gpt-4o")

# Define prompt template and instruct LLM to use context when answering a question
prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.
'''
{context}
'''

Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

# Initialize memory to store conversation history
memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)

# Initialize conversational retrieval chain 
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever,
    return_source_documents=True,
    memory=memory,
    verbose=False,
    combine_docs_chain_kwargs=chain_type_kwargs,
)

# Initialize FastAPI app
app = FastAPI(
    title="LangChain Docs Chatbot",
    version="1.0",
    description="Chat with the LangChain docs!",
)

# Add API routes to the app
add_routes(app, qa_chain)

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app
    uvicorn.run(app, host="localhost", port=8000)