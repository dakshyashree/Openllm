"""
Text Agent (text_agent.py)

Loads a FAISS index for a single text-based document (created under document_index/<file_stem>/),
builds a RetrievalQA chain, and answers user questions.

Expose:
    run(file_path: Path, question: str) -> str
"""
from pathlib import Path
from dotenv import load_dotenv
import os
import streamlit as st
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# ⚠️ Replace these deprecated imports:
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# with:
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def run(file_path: Path, question: str) -> str:
    """
    Given a PDF file_path and a question string, load its precomputed FAISS index,
    run a RetrievalQA chain, and return the answer.

    Args:
        file_path (Path): Path to the original PDF file.
        question (str): User's query.

    Returns:
        str: The LLM's answer.

    Raises:
        RuntimeError: If OPENAI_KEY is not set.
        FileNotFoundError: If the index directory is missing.
    """
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_KEY env var not set")

    # Determine index directory by file stem
    stem = file_path.stem
    index_dir = Path("document_index") / stem
    if not index_dir.exists():
        raise FileNotFoundError(f"Index directory not found for '{stem}': {index_dir}")

    # Load embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vs = FAISS.load_local(
        str(index_dir), embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vs.as_retriever()

    # Initialize LLM and chains
    llm = ChatOpenAI(api_key=api_key, model="gpt-4o", temperature=0)
    retrieval_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_chain = create_stuff_documents_chain(llm, retrieval_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_chain)

    # Run query
    # You can use .run for convenience (single-string input)
    answer = retrieval_chain.invoke({"input": question})
    return answer["answer"]

