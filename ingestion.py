# ingestion.py

from pathlib import Path
import os
from dotenv import load_dotenv

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# loaders for non-CSV types
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)

load_dotenv()

def ingest_to_faiss_per_file(
    file_path: str | Path,
    base_dir: str = "document_index",
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> Path:
    """
    Ingest a single file into its own FAISS index under base_dir/<file_stem>/.
    Returns the path to that index folder.
    """
    file_path = Path(file_path)
    stem = file_path.stem
    index_dir = Path(base_dir) / stem
    index_dir.mkdir(parents=True, exist_ok=True)

    suffix = file_path.suffix.lower()

    # 1) Load into a list of Documents
    if suffix == ".csv":
        # pandas‐based CSV fallback
        import pandas as pd
        try:
            df = pd.read_csv(file_path).fillna("")  # read first
            markdown = df.to_markdown(index=False)
            docs = [ Document(page_content=markdown, metadata={"source": str(file_path)}) ]
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV {file_path}: {e}")
    else:
        # the old loaders for PDF, text, docx, xlsx...
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif suffix in {".txt", ".md"}:
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif suffix == ".docx":
            loader = UnstructuredWordDocumentLoader(str(file_path))
        elif suffix in {".xls", ".xlsx"}:
            loader = UnstructuredExcelLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        docs = loader.load()

    # 2) Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    # 3) Embed & upsert
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_KEY env var not set")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Load existing or create new
    if (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists():
        vs = FAISS.load_local(str(index_dir), embeddings,
                              allow_dangerous_deserialization=True)
        vs.add_documents(chunks)
    else:
        vs = FAISS.from_documents(chunks, embeddings)

    # 4) Save
    vs.save_local(str(index_dir))
    return index_dir
