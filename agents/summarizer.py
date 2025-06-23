# agents/summarizer.py

from pathlib import Path
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)

load_dotenv()


def _load_snippet(path: Path, max_pages: int = 3, max_rows: int = 5) -> Optional[str]:
    """
    Load a short excerpt from the document at `path`.
    For PDFs: first `max_pages` pages.
    For CSVs: first `max_rows` rows as markdown.
    For .txt/.md/.docx/.xlsx: first page/text block.
    Returns None if nothing could be read.
    """
    suffix = path.suffix.lower()

    try:
        if suffix == ".pdf":
            pages = PyPDFLoader(str(path)).load()
            texts = [p.page_content for p in pages[:max_pages]]

        elif suffix in {".txt", ".md"}:
            docs = TextLoader(str(path), encoding="utf-8").load()
            texts = [docs[0].page_content] if docs else []

        elif suffix == ".csv":
            import pandas as pd

            df = pd.read_csv(path, nrows=max_rows).fillna("")
            texts = [df.to_markdown(index=False)]

        elif suffix == ".docx":
            docs = UnstructuredWordDocumentLoader(str(path)).load()
            texts = [docs[0].page_content] if docs else []

        elif suffix in {".xls", ".xlsx"}:
            docs = UnstructuredExcelLoader(str(path)).load()
            texts = [docs[0].page_content] if docs else []

        else:
            # Fallback
            docs = TextLoader(str(path), encoding="utf-8").load()
            texts = [docs[0].page_content] if docs else []

    except Exception:
        return None

    excerpt = "\n\n".join(t for t in texts if t)
    return excerpt or None


def summarize_file(file_path: Path) -> str:
    """
    Summarize the first snippet of `file_path` via LLM, save into
    uploaded_files/<stem>.summary.txt, and return the summary.
    Raises if OPENAI_KEY is missing or no content could be read.
    """
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_KEY env var not set")

    snippet = _load_snippet(file_path)
    if not snippet:
        raise ValueError(f"No previewable content in {file_path.name!r}")

    prompt = PromptTemplate.from_template(
        "Please provide a concise, one-paragraph summary of the following document excerpt:\n\n"
        "{excerpt}\n\n"
        "Summary:"
    )
    llm = ChatOpenAI(api_key=api_key, model="gpt-4o", temperature=0)
    chain = prompt | llm

    raw = chain.invoke({"excerpt": snippet})
    # chain.invoke returns either a str or an AIMessage-like object
    text = getattr(raw, "content", raw if isinstance(raw, str) else str(raw))
    summary = text.strip()

    # Save summary
    out_path = Path("uploaded_files") / f"{file_path.stem}.summary.txt"
    out_path.write_text(summary, encoding="utf-8")

    return summary


def summarize_all(upload_dir: str = "uploaded_files") -> None:
    """
    Iterate through every file in `uploaded_files/` (excluding .summary.txt),
    generate a summary if none exists yet, and print progress to console.
    """
    upload_dir = Path(upload_dir)
    for path in sorted(upload_dir.iterdir()):
        if path.suffix.lower() == ".txt" and path.name.endswith(".summary.txt"):
            continue

        summary_file = upload_dir / f"{path.stem}.summary.txt"
        if summary_file.exists():
            continue

        print(f"Summarizing {path.name}…", end=" ")
        try:
            summary = summarize_file(path)
            print("✔️")
        except Exception as e:
            print(f"❌ skipped ({e})")


if __name__ == "__main__":
    summarize_all()
