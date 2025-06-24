from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Directory where summaries are stored
SUMMARY_DIR = Path("uploaded_files")

# LLM config
API_KEY = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"
TEMP = 0

# Prompt to choose best file based on summaries and question
SELECT_PROMPT = PromptTemplate.from_template(
    """
You are a document and image selector. You have the following summaries:

{summaries}

Given the user question:
{question}

Which single file (provide only its filename with extension) is most likely to contain the answer? If none apply, respond with 'none'.
"""
)


def select_file(question: str) -> Path:
    """
    Chooses the best file (document or image) to answer the question based on precomputed summaries.

    Args:
        question (str): The user's question.

    Returns:
        Path: Path to the selected file under uploaded_files/, or raises if none found.
    """
    if not API_KEY:
        raise RuntimeError("OPENAI_KEY environment variable is not set")

    # Gather all summaries
    entries = []
    for summary_file in SUMMARY_DIR.glob("*.summary.txt"):
        stem = summary_file.stem.replace(".summary", "")
        text = summary_file.read_text(encoding="utf-8").strip()
        entries.append(f"- {stem}: {text}")

    if not entries:
        raise FileNotFoundError(f"No summary files found in {SUMMARY_DIR}")

    summaries_block = "\n".join(entries)

    # Build and invoke LLM
    llm = ChatOpenAI(api_key=API_KEY, model=MODEL, temperature=TEMP)
    chain = SELECT_PROMPT | llm
    inputs = {"summaries": summaries_block, "question": question}
    result = chain.invoke(inputs)
    choice = (result.get("text") or result).strip()

    # Interpret result
    if choice.lower() == "none":
        raise ValueError("No suitable file found for this question.")

    # Match filename in uploaded_files
    candidate = SUMMARY_DIR / choice
    if not candidate.exists():
        # Try appending extensions for docs and images
        extensions = [".pdf", ".csv", ".txt", ".docx", ".xlsx", ".png", ".jpg", ".jpeg"]
        for ext in extensions:
            p = SUMMARY_DIR / (choice + ext)
            if p.exists():
                candidate = p
                break

    if not candidate.exists():
        raise FileNotFoundError(f"Selected file '{choice}' not found in {SUMMARY_DIR}")

    return candidate
