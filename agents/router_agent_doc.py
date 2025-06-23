"""
router_agent_doc.py

Routes user questions to the appropriate Q&A agent based on file extension.
Supported agents:
- CSV Agent (.csv, .xls, .xlsx)
- PDF Agent (.pdf)
- Text Agent (.txt, .md, .docx)

Each agent module should expose a function `run(file_path: Path, question: str) -> str`.
"""

from pathlib import Path
from typing import Callable, Dict

# Import each agent's run function
from agents.csv_agent import run as run_csv_agent
from agents.DocsPdf_agent import run as run_pdf_agent
from agents.text_agent import run as run_text_agent

# Mapping of file extensions to agent run functions
EXTENSION_AGENT_MAP: Dict[str, Callable[[Path, str], str]] = {
    ".csv": run_csv_agent,
    ".xls": run_csv_agent,
    ".xlsx": run_csv_agent,
    ".pdf": run_pdf_agent,
    ".txt": run_text_agent,
    ".md": run_text_agent,
    ".docx": run_text_agent,
}


def route_question(file_path: Path, question: str) -> str:
    """
    Routes the question to the correct agent based on the file extension.

    Args:
        file_path (Path): Path to the document file.
        question (str): User's query string.

    Returns:
        str: The agent's answer.

    Raises:
        ValueError: If no agent is registered for the file extension.
    """
    ext = file_path.suffix.lower()
    agent_fn = EXTENSION_AGENT_MAP.get(ext)
    if agent_fn is None:
        supported = ", ".join(sorted(EXTENSION_AGENT_MAP.keys()))
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported extensions: {supported}"
        )
    # Delegate to the selected agent
    return agent_fn(file_path, question)
