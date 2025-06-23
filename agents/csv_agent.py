# agents/csv_agent.py

from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonREPLTool, PythonAstREPLTool

load_dotenv()


def _load_csv(file_path: Path, max_header_row: int = 5) -> pd.DataFrame:
    """
    Load any CSV, auto-fixing a misplaced header and
    dynamically parsing any date-like column.
    """
    best_df = None
    best_score = -1

    # Try each possible header row
    for header in range(max_header_row):
        try:
            df_try = pd.read_csv(file_path, header=header)
        except Exception:
            continue

        # Score by count of "good" columns (non-empty, not Unnamed)
        good_cols = [
            c
            for c in df_try.columns
            if str(c).strip() and not str(c).startswith("Unnamed")
        ]
        score = len(good_cols)

        if score > best_score:
            best_score = score
            best_df = df_try

    if best_df is None:
        raise ValueError(f"Could not detect header in first {max_header_row} rows")

    # Clean up column names
    best_df.columns = [str(c).strip() for c in best_df.columns]

    # Parse date-like columns
    for col in best_df.columns:
        if best_df[col].dtype == "object":
            parsed = pd.to_datetime(
                best_df[col], errors="coerce", infer_datetime_format=True
            )
            if parsed.notna().mean() > 0.9:
                best_df[col] = parsed

    return best_df


def run(file_path: Path, question: str) -> str:
    """
    Load CSV at file_path, optionally include a precomputed summary,
    spin up a DataFrame agent, and return its answer to question.
    """
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_KEY env var not set")

    # 1) LLM
    llm = ChatOpenAI(api_key=api_key, model="gpt-4o", temperature=0)

    # 2) Load & preprocess CSV
    df = _load_csv(file_path).fillna(0)

    # 3) Load summary if available
    summary_path = file_path.with_suffix(file_path.suffix + ".summary.txt")
    if summary_path.exists():
        summary = summary_path.read_text(encoding="utf-8").strip()
    else:
        summary = None

    # 4) Collect schema information
    columns_list = ", ".join(df.columns)

    # 5) Create pandas DataFrame agent
    agent = create_pandas_dataframe_agent(
        llm=llm,
        extra_tools=[],
        df=df,
        verbose=False,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
        max_iterations=15,
    )

    # 6) Prepare the enriched prompt
    prompt_parts = []
    if summary:
        prompt_parts.append(f"High-level summary of dataset:\n{summary}")
    prompt_parts.append(f"Columns available: {columns_list}")
    prompt_parts.append(f"Question: {question}")
    enriched_prompt = "\n\n".join(prompt_parts)

    # 7) Invoke the agent with enriched context
    try:
        res = agent.invoke(enriched_prompt, handle_parsing_errors=True)
        if isinstance(res, dict) and "output" in res:
            return res["output"]
        else:
            # fallback to stringifying whatever you got back
            return str(res)

    except Exception:
        # Fallback: include summary, schema, and snapshot
        df_head = df.head().to_markdown()
        fallback_prompt = PromptTemplate.from_template(
            """
            {summary_block}
            Columns: {columns}
            Here is a snapshot of the data:
            {df_head}

            Question: {question}
            Answer:
            """
        )
        fb_inputs = {
            "summary_block": f"Dataset summary:\n{summary}\n\n" if summary else "",
            "columns": columns_list,
            "df_head": df_head,
            "question": question,
        }
        fb_chain = fallback_prompt | llm
        fb = fb_chain.invoke(fb_inputs)
        return fb["output"]
