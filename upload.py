# upload.py
from pathlib import Path
import streamlit as st
import mimetypes
from PyPDF2 import PdfReader  # PyPDF2 is typically used for PDF parsing

# Conditional imports for file type handling
# These are placed here to ensure they are available if needed
# You might want to install these: pip install pandas python-docx PyPDF2
try:
    import pandas as pd
except ImportError:
    pd = None
    st.warning(
        "Pandas not installed. CSV/Excel previews might not work. Run `pip install pandas`"
    )
try:
    from docx import Document
except ImportError:
    Document = None
    st.warning(
        "python-docx not installed. DOCX previews might not work. Run `pip install python-docx`"
    )


def upload_and_save_files(upload_dir: str = "uploaded_files") -> list[Path]:
    """
    Allows a user to upload multiple files, saves them to the specified directory,
    and displays a preview for each supported file type.

    Args:
        upload_dir (str): The name of the directory to save uploaded files.

    Returns:
        list[Path]: A list of Path objects for the successfully saved files.
                    Returns an empty list if no files were uploaded or saved.
    """
    UPLOAD_DIR = Path(upload_dir)
    UPLOAD_DIR.mkdir(exist_ok=True)  # Ensure the directory exists

    st.markdown("#### Upload Documents (PDF, TXT, DOCX, CSV, XLSX)")
    uploaded_files = st.file_uploader(
        "Choose files to upload:",
        type=[
            "pdf",
            "txt",
            "docx",
            "csv",
            "xlsx",
        ],  # Expanded types
        accept_multiple_files=True,  # Allow multiple file selection
        help="Select one or more documents from your computer. Supported formats include PDF, TXT, DOCX, CSV, XLSX.",
    )

    saved_file_paths = []
    if uploaded_files:
        st.write(f"Processing {len(uploaded_files)} file(s)...")
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name.replace(" ", "_")  # Sanitize filename
            save_path = UPLOAD_DIR / filename

            # Use a spinner for each file to indicate ongoing work
            with st.spinner(f"Saving '{filename}'..."):
                try:
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_file_paths.append(save_path)
                    st.success(
                        f"File '{filename}' saved successfully to `{save_path}`."
                    )
                except Exception as e:
                    st.error(f"Error saving '{filename}': {e}")
                    continue  # Skip to the next file if saving fails

            # --- File Preview Section for each uploaded file ---
            st.markdown(f"---")
            st.markdown(f"### Preview of `{filename}`")

            mime_type, _ = mimetypes.guess_type(save_path)

            try:
                if filename.lower().endswith(".txt") or (
                    mime_type and mime_type.startswith("text")
                ):
                    content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                    st.text(content[:1000] + "..." if len(content) > 1000 else content)
                    if len(content) > 1000:
                        st.caption("Showing first 1000 characters.")

                elif filename.lower().endswith(".pdf"):
                    try:
                        reader = PdfReader(save_path)
                        # Extract text from first 2 pages for preview
                        preview_text = "\n".join(
                            page.extract_text() or "" for page in reader.pages[:2]
                        )
                        st.text(preview_text.strip()[:2000] + "...")
                        if len(preview_text.strip()) > 2000:
                            st.caption(
                                "Showing first 2000 characters from first 2 pages."
                            )
                        elif not preview_text.strip():
                            st.info("PDF has no readable text content for preview.")
                    except Exception as pdf_e:
                        st.warning(
                            f"⚠ Could not extract text from PDF for preview: {pdf_e}"
                        )

                elif filename.lower().endswith(".csv") and pd:
                    df = pd.read_csv(save_path)
                    st.dataframe(df.head())
                    st.caption("Showing first 5 rows.")

                elif filename.lower().endswith(".xlsx") and pd:
                    df = pd.read_excel(save_path)
                    st.dataframe(df.head())
                    st.caption("Showing first 5 rows.")

                elif filename.lower().endswith(".docx") and Document:
                    doc = Document(save_path)
                    text = "\n".join(p.text for p in doc.paragraphs)
                    st.text(text.strip()[:2000] + "...")
                    if len(text.strip()) > 2000:
                        st.caption("Showing first 2000 characters.")
                    elif not text.strip():
                        st.info("DOCX file has no readable text content for preview.")

                elif mime_type and mime_type.startswith("image"):
                    st.image(
                        str(save_path),
                        caption=f"Image: {filename}",
                        use_column_width=True,
                    )

                else:
                    st.info(
                        f"File '{filename}' saved, but preview not supported for this format (MIME: {mime_type})."
                    )
            except Exception as e:
                st.warning(f"⚠ An error occurred during preview for '{filename}': {e}")

            st.markdown("---")  # Separator between file previews

    return saved_file_paths


# Example of how you might use this (for testing, not part of the main app flow)
if __name__ == "__main__":
    st.set_page_config(layout="centered")
    st.title("Multi-File Uploader Test Page")
    st.write("Upload multiple files to see them saved and previewed.")

    # Simulating the UPLOAD_DIR for testing this file independently
    if not Path("uploaded_files").exists():
        Path("uploaded_files").mkdir()

    uploaded_paths = upload_and_save_files()
    if uploaded_paths:
        st.success(f"All selected files processed. Total saved: {len(uploaded_paths)}.")
    else:
        st.info("No files uploaded yet.")
