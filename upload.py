from pathlib import Path
import streamlit as st
import mimetypes
from PyPDF2 import PdfReader

def upload_and_save_file(upload_dir: str = "uploaded_files") -> Path | None:
    UPLOAD_DIR = Path(upload_dir)
    UPLOAD_DIR.mkdir(exist_ok=True)

    uploaded_file = st.file_uploader("Upload a file", type=None)

    if uploaded_file is not None:
        filename = uploaded_file.name.replace(" ", "_")
        save_path = UPLOAD_DIR / filename

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("✅ File saved successfully!")
        st.code(str(save_path), language="bash")

        # Preview
        mime_type, _ = mimetypes.guess_type(save_path)
        st.write("---")
        st.write("### 🔍 File Preview")

        try:
            if filename.endswith(".txt") or (mime_type and mime_type.startswith("text")):
                content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                st.text(content[:1000] + "..." if len(content) > 1000 else content)

            elif filename.endswith(".pdf"):
                reader = PdfReader(save_path)
                preview_text = "\n".join(page.extract_text() or "" for page in reader.pages[:2])
                st.text(preview_text.strip()[:2000] + "...")

            elif filename.endswith(".csv"):
                import pandas as pd
                df = pd.read_csv(save_path)
                st.dataframe(df.head())

            elif filename.endswith(".xlsx"):
                import pandas as pd
                df = pd.read_excel(save_path)
                st.dataframe(df.head())

            elif filename.endswith(".docx"):
                from docx import Document
                doc = Document(save_path)
                text = "\n".join(p.text for p in doc.paragraphs)
                st.text(text.strip()[:2000] + "...")

            elif mime_type and mime_type.startswith("image"):
                st.image(save_path)

            else:
                st.info("📦 File saved, but preview not supported for this format.")
        except Exception as e:
            st.warning(f"⚠️ Could not preview the file: {e}")

        return save_path

    return None
