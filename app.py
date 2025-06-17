# app.py
from dotenv import load_dotenv
import os
import streamlit as st
from pathlib import Path

# Import authentication functions
from auth import register_user, authenticate_user

# Import your existing RAG functionalities
# Ensure these files and their dependencies (like OpenAI, Langchain, unstructured, pypdf, faiss-cpu) are installed
from upload import upload_and_save_file
from ingestion import ingest_to_faiss_per_file
from agents.summarizer import summarize_file
from agents.router_agent_doc import route_question as router
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="OpenLLM Platform",
    page_icon="✨",  # A more modern, light icon
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS for a highly professional and soothing look (Gemini-inspired) ---
st.markdown("""
<style>
    /* General Page & Font Settings */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Google Sans', 'Roboto', 'Segoe UI', sans-serif; /* Prioritize Google Sans if available */
        color: var(--text-color);
        background-color: var(--background-color);
    }

    /* Load Google Fonts (if 'Google Sans' isn't natively available, Roboto is a good fallback) */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');


    /* Streamlit's main content area padding and width */
    .st-emotion-cache-1cypcdi { /* This targets the main content padding */
        padding-top: 4rem; /* More generous top padding */
        padding-bottom: 4rem; /* More generous bottom padding */
        padding-left: 2rem; /* Adjusted horizontal padding for centered layout */
        padding-right: 2rem;
    }

    /* Global Container styling for rounded corners on the overall content block */
    .st-emotion-cache-nahz7x { /* Streamlit's main content wrapper (might vary, needs testing) */
        border-radius: 20px; /* More rounded overall container */
        overflow: hidden; /* Ensures content respects border radius */
    }

    /* Universal color variables for light and dark mode (Softer Palette) */
    :root {
        /* Light Mode (Softer, Gemini-like) */
        --primary-color: #4285F4; /* Google Blue */
        --secondary-color: #616262; /* Softer gray */
        --background-color: #F8F9FA; /* Very light, almost off-white background */
        --card-background: #FFFFFF; /* Pure white for subtle contrast on background */
        --text-color: #3C4043; /* Google's dark text color */
        --light-text-color: #5F6368; /* For secondary text/placeholders */
        --border-color: #DADCE0; /* Very light gray border */
        --accent-color: #34A853; /* Google Green for success */
        --warning-color: #FBBC04; /* Google Yellow for warning */
        --error-color: #EA4335; /* Google Red for error */
        --shadow-color: rgba(60, 64, 67, 0.1); /* Subtle shadow for depth */
    }

    /* Dark Mode Overrides (Softer, Gemini-like) */
    [data-theme="dark"] {
        --primary-color: #8AB4F8; /* Lighter Google Blue for dark mode */
        --secondary-color: #BDC1C6; /* Lighter gray */
        --background-color: #202124; /* Google Dark Mode Background */
        --card-background: #2D2E30; /* Slightly lighter card background for subtle depth */
        --text-color: #CAD2DB; /* Softer light gray for main text */
        --light-text-color: #9AA0A6; /* For secondary text/placeholders, a bit more muted */
        --border-color: #5F6368; /* Darker, subtle border */
        --accent-color: #81C995; /* Lighter Google Green */
        --warning-color: #FDD663; /* Lighter Google Yellow */
        --error-color: #F28B82; /* Lighter Google Red */
        --shadow-color: rgba(0, 0, 0, 0.4); /* Darker shadow */
    }

    /* Main Header Styling */
    h1 {
        text-align: center;
        color: var(--primary-color);
        font-size: 3rem; /* Larger, more impactful */
        margin-bottom: 0.5rem;
        font-weight: 700; /* Bolder */
        letter-spacing: -0.03em;
    }

    /* Subheader Styling */
    h2 {
        color: var(--text-color); /* General text color for subheaders */
        font-size: 2rem;
        border-bottom: none; /* No hard border below */
        padding-bottom: 0.5rem;
        margin-top: 3rem;
        margin-bottom: 2rem;
        text-align: center; /* Center these too */
        font-weight: 500;
    }

    /* Section Headers */
    h3 {
        color: var(--text-color);
        font-size: 1.6rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-weight: 500;
        text-align: left; /* Align these to left usually */
    }

    /* Paragraphs and general text */
    p, label {
        color: var(--text-color);
        line-height: 1.7;
        margin-bottom: 1rem;
    }

    /* Info/Instructions Text in general markdown */
    div[data-testid="stMarkdownContainer"] p:not([class]) { /* Target generic markdown paragraphs, not those in alerts */
        color: var(--light-text-color);
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    /* Form & Container Styling (Soothing Cards) */
    div[data-testid="stForm"] > div,
    [data-testid="stVerticalBlock"] > div.st-emotion-cache-nahz7x { /* Target Streamlit form elements and containers */
        background-color: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 16px; /* Softer rounded corners */
        padding: 2.5rem 3rem; /* Increased padding inside forms/cards */
        box-shadow: 0 4px 15px var(--shadow-color); /* Subtle, deeper shadow */
        margin-bottom: 2.5rem; /* Space between sections */
    }

    /* Input Fields */
    .st-emotion-cache-vdzxz9 input[type="text"],
    .st-emotion-cache-vdzxz9 input[type="password"],
    .st-emotion-cache-vdzxz9 textarea,
    .st-emotion-cache-vdzxz9 .stSelectbox { /* Target Streamlit inputs, textareas, selectboxes */
        border-radius: 12px; /* Consistent rounding */
        border: 1px solid var(--border-color);
        padding: 0.85rem 1.2rem; /* More comfortable padding */
        background-color: var(--background-color); /* Slightly lighter background than card */
        color: var(--text-color);
        font-size: 1rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .st-emotion-cache-vdzxz9 input[type="text"]:focus,
    .st-emotion-cache-vdzxz9 input[type="password"]:focus,
    .st-emotion-cache-vdzxz9 textarea:focus,
    .st-emotion-cache-vdzxz9 .stSelectbox:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 0.15rem rgba(66, 133, 244, 0.25); /* Google Blue shadow */
        outline: none;
    }
    .st-emotion-cache-vdzxz9 input::placeholder,
    .st-emotion-cache-vdzxz9 textarea::placeholder {
        color: var(--light-text-color);
        opacity: 0.7;
    }


    /* Buttons */
    .st-emotion-cache-use3lb button { /* Targeting Streamlit buttons */
        background-color: var(--primary-color);
        color: white;
        border-radius: 12px; /* Consistent rounding */
        padding: 0.9rem 1.8rem; /* More generous padding */
        font-weight: 600; /* Slightly bolder */
        transition: background-color 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        font-size: 1rem;
    }
    .st-emotion-cache-use3lb button:hover {
        background-color: #357AE8; /* Slightly darker Google Blue on hover */
        transform: translateY(-2px); /* Slight lift effect */
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .st-emotion-cache-use3lb button:active {
        transform: translateY(0); /* Reset on click */
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    /* Alerts (Info, Success, Warning, Error) - Softer, Google-inspired colors */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        padding: 1.2rem 1.8rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05); /* Subtle shadow for alerts */
    }
    /* IMPORTANT: Target the text inside alerts to override default Streamlit text color */
    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
        margin-bottom: 0;
        color: var(--text-color); /* Ensures text color uses our defined variable for better contrast */
    }

    .st-emotion-cache-10grl2h { /* Info alert background */
        background-color: var(--card-background);
        border-left: 5px solid var(--primary-color);
    }
    .st-emotion-cache-1a662z3 { /* Success alert background */
        background-color: var(--card-background);
        border-left: 5px solid var(--accent-color);
    }
    .st-emotion-cache-fcn1f7 { /* Warning alert background */
        background-color: var(--card-background);
        border-left: 5px solid var(--warning-color);
    }
    .st-emotion-cache-1wlbx07 { /* Error alert background */
        background-color: var(--card-background);
        border-left: 5px solid var(--error-color);
    }

    /* Spinner */
    div[data-testid="stSpinner"] .st-emotion-cache-121u44h {
        color: var(--primary-color); /* Spinner color */
    }


    /* Sidebar Styling */
    .st-emotion-cache-1gh7q3p { /* Targeting sidebar background */
        background-color: var(--card-background);
        border-right: 1px solid var(--border-color);
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
        padding-top: 2rem;
    }
    .st-emotion-cache-1gh7q3p h3, .st-emotion-cache-1gh7q3p div { /* Text in sidebar */
        color: var(--text-color);
    }
    .st-emotion-cache-1gh7q3p button { /* Buttons in sidebar */
        background-color: var(--secondary-color); /* Use secondary color for sidebar buttons */
        color: white;
    }
    .st-emotion-cache-1gh7q3p button:hover {
        background-color: #5a6268;
    }

    /* Tabs Styling */
    .st-emotion-cache-ch5d7l button { /* Tabs themselves */
        background-color: var(--background-color);
        color: var(--light-text-color); /* Softer text color for inactive tabs */
        border-bottom: 2px solid transparent; /* No initial border */
        font-weight: 500;
        padding: 0.75rem 1.5rem; /* More padding */
        border-radius: 12px 12px 0 0; /* Consistent rounding */
        transition: color 0.2s ease, border-bottom 0.2s ease;
    }
    .st-emotion-cache-ch5d7l button[aria-selected="true"] {
        color: var(--primary-color); /* Primary color for active tab */
        border-bottom: 2px solid var(--primary-color);
        background-color: var(--card-background); /* Card background for active tab */
    }
    .st-emotion-cache-ch5d7l { /* Tab container */
        border-bottom: 1px solid var(--border-color); /* Subtle line under tabs */
        margin-bottom: 2rem;
    }

    /* Expander Styling for history */
    .st-emotion-cache-s2s6p1 { /* Target the expander header */
        border: 1px solid var(--border-color);
        border-radius: 12px;
        background-color: var(--card-background);
        padding: 0.8rem 1.2rem;
        box-shadow: 0 1px 5px rgba(0,0,0,0.05); /* Subtle shadow */
    }
    .st-emotion-cache-s2s6p1:hover {
        background-color: var(--background-color); /* Slightly change background on hover */
    }
    .st-emotion-cache-1fo623y { /* Expander content */
        background-color: var(--card-background);
        border: 1px solid var(--border-color);
        border-top: none;
        border-radius: 0 0 12px 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 5px rgba(0,0,0,0.05);
    }
    .st-emotion-cache-1fo623y p { /* Text inside expander */
        margin-bottom: 0.5rem;
    }

    /* Adjust specific Streamlit elements that might still have default styling */
    .stFileUploader {
        padding: 1.5rem;
        border: 1px dashed var(--border-color);
        border-radius: 12px;
        background-color: var(--background-color);
        margin-bottom: 1.5rem;
    }
    .stFileUploader label span {
        color: var(--light-text-color);
    }
    .stFileUploader > div:first-child > div:first-child {
        background-color: var(--primary-color);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Session State Initialization for Authentication & RAG
# ────────────────────────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None

# Initialize memory for RAG functionalities
if "doc_qa_history" not in st.session_state:
    st.session_state.doc_qa_history = []  # list of (stem, question, answer)
if "global_qa_history" not in st.session_state:
    st.session_state.global_qa_history = []  # list of (question, stem, answer)


# ────────────────────────────────────────────────────────────────────────────────
# Authentication Pages
# ────────────────────────────────────────────────────────────────────────────────
def login_page():
    """
    Renders the login interface with a professional layout.
    """
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>🔑 Secure Access</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Please enter your credentials to access the OpenLLM platform.</p>",
                    unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", key="login_username_input", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_password_input",
                                     placeholder="Enter your password")

            st.markdown("---")
            login_button = st.form_submit_button("Sign In", use_container_width=True)

            if login_button:
                success, message, role = authenticate_user(username, password)
                if success:
                    st.success(message)
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.role = role
                    st.rerun()
                else:
                    st.error(message)

    st.markdown("---")
    # This is the line that was hard to read, now the text color is inherited from var(--text-color)
    st.info("💡 New to OpenLLM? Register for an account using the 'Register' tab.")


def register_page():
    """
    Renders the registration interface with a professional layout.
    """
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>📝 Create OpenLLM Account</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center;'>Join our platform by creating a new user account. Your role will be assigned automatically.</p>",
            unsafe_allow_html=True)

        with st.form("register_form"):
            new_username = st.text_input("New Username", key="register_username_input",
                                         placeholder="Choose a unique username")
            new_password = st.text_input("New Password", type="password", key="register_password_input",
                                         placeholder="Create a strong password")

            st.markdown("---")
            register_button = st.form_submit_button("Register Account", use_container_width=True)

            if register_button:
                success, message = register_user(new_username, new_password)
                if success:
                    st.success(message)
                    # This info message text color should now be softer in dark mode
                    st.info(
                        "Account created successfully! You can now log in. The **first user registered becomes the Admin**, all subsequent users are **QA users**.")
                else:
                    st.error(message)

    st.markdown("---")
    # This info message text color should now be softer in dark mode
    st.info("Already have an OpenLLM account? Navigate to the 'Login' tab.")


# ────────────────────────────────────────────────────────────────────────────────
# Main Application Content (Visible after Login)
# ────────────────────────────────────────────────────────────────────────────────
def main_rag_app():
    """
    Displays the RAG functionalities based on the user's role.
    """
    st.sidebar.markdown("### User Profile")
    st.sidebar.write(f"**Username:** `{st.session_state.username}`")
    st.sidebar.write(f"**Role:** `{st.session_state.role.capitalize()}`")
    st.sidebar.markdown("---")

    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        st.rerun()

    st.markdown(f"<h1 style='text-align: center; color: var(--primary-color);'>📚 OpenLLM Insight Platform</h1>",
                unsafe_allow_html=True)
    st.markdown(
        f"<p style='text-align: center; color: var(--text-color);'>Welcome, {st.session_state.username}! Your access level: <b>{st.session_state.role.capitalize()}</b></p>",
        unsafe_allow_html=True)
    st.markdown("---")

    # 1) Upload + per-file FAISS ingestion + summarization (Admin Only)
    if st.session_state.role == 'admin':
        st.subheader("⬆️ Document Management & Ingestion")
        st.markdown(
            "<p>Admins can upload new documents (PDF/TXT) to expand the knowledge base. The system will automatically ingest them for QA and generate summaries.</p>",
            unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("### Upload Files")
            file_path: Path | None = upload_and_save_file()
            if file_path:
                stem = file_path.stem

                st.info(f"▶️ Ingesting `{file_path.name}` into FAISS… This may take a moment.")
                try:
                    ingest_to_faiss_per_file(file_path, base_dir="document_index")
                    st.success(f"✅ Ingestion complete! Document index created at `document_index/{stem}`.")
                except Exception as e:
                    st.error(f"❌ Error during ingestion: {e}")
                    st.stop()

                # automatic summary
                st.markdown("---")
                st.markdown("### Automatic Summary")
                st.write("Generating an automatic summary for the uploaded document:")
                try:
                    summary = summarize_file(file_path)
                    st.markdown(summary)
                    st.success("✅ Summary generated successfully.")
                except Exception as e:
                    st.warning(
                        f"⚠️ Summarization failed: {e}. Please ensure your summarizer agent is configured correctly.")
    else:
        st.subheader("🚫 Document Upload Access")
        st.warning(
            "You do not have permission to upload documents. This feature is restricted to **Admin** users to maintain data integrity.")

    st.markdown("---")

    # 2) Per-document QA via router (with memory) - Accessible to all logged-in users
    index_root = Path("document_index")
    available = [d.name for d in index_root.iterdir() if d.is_dir()] if index_root.exists() else []

    st.subheader("🔎 Query Specific Documents")
    st.markdown("<p>Select a document from the dropdown to ask targeted questions and retrieve precise answers.</p>",
                unsafe_allow_html=True)

    if available:
        with st.container(border=True):
            chosen = st.selectbox("Choose document for QA", options=available, key="qa_specific_doc_select",
                                  help="Select a document you wish to query.")
            question = st.text_input("Enter your question about the selected document:", key="qa_specific_doc_question",
                                     placeholder="e.g., What are the key findings?")

            if st.button("Run QA on Selected Document", key="run_qa_selected_button", use_container_width=True):
                upload_dir = Path("uploaded_files")
                matches = list(upload_dir.glob(f"{chosen}.*"))
                if not matches:
                    st.error(
                        f"No original uploaded file found for ‘{chosen}’. Please ensure it was uploaded correctly.")
                else:
                    try:
                        with st.spinner(f"Analyzing '{chosen}' to find the answer..."):
                            answer = router(matches[0], question)
                            st.session_state.doc_qa_history.append((chosen, question, answer))
                            st.markdown(f"**Answer (from `{chosen}`):** \n{answer}")
                            st.success("Query complete!")
                    except Exception as e:
                        st.error(f"❌ QA Error: {e}. Please check the document and your question.")
    else:
        st.info("No documents are currently indexed for specific QA. Please upload some files first (Admin only).")

    # display memory
    if st.session_state.doc_qa_history:
        st.write("#### 📝 Previous Per-Document Queries")
        with st.expander("View Document-Specific Q&A History"):
            for doc_stem, q, a in reversed(st.session_state.doc_qa_history):  # Show most recent first
                st.markdown(f"**Document:** `{doc_stem}`\n**Q:** {q}  \n**A:** {a}")
                st.markdown("---")

    st.markdown("---")

    # 3) 🌐 Global QA across all documents (with memory) - Accessible to all logged-in users
    st.subheader("🌐 Global Knowledge Base Query")
    st.markdown("<p>Ask a question across all available document summaries to get comprehensive insights.</p>",
                unsafe_allow_html=True)

    with st.container(border=True):
        general_q = st.text_input("Enter your global question:", key="qa_global_question",
                                  placeholder="e.g., What are the common themes across all reports?")

        if st.button("Run Global QA", key="run_global_qa_button", use_container_width=True):
            summary_dir = Path("uploaded_files")
            summary_paths = list(summary_dir.glob("*.summary.txt"))
            if not summary_paths:
                st.error("❌ No summary files found. Please ensure documents have been uploaded and summarized.")
            else:
                docs = []
                for sp in summary_paths:
                    text = sp.read_text(encoding="utf-8")
                    stem = sp.stem.removesuffix(".summary")
                    docs.append(Document(page_content=text, metadata={"stem": stem}))

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    st.error(
                        "`OPENAI_API_KEY` environment variable is not set. Please set it to proceed with global QA.")
                    st.stop()

                try:
                    with st.spinner("Building global knowledge base and finding relevant information..."):
                        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                        vs = FAISS.from_documents(docs, embeddings)
                        retriever = vs.as_retriever()

                        top = retriever.get_relevant_documents(general_q)[:1]  # Get top 1 relevant document
                        if not top:
                            st.warning("No relevant document found for your global question. Try rephrasing.")
                        else:
                            chosen_stem = top[0].metadata["stem"]
                            upload_dir = Path("uploaded_files")
                            matches = list(upload_dir.glob(f"{chosen_stem}.*"))
                            if not matches:
                                st.error(
                                    f"No original uploaded file found for ‘{chosen_stem}’ to answer the global question.")
                            else:
                                answer = router(matches[0], general_q)
                                st.session_state.global_qa_history.append((general_q, chosen_stem, answer))
                                st.markdown(f"**Best Match Document:** `{chosen_stem}`  \n**Answer:** {answer}")
                                st.success("Global query complete!")
                except Exception as e:
                    st.error(f"❌ Global QA Error: {e}. Ensure your OpenAI API key is valid and agents are working.")

    # display global QA history
    if st.session_state.global_qa_history:
        st.write("#### 🌟 Previous Global Queries")
        with st.expander("View Global Q&A History"):
            for q, stem, a in reversed(st.session_state.global_qa_history):  # Show most recent first
                st.markdown(f"**Q:** {q}  \n**Best Match:** `{stem}`  \n**A:** {a}")
                st.markdown("---")

    st.markdown("---")
    st.success("OpenLLM Insight Platform: Empowering knowledge discovery!")


# ────────────────────────────────────────────────────────────────────────────────
# Main App Flow: Authentication vs. Main App
# ────────────────────────────────────────────────────────────────────────────────
# Display a welcoming header that appears before login/register tabs
st.markdown("<h1 style='text-align: center; color: var(--primary-color);'>Welcome to OpenLLM</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: var(--text-color);'>Your secure gateway to powerful document intelligence.</p>",
    unsafe_allow_html=True)
st.markdown("---")

if st.session_state.logged_in:
    # If logged in, display the main RAG application content
    main_rag_app()
else:
    # If not logged in, show login/register tabs for authentication
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        login_page()
    with tab2:
        register_page()

