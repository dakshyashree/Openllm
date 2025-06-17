from dotenv import load_dotenv
import os
import streamlit as st
from pathlib import Path
import yaml

# Import authentication functions
from auth import register_user, authenticate_user, change_password, get_all_users_status, update_user_status, delete_user # NEW: import delete_user

# Import RAG functionalities
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
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Inject Custom CSS from file ---
styles_dir = Path("styles")
styles_dir.mkdir(exist_ok=True)
css_file_path = styles_dir / "style.css"

if not css_file_path.exists():
    with open(css_file_path, "w") as f:
        f.write("/* Add your CSS styles here */\n")
    st.warning(f"'{css_file_path}' was not found and has been created. Please paste your CSS into it and restart the application.")
    st.stop()

with open(css_file_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Session State Initialization for Authentication & RAG
# ────────────────────────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None
if "page" not in st.session_state: # NEW: To manage current page
    st.session_state.page = "Main App" # Default page

if "doc_qa_history" not in st.session_state:
    st.session_state.doc_qa_history = []
if "global_qa_history" not in st.session_state:
    st.session_state.global_qa_history = []


# ────────────────────────────────────────────────────────────────────────────────
# Authentication Pages
# ────────────────────────────────────────────────────────────────────────────────
def login_page():
    """
    Renders the login interface.
    """
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>Secure Access</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Enter your credentials to access the platform.</p>",
                    unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", key="login_username_input", placeholder="Enter username")
            password = st.text_input("Password", type="password", key="login_password_input",
                                     placeholder="Enter password")

            st.markdown("---")
            login_button = st.form_submit_button("Sign In", use_container_width=True)

            if login_button:
                success, message, role = authenticate_user(username, password)
                if success:
                    st.success(message)
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.role = role
                    st.session_state.page = "Main App" # Redirect to main app after login
                    st.rerun()
                else:
                    st.error(message)

    st.markdown("---")
    st.info("New to OpenLLM? Register for an account using the 'Register' tab.")


def register_page():
    """
    Renders the registration interface.
    """
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>Create OpenLLM Account</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center;'>Create a new user account. Your role will be assigned automatically.</p>",
            unsafe_allow_html=True)

        with st.form("register_form"):
            new_username = st.text_input("New Username", key="register_username_input",
                                         placeholder="Choose a unique username")
            new_password = st.text_input("New Password", type="password", key="register_password_input",
                                         placeholder="Create a strong password")

            # --- Logic to determine and display the role ---
            users_file = Path("users.yaml")
            current_users_data = {}
            if users_file.exists():
                try:
                    with open(users_file, 'r') as f:
                        current_users_data = yaml.safe_load(f) or {}
                except yaml.YAMLError:
                    st.warning("Could not read users.yaml to accurately determine next role. Assuming QA by default.")
                    current_users_data = {"users": {}} # Ensure it's a dict for safety

            current_users = current_users_data.get("users", {}) # Get the 'users' key
            predicted_role = "admin" if not current_users else "qa_user"
            st.info(f"You will be registered as: **{predicted_role.replace('_', ' ').capitalize()}**")
            # --- End Logic ---

            st.markdown("---")
            register_button = st.form_submit_button("Register Account", use_container_width=True)

            if register_button:
                success, message = register_user(new_username, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

    st.markdown("---")
    st.info("Already have an OpenLLM account? Navigate to the 'Login' tab.")


# ────────────────────────────────────────────────────────────────────────────────
# Admin User Management Page
# ────────────────────────────────────────────────────────────────────────────────
def admin_user_management_page():
    """
    Dedicated page for Admin User Management.
    """
    st.markdown(f"<h1 style='text-align: center; color: var(--neon-accent);'>Admin User Management</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>View and manage user accounts: activate, deactivate, or delete users.</p>", unsafe_allow_html=True)
    st.markdown("---")

    users_data = get_all_users_status()
    if users_data:
        st.write("#### Current User Accounts")
        user_list = []
        for uname, info in users_data.items():
            user_list.append({
                "Username": uname,
                "Role": info.get("role", "N/A").capitalize(),
                "Active": "✅ Active" if info.get("active", True) else "❌ Inactive",
                "Last Login": info.get("last_login", "N/A")
            })
        st.dataframe(user_list, use_container_width=True)

        st.markdown("---")
        st.write("#### Change User Status")
        col_select_status, col_status_toggle, col_button_status = st.columns([2, 1, 1])

        with col_select_status:
            # Prevent current admin from deactivating themselves via this control
            users_to_manage_status = [u for u in users_data.keys()] # All users for selection
            selected_user_status = st.selectbox("Select User for Status Change", options=users_to_manage_status, key="select_user_to_manage_status")
        with col_status_toggle:
            current_active_status = users_data.get(selected_user_status, {}).get("active", True)
            new_status = st.checkbox(f"Set to Active", value=current_active_status, key=f"user_status_checkbox_{selected_user_status}")
        with col_button_status:
            st.write("") # For alignment
            st.write("")
            if st.button(f"Update Status", key=f"update_status_button_{selected_user_status}", use_container_width=True):
                # Ensure an admin cannot deactivate themselves if they are the sole active admin
                if selected_user_status == st.session_state.username and not new_status and users_data[selected_user_status]['role'] == 'admin':
                    active_admins = [u for u, info in users_data.items() if info.get('role') == 'admin' and info.get('active', True) and u != selected_user_status]
                    if not active_admins:
                        st.error("You cannot deactivate your own admin account if you are the sole active administrator.")
                    else:
                        success, msg = update_user_status(selected_user_status, new_status)
                        if success:
                            st.success(msg)
                            st.rerun() # Rerun to refresh the user list
                        else:
                            st.error(msg)
                else:
                    success, msg = update_user_status(selected_user_status, new_status)
                    if success:
                        st.success(msg)
                        st.rerun() # Rerun to refresh the user list
                    else:
                        st.error(msg)

        st.markdown("---")
        st.write("#### Delete User")
        col_select_delete, col_button_delete = st.columns([3, 1])

        with col_select_delete:
            # Prevent current admin from appearing in delete list to reinforce 'cannot delete self'
            users_to_delete = [u for u in users_data.keys() if u != st.session_state.username]
            if not users_to_delete:
                st.info("No other users to delete.")
            else:
                selected_user_delete = st.selectbox("Select User to Delete", options=users_to_delete, key="select_user_to_delete")
        with col_button_delete:
            st.write("") # For alignment
            st.write("")
            if st.button(f"Delete {selected_user_delete}", key=f"delete_user_button_{selected_user_delete}", use_container_width=True, type="secondary"): # Use type="secondary" for delete button
                if st.session_state.username == selected_user_delete: # Double check, though UI should prevent this
                    st.error("You cannot delete your own account.")
                elif users_data[selected_user_delete]['role'] == 'admin': # Additional check for deleting other admins
                    active_admins_after_delete = [u for u, info in users_data.items() if info.get('role') == 'admin' and info.get('active', True) and u != selected_user_delete]
                    if not active_admins_after_delete: # If no active admins remain after this deletion
                        st.error(f"Cannot delete '{selected_user_delete}'. Deleting this admin would leave no active administrator account.")
                    else:
                        success, msg = delete_user(selected_user_delete, st.session_state.username)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                else: # Non-admin user deletion
                    success, msg = delete_user(selected_user_delete, st.session_state.username)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
    else:
        st.info("No users registered yet.")

    st.markdown("---")


# ────────────────────────────────────────────────────────────────────────────────
# Main Application Content (General RAG Page)
# ────────────────────────────────────────────────────────────────────────────────
def main_rag_app_page():
    """
    Displays the core RAG functionalities.
    """
    st.markdown(f"<h1 style='text-align: center; color: var(--neon-primary);'>OpenLLM Insight Platform</h1>",
                unsafe_allow_html=True)
    st.markdown(
        f"<p style='text-align: center; color: var(--text);'>Welcome, {st.session_state.username}. Your access level: <b>{st.session_state.role.capitalize()}</b></p>",
        unsafe_allow_html=True)
    st.markdown("---")

    # 1) Upload + per-file FAISS ingestion + summarization (Admin Only)
    if st.session_state.role == 'admin':
        st.subheader("Document Management & Ingestion")
        st.markdown(
            "<p>Administrators can upload new documents (PDF/TXT) to expand the knowledge base. The system will automatically ingest them for QA and generate summaries.</p>",
            unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("### Upload Files")
            file_path: Path | None = upload_and_save_file()
            if file_path:
                stem = file_path.stem

                st.info(f"Ingesting `{file_path.name}` into FAISS. This may take a moment.")
                try:
                    ingest_to_faiss_per_file(file_path, base_dir="document_index")
                    st.success(f"Ingestion complete. Document index created at `document_index/{stem}`.")
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")
                    # st.stop() # Only stop if critical, otherwise allow app to continue with error message

                # automatic summary
                st.markdown("---")
                st.markdown("### Automatic Summary")
                st.write("Generating an automatic summary for the uploaded document:")
                try:
                    summary = summarize_file(file_path)
                    st.markdown(summary)
                    st.success("Summary generated successfully.")
                except Exception as e:
                    st.warning(
                        f"Summarization failed: {e}. Ensure your summarizer agent is configured correctly.")
        st.markdown("---")


    # 2) Per-document QA via router (with memory) - Accessible to all logged-in users
    index_root = Path("document_index")
    available = [d.name for d in index_root.iterdir() if d.is_dir()] if index_root.exists() else []

    st.subheader("Query Specific Documents")
    st.markdown("<p>Select a document from the dropdown to ask targeted questions and retrieve precise answers.</p>",
                unsafe_allow_html=True)

    if available:
        with st.container(border=True):
            chosen = st.selectbox("Choose document for QA", options=available, key="qa_specific_doc_select",
                                  help="Select a document to query.")
            question = st.text_input("Enter your question about the selected document:", key="qa_specific_doc_question",
                                     placeholder="e.g., What are the key findings?")

            if st.button("Run QA on Selected Document", key="run_qa_selected_button", use_container_width=True):
                upload_dir = Path("uploaded_files")
                matches = list(upload_dir.glob(f"{chosen}.*"))
                if not matches:
                    st.error(
                        f"No original uploaded file found for ‘{chosen}’. Ensure it was uploaded correctly.")
                else:
                    try:
                        with st.spinner(f"Analyzing '{chosen}' to find the answer..."):
                            answer = router(matches[0], question)
                            st.session_state.doc_qa_history.append((chosen, question, answer))
                            st.markdown(f"**Answer (from `{chosen}`):** \n{answer}")
                            st.success("Query complete.")
                    except Exception as e:
                        st.error(f"QA Error: {e}. Please check the document and your question.")
    else:
        st.info("No documents are currently indexed for specific QA. Please upload files (Admin only).")

    # display memory
    if st.session_state.doc_qa_history:
        st.write("#### Previous Per-Document Queries")
        with st.expander("View Document-Specific Q&A History"):
            for doc_stem, q, a in reversed(st.session_state.doc_qa_history):
                st.markdown(f"**Document:** `{doc_stem}`\n**Q:** {q}  \n**A:** {a}")
                st.markdown("---")

    st.markdown("---")

    # 3) 🌐 Global QA across all documents (with memory) - Accessible to all logged-in users
    st.subheader("Global Knowledge Base Query")
    st.markdown("<p>Ask a question across all available document summaries to gain comprehensive insights.</p>",
                unsafe_allow_html=True)

    with st.container(border=True):
        general_q = st.text_input("Enter your global question:", key="qa_global_question",
                                  placeholder="e.g., What are the common themes across all reports?")

        if st.button("Run Global QA", key="run_global_qa_button", use_container_width=True):
            summary_dir = Path("uploaded_files")
            summary_paths = list(summary_dir.glob("*.summary.txt"))
            if not summary_paths:
                st.error("No summary files found. Ensure documents have been uploaded and summarized.")
            else:
                docs = []
                for sp in summary_paths:
                    text = sp.read_text(encoding="utf-8")
                    stem = sp.stem.removesuffix(".summary")
                    docs.append(Document(page_content=text, metadata={"stem": stem}))

                api_key = os.getenv("OPENAI_KEY")
                if not api_key:
                    st.error(
                        "`OPENAI_KEY` environment variable is not set. Please set it to proceed with global QA.")
                    # st.stop() # Only stop if critical, otherwise allow app to continue with error message

                try:
                    with st.spinner("Building global knowledge base and finding relevant information..."):
                        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                        vs = FAISS.from_documents(docs, embeddings)
                        retriever = vs.as_retriever()

                        top = retriever.get_relevant_documents(general_q)[:1]
                        if not top:
                            st.warning("No relevant document found for your global question. Consider rephrasing.")
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
                                st.markdown(f"**Best Match Document:** \n`{chosen_stem}` \n**Answer:** {answer}")
                                st.success("Global query complete.")
                except Exception as e:
                    st.error(f"Global QA Error: {e}. Ensure your OpenAI API key is valid and agents are working.")

    # display global QA history
    if st.session_state.global_qa_history:
        st.write("#### Previous Global Queries")
        with st.expander("View Global Q&A History"):
            for q, stem, a in reversed(st.session_state.global_qa_history):
                st.markdown(f"**Q:** {q}  \n**Best Match:** `{stem}`  \n**A:** {a}")
                st.markdown("---")

    st.markdown("---")
    st.success("OpenLLM Insight Platform: Empowering knowledge discovery.")


# ────────────────────────────────────────────────────────────────────────────────
# Main App Flow: Authentication vs. Main App
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align: center; color: var(--primary-color);'>Welcome to OpenLLM</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: var(--text-color);'>Your secure gateway to powerful document intelligence.</p>",
    unsafe_allow_html=True)
st.markdown("---")

if st.session_state.logged_in:
    # --- Sidebar Navigation ---
    st.sidebar.markdown("### Navigation")
    # Determine which pages are available
    pages = ["Main App"]
    if st.session_state.role == 'admin':
        pages.append("Admin User Management")

    selected_page = st.sidebar.radio(
        "Go to",
        options=pages,
        index=pages.index(st.session_state.page) if st.session_state.page in pages else 0, # Set default to current page
        key="main_navigation_radio"
    )
    st.session_state.page = selected_page # Update session state with selected page

    # --- Render Page Content Based on Selection ---
    if st.session_state.page == "Main App":
        main_rag_app_page()
    elif st.session_state.page == "Admin User Management":
        if st.session_state.role == 'admin': # Double-check role for admin page access
            admin_user_management_page()
        else:
            st.error("Access Denied: You do not have permission to view this page.")
            st.session_state.page = "Main App" # Redirect non-admins back
            st.rerun()

    # --- User Profile and Logout always visible in sidebar when logged in ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### User Profile")
    st.sidebar.write(f"**Username:** `{st.session_state.username}`")
    st.sidebar.write(f"**Role:** `{st.session_state.role.capitalize()}`")

    # Moved logout and password change here to always be in sidebar regardless of selected page
    with st.sidebar.expander("Change Password"):
        st.write("Update your account password.")
        current_password = st.text_input("Current Password", type="password", key="sidebar_current_password_input")
        new_password = st.text_input("New Password", type="password", key="sidebar_new_password_input")
        confirm_new_password = st.text_input("Confirm New Password", type="password", key="sidebar_confirm_new_password_input")

        if st.button("Update Password", use_container_width=True, key="sidebar_update_password_button"):
            if not current_password or not new_password or not confirm_new_password:
                st.error("All password fields are required.")
            elif new_password != confirm_new_password:
                st.error("New password and confirmation do not match.")
            elif len(new_password) < 6: # Simple password strength check
                st.error("New password must be at least 6 characters long.")
            else:
                success, message = change_password(st.session_state.username, current_password, new_password)
                if success:
                    st.success(message)
                    # Clear inputs after successful change by resetting session state keys
                    st.session_state.sidebar_current_password_input = ""
                    st.session_state.sidebar_new_password_input = ""
                    st.session_state.sidebar_confirm_new_password_input = ""
                    st.rerun() # Rerun to ensure inputs clear visibly
                else:
                    st.error(message)

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        st.session_state.page = "Main App" # Reset page on logout
        st.rerun()

else: # Not logged in
    st.markdown("<h1 style='text-align: center; color: var(--primary-color);'>Welcome to OpenLLM</h1>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: var(--text-color);'>Your secure gateway to powerful document intelligence.</p>",
        unsafe_allow_html=True)
    st.markdown("---")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        login_page()
    with tab2:
        register_page()