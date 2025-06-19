# app.py
from dotenv import load_dotenv
import os
import streamlit as st
from pathlib import Path
import yaml  # Used for reading users.yaml to get admin count for display
import zipfile  # Import zipfile module for ZIP handling
import shutil  # For removing directories

# Import authentication functions
from auth import (
    register_user,
    authenticate_user,
    change_password,
    get_all_users_status,
    update_user_status,
    delete_user,
    get_admin_count,  # Import MAX_ADMIN_COUNT and get_admin_count
    MAX_ADMIN_COUNT
)

# Import your existing RAG functionalities
# Ensure these files and their dependencies (like OpenAI, Langchain, unstructured, pypdf, faiss-cpu) are installed
from upload import upload_and_save_files  # Corrected import for multi-file upload
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
    page_icon="✨",  # Restored a meaningful icon
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Inject Custom CSS from file ---
# Ensure 'styles' directory and 'style.css' exist.
# This approach makes the CSS easier to manage and modify externally.
styles_dir = Path("styles")
styles_dir.mkdir(exist_ok=True)  # Create styles directory if it doesn't exist
css_file_path = styles_dir / "style.css"

if not css_file_path.exists():
    # If the CSS file doesn't exist, create a placeholder and instruct the user.
    # The actual CSS content needs to be pasted into this newly created file.
    with open(css_file_path, "w") as f:
        f.write("/* Paste your CSS styles here from the provided block */\n")
    st.warning(
        f"'{css_file_path}' was not found and has been created. Please paste the CSS content provided into this file and restart the application.")
    st.stop()  # Stop the app to prompt user to add CSS

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
if "page" not in st.session_state:  # To manage current active page/tab in the main app
    st.session_state.page = "Main App"  # Default page after login/when not logged in

if "doc_qa_history" not in st.session_state:
    st.session_state.doc_qa_history = []
if "global_qa_history" not in st.session_state:
    st.session_state.global_qa_history = []


# ────────────────────────────────────────────────────────────────────────────────
# Authentication Pages
# ────────────────────────────────────────────────────────────────────────────────
def login_page():
    """
    Renders the login interface with a professional layout.
    """
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>Secure Access</h2>", unsafe_allow_html=True)
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
                    st.session_state.page = "Main App"  # Redirect to main app after login
                    st.rerun()
                else:
                    st.error(message)

    st.markdown("---")
    st.info("New to OpenLLM? Register for an account using the 'Register' tab.")


def register_page():
    """
    Renders the registration interface with a professional layout,
    including role selection and admin cap logic.
    """
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>Create OpenLLM Account</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center;'>Join our platform by creating a new user account. Select your desired role below.</p>",
            unsafe_allow_html=True)

        current_admin_count = get_admin_count()
        admin_slots_available = MAX_ADMIN_COUNT - current_admin_count

        # Prepare role options and default selection
        role_options_labels = ["QA User"]  # QA User is always an option
        default_role_index = 0  # Default to QA User

        if admin_slots_available > 0:
            role_options_labels.insert(0, "Admin")  # Add Admin as the first option if available
            st.info(f"Admin slots available: **{admin_slots_available}** out of **{MAX_ADMIN_COUNT}**.")
            default_role_index = 0  # Admin will be the default pre-selection if available
        # The explicit warning when admin slots are full is removed as per previous request.
        # UI dynamically removes the 'Admin' option instead.

        with st.form("register_form"):
            new_username = st.text_input("New Username", key="register_username_input",
                                         placeholder="Choose a unique username")
            new_password = st.text_input("New Password", type="password", key="register_password_input",
                                         placeholder="Create a strong password")

            selected_role_label = st.radio(
                "Select your desired role:",
                options=role_options_labels,
                index=default_role_index,  # Set default selection based on availability
                key="register_role_select"
            )
            # Map the selected label back to the internal role string ('admin' or 'qa')
            selected_role = "admin" if selected_role_label == "Admin" else "qa"

            st.markdown("---")
            register_button = st.form_submit_button("Register Account", use_container_width=True)

            if register_button:
                # Pass the selected_role to the register_user function
                success, message = register_user(new_username, new_password, selected_role)
                if success:
                    st.success(message)
                    st.info("Account created successfully! You can now log in.")
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
    st.markdown(f"<h1 style='text-align: center; color: var(--primary-color);'>Admin User Management</h1>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center;'>View and manage user accounts: activate, deactivate, or delete users. Exercise caution with admin accounts.</p>",
        unsafe_allow_html=True)
    st.markdown("---")

    users_data = get_all_users_status()  # Get all user data
    if users_data:
        st.write("#### Current User Accounts")
        user_list_for_df = []
        for uname, info in users_data.items():
            user_list_for_df.append({
                "Username": uname,
                "Role": info.get("role", "N/A").capitalize(),
                "Active": "Active" if info.get("active", True) else "Inactive",
                "Created At": info.get("created_at", "N/A"),
                "Last Login": info.get("last_login", "N/A")
            })
        st.dataframe(user_list_for_df, use_container_width=True)

        st.markdown("---")
        st.write("#### Change User Status")
        col_select_status, col_status_toggle = st.columns([3, 2])  # Adjusted columns for better layout

        with col_select_status:
            # Users the current admin can change status for (cannot change their own via this dropdown)
            users_for_status_change = [u for u in
                                       users_data.keys()]  # All users for selection, logic handles self-deactivation
            selected_user_status = st.selectbox("Select User to Change Status", options=users_for_status_change,
                                                key="select_user_to_manage_status")

        # Only show controls if a user is selected
        if selected_user_status:
            current_active_status = users_data.get(selected_user_status, {}).get("active", True)

            with st.container(border=True):  # Use a container for the form-like input
                st.markdown(
                    f"**Current Status of '{selected_user_status}':** {'Active' if current_active_status else 'Inactive'}")
                new_status = st.checkbox(f"Set '{selected_user_status}' to Active", value=current_active_status,
                                         key=f"user_status_checkbox_{selected_user_status}")

                if st.button(f"Apply Status Change", key=f"update_status_button_{selected_user_status}",
                             use_container_width=True):
                    # Prevent admin from deactivating themselves if they are the sole active admin
                    if selected_user_status == st.session_state.username and not new_status:
                        active_admins_count = sum(1 for u, info in users_data.items() if
                                                  info.get('role') == 'admin' and info.get('active',
                                                                                           True) and u != selected_user_status)
                        if users_data[selected_user_status]['role'] == 'admin' and active_admins_count == 0:
                            st.error(
                                "You cannot deactivate your own admin account if you are the sole active administrator remaining.")
                            st.rerun()  # Rerun to show error and keep status as is
                        else:
                            success, msg = update_user_status(selected_user_status, new_status)
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    else:
                        success, msg = update_user_status(selected_user_status, new_status)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

        st.markdown("---")
        st.write("#### Delete User Account")
        col_select_delete, col_button_delete = st.columns([3, 1])

        with col_select_delete:
            # Prevent current admin from appearing in delete list to reinforce 'cannot delete self'
            users_to_delete_options = [u for u in users_data.keys() if u != st.session_state.username]
            if not users_to_delete_options:
                st.info("No other user accounts to delete.")
                selected_user_delete = None  # No user to select
            else:
                selected_user_delete = st.selectbox("Select User to Permanently Delete",
                                                    options=users_to_delete_options, key="select_user_to_delete")

        if selected_user_delete:
            with col_button_delete:
                st.write("")  # For alignment
                st.write("")
                if st.button(f"Delete '{selected_user_delete}'", key=f"delete_user_button_{selected_user_delete}",
                             use_container_width=True, type="secondary"):
                    # Additional checks before actual deletion
                    # Check if deleting another admin would leave no active admins
                    if users_data[selected_user_delete]['role'] == 'admin':
                        active_admins_count = sum(
                            1 for u, info in users_data.items()
                            if info.get('role') == 'admin' and info.get('active', True) and u != selected_user_delete
                            # Exclude user being deleted
                        )
                        if active_admins_count == 0:
                            st.error(
                                f"Cannot delete '{selected_user_delete}'. Deleting this admin would leave no active administrator account.")
                            st.rerun()
                            return  # Stop further execution

                    # Proceed with deletion if checks pass
                    success, msg = delete_user(selected_user_delete, st.session_state.username)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
    else:
        st.info("No users registered yet to manage.")

    st.markdown("---")


# ────────────────────────────────────────────────────────────────────────────────
# Main Application Content (General RAG Page)
# ────────────────────────────────────────────────────────────────────────────────
def main_rag_app_page():
    """
    Displays the core RAG functionalities.
    """
    st.markdown(f"<h1 style='text-align: center; color: var(--primary-color);'>OpenLLM Insight Platform</h1>",
                unsafe_allow_html=True)
    st.markdown(
        f"<p style='text-align: center; color: var(--text-color);'>Welcome, {st.session_state.username}! Your access level: <b>{st.session_state.role.capitalize()}</b></p>",
        unsafe_allow_html=True)
    st.markdown("---")

    # 1) Upload + per-file FAISS ingestion + summarization (Admin Only)
    if st.session_state.role == 'admin':
        st.subheader("⬆ Document Management & Ingestion")
        st.markdown(
            "<p>Administrators can upload new documents (PDF, TXT, DOCX, CSV, XLSX, Images) to expand the knowledge base. The system will automatically ingest them for QA and generate summaries.</p>",
            unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("### Upload Documents")
            st.info(
                "To upload an entire folder of documents, please compress it into a **.zip file** and upload it below. This will unpack all contained files for processing.")

            zip_file = st.file_uploader(
                "Upload a folder as ZIP",
                type="zip",
                help="Compress your folder of documents into a single .zip file before uploading. All supported files inside will be processed.",
                key="zip_uploader"
            )

            if zip_file:
                # Create a temporary extraction directory within uploaded_files
                # This temporary directory ensures isolation during extraction
                zip_name_stem = Path(zip_file.name).stem  # get name without .zip
                temp_extract_dir = Path("uploaded_files") / f"temp_zip_{zip_name_stem}_{os.urandom(4).hex()}"
                temp_extract_dir.mkdir(parents=True, exist_ok=True)

                st.info(f"Extracting '{zip_file.name}' to a temporary location for processing...")
                try:
                    with zipfile.ZipFile(zip_file, "r") as z:
                        z.extractall(temp_extract_dir)  # Extracts with internal structure into temp_extract_dir

                    # List all files extracted, including those in subdirectories
                    files_to_process = [p for p in temp_extract_dir.rglob("*") if p.is_file()]
                    st.success(f"Successfully extracted {len(files_to_process)} items from '{zip_file.name}'.")

                    processed_count = 0
                    failed_files = []

                    # Target directory for the final flattened files
                    target_upload_dir = Path("uploaded_files")

                    for doc_path_in_temp_dir in files_to_process:
                        # Construct the target path in the main uploaded_files directory
                        # We use doc_path_in_temp_dir.name to get just the filename, effectively flattening the structure
                        final_doc_path = target_upload_dir / doc_path_in_temp_dir.name

                        # Handle potential filename conflicts (if a file with the same name already exists in uploaded_files)
                        # Append a counter for uniqueness
                        counter = 1
                        original_final_doc_path = final_doc_path
                        while final_doc_path.exists():
                            final_doc_path = original_final_doc_path.parent / f"{original_final_doc_path.stem}_{counter}{original_final_doc_path.suffix}"
                            counter += 1

                        st.markdown(f"---")
                        st.info(
                            f"Processing extracted file: `{doc_path_in_temp_dir.relative_to(temp_extract_dir)}` (will be saved as `{final_doc_path.name}`)")

                        try:
                            # Move the file from the temporary extraction location to the flattened uploaded_files directory
                            shutil.move(str(doc_path_in_temp_dir), str(final_doc_path))

                            # Ingestion
                            ingest_to_faiss_per_file(final_doc_path, base_dir="document_index")
                            st.success(f"Ingested `{final_doc_path.name}` into FAISS.")

                            # Always attempt summarization
                            with st.spinner(f"Generating summary for `{final_doc_path.name}`..."):
                                try:
                                    summary = summarize_file(final_doc_path)
                                    st.markdown(f"**Summary for `{final_doc_path.name}`:**\n{summary}")
                                    st.success(f"Summary generated for `{final_doc_path.name}`.")
                                except Exception as sum_e:
                                    st.warning(
                                        f"⚠ Summarization failed for `{final_doc_path.name}`: {sum_e}. Please ensure your summarizer agent can handle this file type.")

                            processed_count += 1
                        except Exception as e:
                            st.error(f"Failed to process `{doc_path_in_temp_dir.name}`: {e}")
                            failed_files.append(doc_path_in_temp_dir.name)
                            continue

                    if failed_files:
                        st.warning(
                            f"⚠ Completed ZIP processing with some failures. Failed files: {', '.join(failed_files)}")
                    else:
                        st.success(f"All {processed_count} files from '{zip_file.name}' processed successfully!")

                except zipfile.BadZipFile:
                    st.error("The uploaded file is not a valid ZIP archive.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during ZIP extraction: {e}")
                finally:
                    # Always clean up the temporary extraction directory
                    if temp_extract_dir.exists():
                        st.info(f"Cleaning up temporary extracted directory: `{temp_extract_dir}`...")
                        try:
                            shutil.rmtree(temp_extract_dir)
                            st.success("Temporary directory cleaned up.")
                        except Exception as e:
                            st.warning(f"⚠ Could not remove temporary directory `{temp_extract_dir}`: {e}")

            st.markdown("---")
            st.markdown("### Upload Individual Files")
            st.info("Alternatively, you can select one or more individual files from different locations.")
            # Existing multi-file uploader for individual files
            uploaded_file_paths: list[Path] = upload_and_save_files()

            if uploaded_file_paths:
                st.write(f"Initiating processing for {len(uploaded_file_paths)} individual file(s)...")
                for file_path in uploaded_file_paths:  # Iterate through each uploaded file
                    stem = file_path.stem

                    st.info(f"▶ Ingesting `{file_path.name}` into FAISS… This may take a moment.")
                    try:
                        ingest_to_faiss_per_file(file_path, base_dir="document_index")
                        st.success(
                            f"Ingestion complete for `{file_path.name}`. Index created at `document_index/{stem}`.")
                    except Exception as e:
                        st.error(f"Error during ingestion for `{file_path.name}`: {e}")
                        # Continue to next file even if one fails
                        continue

                    # Always attempt summarization
                    with st.spinner(f"Generating summary for `{file_path.name}`..."):
                        try:
                            summary = summarize_file(file_path)
                            st.markdown(f"**Summary for `{file_path.name}`:**\n{summary}")
                            st.success(f"Summary generated for `{file_path.name}`.")
                        except Exception as sum_e:
                            st.warning(
                                f"⚠ Summarization failed for `{file_path.name}`: {sum_e}. Please ensure your summarizer agent can handle this file type.")
                st.success(
                    "All selected individual files have been processed for ingestion and summarization (if applicable).")

        st.markdown("---")
    else:
        st.subheader("Document Upload Access")
        st.warning(
            "You do not have permission to upload documents. This feature is restricted to **Admin** users to maintain data integrity.")

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
            question = st.text_input("Enter your question about the selected document:",
                                     key="qa_specific_doc_question",
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

    # 3) Global QA across all documents (with memory) - Accessible to all logged-in users
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

                api_key = os.getenv("OPENAI_API_KEY")  # Ensure this matches your .env key name
                if not api_key:
                    st.error(
                        "`OPENAI_API_KEY` environment variable is not set. Please set it to proceed with global QA.")
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
# Display a welcoming header that appears before login/register tabs
st.markdown("<h1 style='text-align: center; color: var(--primary-color);'>Welcome to OpenLLM</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: var(--text-color);'>Your secure gateway to powerful document intelligence.</p>",
    unsafe_allow_html=True)
st.markdown("---")

if st.session_state.logged_in:
    # --- Sidebar Navigation ---
    st.sidebar.markdown("### Navigation")
    # Determine which pages are available based on role
    pages = ["Main App"]
    if st.session_state.role == 'admin':
        pages.append("Admin User Management")

    selected_page = st.sidebar.radio(
        "Go to",
        options=pages,
        # Set default to current page or 'Main App' if current page is no longer valid for the role
        index=pages.index(st.session_state.page) if st.session_state.page in pages else 0,
        key="main_navigation_radio"
    )
    st.session_state.page = selected_page  # Update session state with selected page

    # --- Render Page Content Based on Selection ---
    if st.session_state.page == "Main App":
        main_rag_app_page()
    elif st.session_state.page == "Admin User Management":
        if st.session_state.role == 'admin':  # Double-check role for admin page access
            admin_user_management_page()
        else:
            # This should ideally not happen if navigation is controlled, but as a fallback
            st.error("Access Denied: You do not have permission to view this page.")
            st.session_state.page = "Main App"  # Redirect non-admins back
            st.rerun()  # Rerun to force redirection

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
        confirm_new_password = st.text_input("Confirm New Password", type="password",
                                             key="sidebar_confirm_new_password_input")

        if st.button("Update Password", use_container_width=True, key="sidebar_update_password_button"):
            if not current_password or not new_password or not confirm_new_password:
                st.error("All password fields are required.")
            elif new_password != confirm_new_password:
                st.error("New password and confirmation do not match.")
            elif len(new_password) < 6:  # Simple password strength check
                st.error("New password must be at least 6 characters long.")
            else:
                success, message = change_password(st.session_state.username, current_password, new_password)
                if success:
                    st.success(message)
                    # Clear inputs after successful change by resetting session state keys
                    st.session_state["sidebar_current_password_input"] = ""
                    st.session_state["sidebar_new_password_input"] = ""
                    st.session_state["sidebar_confirm_new_password_input"] = ""
                    st.rerun()  # Rerun to ensure inputs clear visibly
                else:
                    st.error(message)

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        st.session_state.page = "Main App"  # Reset page on logout
        st.rerun()

else:  # Not logged in, show login/register tabs
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
