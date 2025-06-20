# OpenLLM Insight Platform

---

## Project Overview

The OpenLLM Insight Platform is a secure, interactive web application built with Streamlit, designed to leverage Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) for advanced document intelligence. This platform enables users to upload, ingest, summarize, and query documents, facilitating deep insights from their knowledge base. Featuring robust user authentication and administrative controls, it provides a secure and efficient environment for knowledge discovery.

---

## Key Features

* **Secure User Authentication:**
    * **User Registration:** New users can securely register. The initial registered user is automatically designated as an **Administrator**, with subsequent users assigned a **QA User** role.
    * **User Login/Logout:** Provides secure access to the platform via authenticated credentials.
    * **Password Management:** Users can update their account passwords post-login.
    * **Account Status Control:** Administrators possess the ability to activate or deactivate user accounts.
* **Admin User Management:** (Accessible exclusively to Administrators via a dedicated navigation link)
    * **User Overview:** Displays a comprehensive list of all registered users, including their active status and assigned roles.
    * **Status Modification:** Allows administrators to change user account statuses (active/inactive).
    * **User Deletion:** Provides functionality to permanently delete user accounts, incorporating safeguards to prevent the deletion of the sole active administrator or the currently logged-in administrator's account.
* **Document Management & Ingestion (Administrator Exclusive):**
    * **File Upload:** Supports secure upload of PDF and TXT documents.
    * **FAISS Ingestion:** Automatically processes uploaded documents, creating dedicated FAISS vector stores for each file to enable efficient information retrieval.
    * **Automated Summarization:** Generates concise summaries of uploaded documents utilizing LLMs.
* **Intelligent Document Query (RAG):**
    * **Per-Document QA:** Users can pose specific questions against individual uploaded documents, leveraging their unique vector indexes for precise answers.
    * **Global Knowledge Base Query:** Facilitates cross-document querying by analyzing all uploaded document summaries to identify overarching themes, aggregate information, and pinpoint the most relevant source for a given query.
    * **Query History:** Maintains a clear record of both document-specific and global queries for user reference.

---

## Technology Stack

* **Web Framework:** Streamlit
* **Authentication:** Custom system utilizing `bcrypt` for secure password hashing.
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **LLM Orchestration:** LangChain (for RAG workflows, agent integration)
* **Embeddings & LLM Provider:** OpenAI Embeddings / OpenAI GPT models (via `langchain-openai`)
* **Document Parsing:** `unstructured`, `pypdf2`
* **Environment Management:** `python-dotenv`
* **Data Serialization:** `PyYAML`
* **Styling:** Custom CSS

---

## Project Structure
```text
.
├── app.py                      # Main Streamlit application entry point
├── auth.py                     # Core authentication functions (registration, login, password changes, user management)
├── upload.py                   # Handles document upload and storage
├── ingestion.py                # Manages vector store creation from documents
├── agents/
│   ├── summarizer.py           # LLM agent responsible for document summarization
│   └── router_agent_doc.py     # LLM agent for intelligent question routing to documents
├── styles/
│   └── style.css               # Custom cascading style sheet for UI customization
├── pages/
│   └── 01_User_Management.py   # Dedicated Streamlit page for Administrator-level user controls
├── document_index/             # Directory for storing document-specific FAISS vector indexes
│   └── &lt;document_stem>/
│       └── ...
├── uploaded_files/             # Directory for storing original uploaded documents and their generated summaries
│   └── &lt;filename>.&lt;ext>
│   └── &lt;filename>.summary.txt
├── users.yaml                  # YAML file for persistent storage of user credentials and roles
├── .env.example                # Template for required environment variables
├── .env                        # Environment variables configuration (e.g., OPENAI_API_KEY)
└── requirements.txt            # Project's Python dependency list
```

---

## Setup and Installation

Follow these instructions to set up and run the OpenLLM Insight Platform:

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd openllm-insight-platform # Navigate to your project directory
```
### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

Install all necessary Python packages:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a file named .env in your project's root directory. Add your OpenAI API key to it like this:

```Ini, TOML
OPENAI_API_KEY = "your_openai_api_key_here"
```
Note: Remember to replace "your_openai_api_key_here" with your actual OpenAI API key.

### 5. Run the Application

Launch the Streamlit application from your terminal:

```bash
streamlit run app.py
```

Your web browser should automatically open to the application, usually at http://localhost:8501.

# Initial Setup & Administrator Account

When you launch the application and register your first user:

1. Go to the **Register** tab.  
2. The first account registered will automatically be assigned the **admin** role.  
3. All subsequent registrations will create **qa_user** accounts.  
4. If you ever need to set up a new administrator (e.g., if you delete the existing `users.yaml`), simply delete the `users.yaml` file and register a new user.

---

# Usage Guide

## Login and Navigation

- Access the platform using the **Login** tab with your registered credentials.  
- Once logged in, your sidebar will show:
  - **User Profile** (username, role)  
  - **Change Password**  
  - **Logout** button  
- **Administrators** will also find an **Admin User Management** link in the sidebar’s **Navigation** section, providing access to user controls.

## Document Management (Administrators Only)

On the **Main App** page, administrators can:

1. Use the **Document Management & Ingestion** section to upload PDF or TXT files.  
2. The system will automatically:
   - Create a dedicated FAISS index for each file.  
   - Generate and store a summary for each document.

## Querying Documents

1. **Query Specific Documents**  
   - Select an available document from the dropdown under **Query Specific Documents**.  
   - Type your question to get an answer based solely on that document.

2. **Global Knowledge Base Query**  
   - Use the **Global Knowledge Base Query** section to ask questions across all summarized documents.  
   - The platform will identify the most relevant summary and provide a comprehensive answer.

---

# Contributing

Contributions are highly valued! If you have suggestions for improvements, new features, or bug fixes, please:

1. **Fork** the repository.  
2. Create a new branch:  
   ```bash
   git checkout -b feature/your-feature-name
    ```