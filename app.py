# app.py

import streamlit as st
import os
import shutil
import time
from typing import List, Dict, Any
import sys

# Ensure we can import from the agents and utils directories
# This is crucial for local development and deployment where Python might not find modules directly.
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, 'agents'))
sys.path.append(os.path.join(current_dir, 'utils'))

# Import core components from our agentic system
from agents.coordinator_agent import CoordinatorAgent
from agents.mcp_messages import FinalResponsePayload, QueryRequestPayload # Import QueryRequestPayload for type hinting

# --- Configuration ---
# Directory for temporarily saving uploaded documents
UPLOAD_DIR = "uploaded_docs"
# Directory for documents automatically loaded on application startup (e.g., project documentation)
DEFAULT_DOCS_DIR = "docs"
# Directory where the vector store (FAISS index and chunk data) will be persisted
VECTOR_STORE_PERSIST_DIR = "vector_store_data"

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
# The VECTOR_STORE_PERSIST_DIR is handled by the RetrievalAgent's initialization.
# The DEFAULT_DOCS_DIR is expected to be part of the project's git repo.


# --- Streamlit UI Setup ---
st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")
st.title("📚 Agentic RAG Chatbot for Multi-Format Documents")
st.markdown("""
    Upload your documents (PDF, DOCX, PPTX, CSV, TXT, MD) and ask questions.
    This chatbot utilizes an **Agent-Based Retrieval-Augmented Generation (RAG)** architecture
    with a **Model Context Protocol (MCP)** for inter-agent communication.
    It supports multi-turn conversations and persists the knowledge base across sessions.
""")

# --- Initialize CoordinatorAgent (Singleton Pattern) ---
# The CoordinatorAgent and its associated state (like chat history, uploaded files)
# are stored in Streamlit's session_state to ensure persistence across reruns
# without re-initializing expensive components (like the embedding model or vector store).
if "coordinator" not in st.session_state:
    # Initialize the CoordinatorAgent, passing the directory for vector store persistence.
    st.session_state.coordinator = CoordinatorAgent(vector_store_dir=VECTOR_STORE_PERSIST_DIR)
    st.session_state.chat_history = [] # Stores user and assistant messages for display
    st.session_state.uploaded_file_names = [] # Tracks names of all ingested files (default + uploaded)
    st.session_state.last_uploaded_files = [] # Helps prevent re-processing `st.file_uploader` duplicates
    st.session_state.default_ingestion_done = False # Flag to ensure default docs are ingested only once per session
    st.session_state.file_uploader_key = 0 # Initialize a key for the file uploader

# Access the coordinator instance from session state for consistent use
coordinator: CoordinatorAgent = st.session_state.coordinator

# --- Default Document Ingestion (runs once per fresh session) ---
# This block attempts to load documents from the 'docs' folder.
# It only runs if 'default_ingestion_done' flag is False, meaning it's a new session or after a full reset.
if not st.session_state.default_ingestion_done:
    if os.path.exists(DEFAULT_DOCS_DIR):
        default_files_to_ingest = []
        for f_name in os.listdir(DEFAULT_DOCS_DIR):
            file_path = os.path.join(DEFAULT_DOCS_DIR, f_name)
            if os.path.isfile(file_path):
                # Only add if the file hasn't been tracked yet in this session
                if f_name not in st.session_state.uploaded_file_names:
                    default_files_to_ingest.append(file_path)
                    st.session_state.uploaded_file_names.append(f_name)
                    st.session_state.last_uploaded_files.append(f_name) # Also add to prevent re-upload detection

        if default_files_to_ingest:
            st.sidebar.info(f"Automatically loading default documents from '{DEFAULT_DOCS_DIR}'...")
            doc_paths_for_ingestion = [
                {"path": file_path, "type": os.path.basename(file_path).split('.')[-1]}
                for file_path in default_files_to_ingest
            ]

            # Display a spinner while processing default documents
            with st.spinner("Processing default documents... This may take a moment."):
                ingestion_trace_id = st.session_state.coordinator.handle_document_upload(doc_paths_for_ingestion)
                # Small synchronous wait to allow agents to process and save the vector store
                # In a production async setup, you'd poll for a specific 'INGESTION_COMPLETE' message.
                time.sleep(2) # Adjust based on expected ingestion time

            st.sidebar.success(f"Default documents loaded successfully.")
    # Set the flag to True so this block doesn't re-execute on subsequent reruns within the same session
    st.session_state.default_ingestion_done = True


# --- Document Upload Section (for user-uploaded files) ---
st.sidebar.header("Upload New Documents")
uploaded_files = st.sidebar.file_uploader(
    "Choose additional files to add to the knowledge base:",
    type=["pdf", "docx", "pptx", "csv", "txt", "md"],
    accept_multiple_files=True,
    key=st.session_state.file_uploader_key # --- ADD THIS KEY ---
)

if uploaded_files:
    new_files_to_process = []
    for uploaded_file in uploaded_files:
        # Check if this file has already been processed in the current session
        if uploaded_file.name not in st.session_state.last_uploaded_files:
            new_files_to_process.append(uploaded_file)
            st.session_state.last_uploaded_files.append(uploaded_file.name)

            # Save the uploaded file to the UPLOAD_DIR
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"Saved: {uploaded_file.name}")

    if new_files_to_process:
        with st.spinner("Processing new uploaded documents... This may take a moment."):
            doc_paths_for_ingestion = [
                {"path": os.path.join(UPLOAD_DIR, f.name), "type": f.name.split('.')[-1]}
                for f in new_files_to_process
            ]

            # Initiate ingestion for the newly uploaded documents
            ingestion_trace_id = coordinator.handle_document_upload(doc_paths_for_ingestion)
            st.sidebar.info(f"Ingestion initiated for new files (Trace ID: {ingestion_trace_id}).")

            # Add new file names to the list of all uploaded files for display
            st.session_state.uploaded_file_names.extend([f.name for f in new_files_to_process])

            # Small synchronous wait to allow agents to process and save.
            time.sleep(2) # Adjust based on expected ingestion time

# --- Display Current Documents in Store ---
st.sidebar.markdown("---")
st.sidebar.header("Current Documents in Knowledge Base")
if st.session_state.uploaded_file_names:
    # Display unique file names, sorted alphabetically
    for f_name in sorted(list(set(st.session_state.uploaded_file_names))):
        st.sidebar.write(f"- {f_name}")
else:
    st.sidebar.write("No documents loaded yet.")

# --- Clear Data & Restart Button ---
st.sidebar.markdown("---")
if st.sidebar.button("Clear All Data & Restart"):
    # Clear the persistent vector store data from disk
    coordinator.clear_vector_store()

    # Remove all uploaded files from the UPLOAD_DIR
    if os.path.exists(UPLOAD_DIR):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Clearing uploaded files from {UPLOAD_DIR}...")
        shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR) # Recreate the empty directory
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Uploaded files cleared and {UPLOAD_DIR} recreated.")

    # --- ADD THIS LINE TO RESET THE FILE UPLOADER STATE ---
    st.session_state.file_uploader_key += 1 # Increment the key to force re-render/reset

    # Reset Streamlit's session state to effectively restart the application
    st.session_state.coordinator = CoordinatorAgent(vector_store_dir=VECTOR_STORE_PERSIST_DIR) # Re-initialize
    st.session_state.chat_history = []
    st.session_state.uploaded_file_names = []
    st.session_state.last_uploaded_files = []
    st.session_state.default_ingestion_done = False # Reset flag to allow default ingestion on next run
    st.rerun() # Force Streamlit to rerun the script from top

# --- Chat Interface ---
st.subheader("Chat with your Documents")

# Display historical chat messages
for msg_idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If the message has source chunks (i.e., it's an assistant's RAG response)
        if "source_chunks" in message and message["source_chunks"]:
            with st.expander("Show Sources"):
                for i, chunk_content in enumerate(message["source_chunks"]):
                    unique_key = f"source_chunk_{msg_idx}_{i}" # Unique key for each text_area
                    # Display chunk content, with a fallback message if content is empty/whitespace
                    display_chunk = chunk_content if chunk_content and len(chunk_content.strip()) > 0 else f"No content available for Source Chunk {i+1} (possibly empty extraction)."
                    st.text_area(f"Source Chunk {i+1}", display_chunk, height=100, disabled=True, key=unique_key)


# Handle user input for new chat messages
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user's message to chat history immediately
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Use a placeholder for the assistant's response to update it later
    # This allows the spinner to show, then the final text, and then we re-render
    # the history to include sources.
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Create an empty container for the assistant's response
        
        with st.spinner("Thinking..."):
            conversation_history_for_llm = []
            for msg in st.session_state.chat_history[:-1]:
                conversation_history_for_llm.append({"role": msg["role"], "content": msg["content"]})
            
            query_trace_id = coordinator.handle_query(prompt, conversation_history=conversation_history_for_llm)
            
            final_response: FinalResponsePayload | None = None
            max_attempts = 60
            attempts = 0
            while final_response is None and attempts < max_attempts:
                response_msg = coordinator.get_ui_response(query_trace_id)
                if response_msg and response_msg["type"] == "FINAL_RESPONSE":
                    final_response = response_msg["payload"]
                    break
                elif response_msg and response_msg["type"] == "ERROR":
                    error_message = response_msg['payload'].get('message', 'An unknown error occurred during processing.')
                    st.error(f"An error occurred: {error_message}")
                    final_response = {"answer": "Sorry, an internal error occurred while processing your request.", "source_chunks": []}
                    break
                time.sleep(0.5)
                attempts += 1
            
            if final_response:
                # Update the placeholder with the answer
                message_placeholder.markdown(final_response["answer"]) 
                
                # Add assistant's response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final_response["answer"],
                    "source_chunks": final_response["source_chunks"]
                })
            else:
                fallback_answer = "I'm sorry, I couldn't get a response from the agents. They might be busy, or an internal issue occurred."
                # Update the placeholder with the fallback answer
                message_placeholder.markdown(fallback_answer)
                st.session_state.chat_history.append({"role": "assistant", "content": fallback_answer, "source_chunks": []})
   
    # Rerender the chat history to include the new message and sources
    st.rerun()