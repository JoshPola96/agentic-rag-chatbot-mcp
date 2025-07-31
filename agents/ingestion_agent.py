# agents/ingestion_agent.py

import os
import uuid # To generate unique IDs for each chunk
from typing import List, Dict, Any, Callable

# Import the base agent class and MCP message definitions
from agents.base_agent import Agent
from agents.mcp_messages import MCPMessage, create_mcp_message, DocumentUploadPayload, IngestionCompletePayload, ErrorPayload
from utils.document_parser import parse_document # Utility to extract text from various document types
from utils.text_splitter import chunk_text # Utility to split text into manageable chunks
import time
import shutil

class IngestionAgent(Agent):
    """
    The IngestionAgent is responsible for processing raw documents.
    Its tasks include:
    1. Receiving 'DOCUMENT_UPLOAD' messages from the CoordinatorAgent (triggered by the UI).
    2. Parsing the content from specified document paths using the document_parser utility.
    3. Chunking the extracted text into smaller, semantically meaningful units using the text_splitter utility.
    4. Attaching metadata (like source file info) to each chunk.
    5. Sending an 'INGESTION_COMPLETE' message containing all processed chunks to the RetrievalAgent.
    6. Handling errors during parsing or chunking and reporting them.
    """
    def __init__(self, message_broker: Callable[[MCPMessage], None]):
        """
        Initializes the IngestionAgent.

        Args:
            message_broker (Callable[[MCPMessage], None]): The method to send messages to the Coordinator.
        """
        super().__init__("IngestionAgent", message_broker)
        print(f"[{self.name}] Initialized.")

    def process_message(self, message: MCPMessage):
        """
        Processes incoming MCP messages for the IngestionAgent.
        It primarily expects messages of type "DOCUMENT_UPLOAD".

        Args:
            message (MCPMessage): The incoming message dictionary.
        """
        # Ensure the message is intended for this agent
        if message["receiver"] != self.name:
            return

        trace_id = message.get("trace_id", "N/A") # Get trace_id for logging/error reporting
        sender = message["sender"] # Get sender for sending back error messages

        print(f"[{self.name}] Received message of type: {message['type']} (Trace ID: {trace_id})")

        if message["type"] == "DOCUMENT_UPLOAD":
            try:
                payload: DocumentUploadPayload = message["payload"]
                document_paths_info: List[Dict[str, str]] = payload.get("document_paths", [])
                
                all_processed_chunks: List[Dict[str, Any]] = [] # To store all structured chunks

                if not document_paths_info:
                    print(f"[{self.name}] No document paths provided in DOCUMENT_UPLOAD message (Trace ID: {trace_id}).")
                    # Send an error message if no documents were provided
                    self.send_message(
                        receiver=sender,
                        msg_type="ERROR",
                        payload={"message": "No document paths provided for upload.", "original_trace_id": trace_id},
                        trace_id=trace_id
                    )
                    return # Exit early if no documents to process
                
                for doc_info in document_paths_info:
                    file_path = doc_info.get("path")
                    file_type = doc_info.get("type")

                    if not file_path or not file_type:
                        print(f"[{self.name}] Skipping document due to missing 'path' or 'type' in info: {doc_info}")
                        # Could send a more specific error for this malformed doc_info
                        continue

                    if not os.path.exists(file_path):
                        print(f"[{self.name}] Skipping non-existent file: {file_path}")
                        # Optionally, send an error for each non-existent file
                        self.send_message(
                            receiver=sender,
                            msg_type="ERROR",
                            payload={"message": f"File not found: {os.path.basename(file_path)}", "original_trace_id": trace_id},
                            trace_id=trace_id
                        )
                        continue

                    print(f"[{self.name}] Parsing and chunking {file_path} (Type: {file_type})...")
                    content = parse_document(file_path, file_type)
                    
                    if content:
                        # Chunk the extracted content
                        raw_chunks: List[str] = chunk_text(content, chunk_size=500, chunk_overlap=100)
                        
                        # Convert raw string chunks into structured dictionaries with metadata
                        structured_chunks = []
                        for i, chunk_content in enumerate(raw_chunks):
                            chunk_id = str(uuid.uuid4()) # Generate a unique ID for each chunk
                            structured_chunks.append({
                                "id": chunk_id,
                                "content": chunk_content,
                                "metadata": {
                                    "source_filename": os.path.basename(file_path),
                                    "original_filepath": file_path,
                                    "file_type": file_type,
                                    "chunk_index": i # Helpful for debugging/ordering
                                }
                            })
                        
                        all_processed_chunks.extend(structured_chunks)
                        print(f"[{self.name}] Extracted {len(content)} characters, created {len(structured_chunks)} chunks from {os.path.basename(file_path)}.")
                    else:
                        print(f"[{self.name}] Could not extract content from {os.path.basename(file_path)}. Skipping.")
                        # Send an error for files that couldn't be parsed
                        self.send_message(
                            receiver=sender,
                            msg_type="ERROR",
                            payload={"message": f"Could not extract content from {os.path.basename(file_path)} (unsupported format or empty).", "original_trace_id": trace_id},
                            trace_id=trace_id
                        )

                # After processing all documents in the batch:
                if all_processed_chunks:
                    print(f"[{self.name}] Total processed chunks for this batch: {len(all_processed_chunks)}")
                    # Send INGESTION_COMPLETE message to the RetrievalAgent
                    ingestion_complete_payload: IngestionCompletePayload = {
                        "chunks": all_processed_chunks, # Renamed from 'processed_chunks' for clarity with RetrievalAgent's expectation
                        "metadata": {} # Can add batch-level metadata if needed
                    }
                    self.send_message(
                        receiver="RetrievalAgent", # Next in the pipeline
                        msg_type="INGESTION_COMPLETE",
                        payload=ingestion_complete_payload,
                        trace_id=trace_id
                    )
                else:
                    # If no chunks were processed from any document in the batch
                    print(f"[{self.name}] No chunks processed from any uploaded documents. Sending ERROR message.")
                    self.send_message(
                        receiver=sender, # Send error back to the original sender (UI/Coordinator)
                        msg_type="ERROR",
                        payload={"message": "No content was successfully processed from any uploaded document.", "original_trace_id": trace_id},
                        trace_id=trace_id
                    )

            except Exception as e:
                # Catch any unexpected errors during the process
                print(f"[{self.name}] Fatal error during DOCUMENT_UPLOAD processing (Trace ID: {trace_id}): {e}")
                self.send_message(
                    receiver=sender, # Send error back to the original sender
                    msg_type="ERROR",
                    payload={"message": f"Critical ingestion error: {e}", "original_trace_id": trace_id},
                    trace_id=trace_id
                )
        else:
            print(f"[{self.name}] Unhandled message type: {message['type']} (Trace ID: {trace_id})")


if __name__ == '__main__':
    # --- Standalone Test for IngestionAgent ---

    print("--- Testing IngestionAgent standalone ---")

    # Define a dummy message broker function for testing.
    # It will capture messages sent by the IngestionAgent.
    received_messages_by_broker = []
    def dummy_message_broker_ingestion(msg: MCPMessage):
        print(f"[Dummy Broker] Received message: {msg['type']} from {msg['sender']} to {msg['receiver']} (Trace ID: {msg.get('trace_id', 'N/A')})")
        received_messages_by_broker.append(msg)

    # Initialize the IngestionAgent with the dummy broker
    ingestion_agent = IngestionAgent(dummy_message_broker_ingestion)

    # --- Setup dummy test_docs directory for this test ---
    TEST_DIR = "test_ingestion_docs"
    # Ensure a clean slate for the test
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    
    # Create sample text files
    with open(os.path.join(TEST_DIR, "short_doc.txt"), "w") as f:
        f.write("This is a short document for testing.")
    with open(os.path.join(TEST_DIR, "long_doc.txt"), "w") as f:
        f.write("This is a longer document with more content that should be split into multiple chunks. " * 50)
    
    # Create dummy files that are expected to cause parsing errors
    with open(os.path.join(TEST_DIR, "corrupt.docx"), "w") as f:
        f.write("This is not a real DOCX file.") # Intentionally invalid content
    with open(os.path.join(TEST_DIR, "empty.pdf"), "w") as f:
        f.write("") # Empty file
    
    # Define document paths for the simulated upload
    upload_doc_paths: List[Dict[str, str]] = [
        {"path": os.path.join(TEST_DIR, "short_doc.txt"), "type": "txt"},
        {"path": os.path.join(TEST_DIR, "long_doc.txt"), "type": "txt"},
        {"path": os.path.join(TEST_DIR, "non_existent.txt"), "type": "txt"}, # Non-existent file
        {"path": os.path.join(TEST_DIR, "corrupt.docx"), "type": "docx"},   # Corrupt file
        {"path": os.path.join(TEST_DIR, "empty.pdf"), "type": "pdf"},       # Empty parseable file
    ]
    
    # Create a DOCUMENT_UPLOAD message
    upload_message = create_mcp_message(
        sender="UI",
        receiver="IngestionAgent",
        msg_type="DOCUMENT_UPLOAD",
        payload={"document_paths": upload_doc_paths},
        trace_id="upload_test_123"
    )

    # Process the message
    ingestion_agent.process_message(upload_message)
    time.sleep(1) # Give a moment for processing and message sending

    print("\n--- Messages received by Dummy Broker from IngestionAgent ---")
    
    ingestion_complete_msgs = [msg for msg in received_messages_by_broker if msg['type'] == 'INGESTION_COMPLETE']
    error_msgs = [msg for msg in received_messages_by_broker if msg['type'] == 'ERROR']

    # Assertions for INGESTION_COMPLETE message
    if ingestion_complete_msgs:
        ingestion_complete_payload = ingestion_complete_msgs[0]['payload']
        chunks = ingestion_complete_payload['chunks']
        print(f"Found INGESTION_COMPLETE message with {len(chunks)} chunks.")
        assert len(chunks) > 0, "Should have processed chunks from valid documents."
        
        # Verify structure of a sample chunk
        sample_chunk = chunks[0]
        assert "id" in sample_chunk and isinstance(sample_chunk["id"], str)
        assert "content" in sample_chunk and isinstance(sample_chunk["content"], str)
        assert "metadata" in sample_chunk and isinstance(sample_chunk["metadata"], dict)
        assert "source_filename" in sample_chunk["metadata"]
        print("Sample chunk structure verified.")
    else:
        print("No INGESTION_COMPLETE message received.")
        assert False, "INGESTION_COMPLETE message was not sent."

    # Assertions for ERROR messages (for non-existent/corrupt/empty files)
    print(f"\nFound {len(error_msgs)} ERROR messages.")
    assert len(error_msgs) >= 3, "Expected at least 3 ERROR messages for problematic files (non-existent, corrupt docx, empty pdf)."
    
    error_messages_content = [msg['payload']['message'] for msg in error_msgs]
    print("Error messages received:")
    for msg_content in error_messages_content:
        print(f"- {msg_content}")
    
    assert any("File not found" in m for m in error_messages_content)
    assert any("Could not extract content from corrupt.docx" in m for m in error_messages_content)
    assert any("Could not extract content from empty.pdf" in m for m in error_messages_content) # Assuming empty files are unparseable

    # --- Test case for no documents uploaded at all ---
    print("\n--- Simulating DOCUMENT_UPLOAD with no documents provided ---")
    received_messages_by_broker.clear() # Clear previous messages
    no_doc_upload_message = create_mcp_message(
        sender="UI",
        receiver="IngestionAgent",
        msg_type="DOCUMENT_UPLOAD",
        payload={"document_paths": []}, # Empty list of documents
        trace_id="no_doc_test_456"
    )
    ingestion_agent.process_message(no_doc_upload_message)
    time.sleep(0.5)

    assert len(received_messages_by_broker) == 1
    assert received_messages_by_broker[0]['type'] == 'ERROR'
    assert "No document paths provided for upload." in received_messages_by_broker[0]['payload']['message']
    print("Test for no documents uploaded passed.")

    # --- Test case for documents that all fail to parse ---
    print("\n--- Simulating DOCUMENT_UPLOAD where all documents fail to parse ---")
    received_messages_by_broker.clear() # Clear previous messages
    all_fail_doc_paths: List[Dict[str, str]] = [
        {"path": os.path.join(TEST_DIR, "corrupt.docx"), "type": "docx"},
        {"path": os.path.join(TEST_DIR, "empty.pdf"), "type": "pdf"},
    ]
    all_fail_upload_message = create_mcp_message(
        sender="UI",
        receiver="IngestionAgent",
        msg_type="DOCUMENT_UPLOAD",
        payload={"document_paths": all_fail_doc_paths},
        trace_id="all_fail_test_789"
    )
    ingestion_agent.process_message(all_fail_upload_message)
    time.sleep(0.5)

    # Expecting one ERROR message for each failed parse, plus one for 'no chunks processed' overall
    # The current logic will send an error per failed file and then one aggregate error if no chunks result.
    # So for 2 failed files, expect 3 errors total.
    assert len(received_messages_by_broker) == 3 
    final_aggregate_error = [msg for msg in received_messages_by_broker if "No content was successfully processed" in msg['payload']['message']]
    assert len(final_aggregate_error) == 1
    print("Test for all documents failing to parse passed.")

    # --- Cleanup ---
    print(f"\n--- Cleaning up temporary directory: {TEST_DIR} ---")
    shutil.rmtree(TEST_DIR)
    print("IngestionAgent standalone test complete.")