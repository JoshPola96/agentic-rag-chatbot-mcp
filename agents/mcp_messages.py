# agents/mcp_messages.py

from typing import TypedDict, List, Dict, Any, Literal
import uuid
from typing import Optional

# Define types for common payloads
class DocumentPayload(TypedDict):
    path: str
    type: str

class DocumentUploadPayload(TypedDict):
    document_paths: List[DocumentPayload]

class ErrorPayload(TypedDict):
    message: str
    details: Optional[str] # Optional field for more specific error info
    
class IngestionCompletePayload(TypedDict):
    processed_chunks: List[str]
    # Add any other metadata you want to carry forward, e.g., document_ids, source_filenames
    metadata: Dict[str, Any]

class QueryRequestPayload(TypedDict):
    query: str

class RetrievalResultPayload(TypedDict):
    retrieved_context: List[str]
    query: str

class FinalResponsePayload(TypedDict):
    answer: str
    source_chunks: List[str]
    # Add other info like trace_id or original_query if needed for UI display

# Define the core MCP Message structure
class MCPMessage(TypedDict):
    sender: str
    receiver: str
    type: Literal[
        "DOCUMENT_UPLOAD",
        "INGESTION_COMPLETE",
        "QUERY_REQUEST",
        "RETRIEVAL_RESULT",
        "FINAL_RESPONSE",
        "ERROR", # Good to have for error handling
        # Add more message types as your system evolves
    ]
    trace_id: str # Unique ID for tracing a request through agents
    payload: Dict[str, Any] # Use a generic Dict for payload for flexibility initially

def create_mcp_message(
    sender: str,
    receiver: str,
    msg_type: Literal[
        "DOCUMENT_UPLOAD",
        "INGESTION_COMPLETE",
        "QUERY_REQUEST",
        "RETRIEVAL_RESULT",
        "FINAL_RESPONSE",
        "ERROR"
    ],
    payload: Dict[str, Any],
    trace_id: str = None
) -> MCPMessage:
    """Helper function to create an MCP message."""
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    return {
        "sender": sender,
        "receiver": receiver,
        "type": msg_type,
        "trace_id": trace_id,
        "payload": payload
    }

if __name__ == "__main__":
    print("--- Testing MCP Message Definitions ---")

    # Example: Document Upload Message
    doc_upload_payload: DocumentUploadPayload = {
        "document_paths": [
            {"path": "/docs/report.pdf", "type": "pdf"},
            {"path": "/docs/notes.txt", "type": "txt"}
        ]
    }
    upload_msg = create_mcp_message(
        sender="UI",
        receiver="IngestionAgent",
        msg_type="DOCUMENT_UPLOAD",
        payload=doc_upload_payload
    )
    print(f"\nDocument Upload Message: {upload_msg}")

    # Example: Ingestion Complete Message
    ingestion_payload: IngestionCompletePayload = {
        "processed_chunks": ["chunk 1 content", "chunk 2 content"],
        "metadata": {"doc_id": "doc-123", "filename": "report.pdf"}
    }
    ingestion_msg = create_mcp_message(
        sender="IngestionAgent",
        receiver="RetrievalAgent",
        msg_type="INGESTION_COMPLETE",
        payload=ingestion_payload,
        trace_id=upload_msg['trace_id'] # Use same trace_id
    )
    print(f"\nIngestion Complete Message: {ingestion_msg}")

    # Example: Query Request Message
    query_payload: QueryRequestPayload = {"query": "What are the KPIs?"}
    query_msg = create_mcp_message(
        sender="UI",
        receiver="RetrievalAgent",
        msg_type="QUERY_REQUEST",
        payload=query_payload
    )
    print(f"\nQuery Request Message: {query_msg}")

    # Example: Retrieval Result Message
    retrieval_payload: RetrievalResultPayload = {
        "retrieved_context": ["context chunk 1", "context chunk 2"],
        "query": "What are the KPIs?"
    }
    retrieval_msg = create_mcp_message(
        sender="RetrievalAgent",
        receiver="LLMResponseAgent",
        msg_type="RETRIEVAL_RESULT",
        payload=retrieval_payload,
        trace_id=query_msg['trace_id']
    )
    print(f"\nRetrieval Result Message: {retrieval_msg}")