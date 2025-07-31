# agents/llm_response_agent.py

import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import Callable, List, Dict, Any, Optional
import time

# Import the base agent class and MCP message definitions
from agents.base_agent import Agent
from agents.mcp_messages import MCPMessage, create_mcp_message, RetrievalResultPayload, FinalResponsePayload, ErrorPayload

# Load environment variables from .env file (e.g., GOOGLE_API_KEY)
load_dotenv()

# --- Gemini API Configuration ---
# Configure Gemini API if not already configured globally or by other modules.
# It's good practice to ensure it's configured here for this agent's operation.
try:
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=gemini_api_key)
    print("[LLMResponseAgent] Gemini API configured successfully.")
except Exception as e:
    print(f"[LLMResponseAgent] Error configuring Gemini API: {e}")
    print("[LLMResponseAgent] This agent may not be able to call the LLM.")


class LLMResponseAgent(Agent):
    """
    Agent responsible for generating the final LLM response using retrieved context
    and incorporating multi-turn conversation history.
    It receives RETRIEVAL_RESULT messages and sends FINAL_RESPONSE messages.
    """
    def __init__(self, message_broker: Callable[[MCPMessage], None], model_name: str = "gemini-1.5-flash"): # Changed to 1.5-flash for potential faster response. Adjust if needed.
        super().__init__("LLMResponseAgent", message_broker)
        # Initialize the GenerativeModel. Ensure the model name is compatible with your access.
        # 'gemma-3-27b-it' is a good choice, 'gemini-1.5-flash' is generally faster for chat.
        self.model = genai.GenerativeModel(model_name)
        print(f"[{self.name}] Initialized with model: {model_name}.")

    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """
        Formats the conversation history into a string suitable for the LLM prompt.
        """
        formatted_history = []
        for entry in history:
            role = "User" if entry["role"] == "user" else "Assistant"
            formatted_history.append(f"{role}: {entry['content']}")
        return "\n".join(formatted_history)

    def _generate_prompt(self, query: str, context_chunks: List[str], 
                         conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Constructs a robust prompt for the LLM, incorporating the query,
        retrieved context, and conversation history for multi-turn support.

        Args:
            query (str): The current user's question.
            context_chunks (List[str]): List of relevant text snippets retrieved from the vector store.
            conversation_history (Optional[List[Dict[str, str]]]): List of prior messages in the conversation.

        Returns:
            str: The fully constructed prompt string for the LLM.
        """
        # Base instructions for the LLM
        prompt_parts = [
            "You are a helpful and accurate AI assistant. Your goal is to provide concise and direct answers.",
            "Carefully analyze the following information to formulate your response."
        ]

        # Add conversation history if available for multi-turn context
        if conversation_history:
            formatted_history = self._format_conversation_history(conversation_history)
            prompt_parts.append(
                f"\n--- Conversation History ---\n"
                f"Below is the recent conversation between you and the user. Use it to understand the context of the current question, especially for follow-up questions. Do not repeat previous answers verbatim, but build upon them.\n"
                f"{formatted_history}"
            )

        # Add context chunks for RAG
        if context_chunks:
            context_str = "\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
            prompt_parts.append(
                f"\n--- Retrieved Information ---\n"
                f"These are the most relevant sections from the documents you have access to. Prioritize this information heavily.\n"
                f"{context_str}\n"
            )
        else:
            # Fallback if no relevant context is found
            prompt_parts.append(
                f"\n--- Retrieved Information ---\n"
                f"No highly relevant information was found in the documents for the current query."
            )

        # Add the current user query and instructions on how to respond
        prompt_parts.append(
            f"\n--- Current Query ---\n"
            f"User: {query}\n\n"
            f"--- Instructions ---\n"
            f"1. **Answer directly and concisely.** Aim for clarity and avoid unnecessary conversational filler unless it helps to clarify that the information is limited.\n"
            f"2. **Strictly use the 'Retrieved Information' and 'Conversation History' for your answer.** If the answer cannot be found or inferred from these sources, state clearly that you do not have enough information to answer the question, or that the information is not in the provided context.\n"
            f"3. **Do NOT mention the context explicitly** (e.g., 'Based on the provided documents...'). Just provide the answer.\n"
            f"4. **Maintain conversational flow** for follow-up questions, building on previous answers if relevant.\n"
            f"5. If the 'Retrieved Information' is empty or irrelevant, but the question can be answered from 'Conversation History' (e.g., a simple acknowledgement or follow-up on your own previous statement), do so.\n"
            f"6. If the question is outside the scope of the provided documents/history, state that the information is not available.\n"
            f"Answer:"
        )

        return "\n".join(prompt_parts)

    def process_message(self, message: MCPMessage):
        """
        Processes incoming MCP messages.
        Expects messages of type "RETRIEVAL_RESULT" to generate a response.
        """
        # Ensure the message is intended for this agent
        if message["receiver"] != self.name:
            return

        print(f"[{self.name}] Received message of type: {message['type']} (Trace ID: {message['trace_id']})")

        if message["type"] == "RETRIEVAL_RESULT":
            trace_id = message["trace_id"]
            try:
                payload: RetrievalResultPayload = message["payload"]
                query = payload.get("query")
                # Retrieve the full chunk content from the payload
                retrieved_chunks_full = payload.get("retrieved_chunks", [])
                
                # Extract only the 'content' for the LLM prompt, but keep full for source display
                retrieved_context_contents = [chunk["content"] for chunk in retrieved_chunks_full if "content" in chunk]

                # Extract conversation history, which is now part of QueryRequestPayload
                # It's passed from CoordinatorAgent to RetrievalAgent, and then forwarded here.
                conversation_history = payload.get("conversation_history", [])


                if not query:
                    print(f"[{self.name}] No query in RETRIEVAL_RESULT message. Cannot generate response. Trace ID: {trace_id}")
                    # Send an error message back to the sender (CoordinatorAgent/UI)
                    self.send_message(
                        receiver="CoordinatorAgent", # Send back to coordinator to pass to UI
                        msg_type="ERROR",
                        payload={"message": "Missing query in retrieval result payload.", "original_trace_id": trace_id},
                        trace_id=trace_id
                    )
                    return

                print(f"[{self.name}] Generating prompt for LLM, incorporating {len(retrieved_context_contents)} chunks and {len(conversation_history)} history entries.")
                
                # Generate the prompt using the query, context, and conversation history
                llm_prompt = self._generate_prompt(query, retrieved_context_contents, conversation_history)
                
                # print(f"[{self.name}] Full LLM Prompt:\n{llm_prompt}\n---End Prompt---") # For debugging, uncomment if needed
                print(f"[{self.name}] Calling LLM with prompt (first 200 chars):\n'{llm_prompt[:200]}...'")
                
                # Make the LLM call
                # Consider adding a timeout or retry logic for production
                response = self.model.generate_content(llm_prompt)
                
                # Safely extract the text response from the LLM
                final_answer = ""
                if hasattr(response, 'text'):
                    final_answer = response.text
                elif response.parts: # For newer versions or different response types
                    final_answer = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                
                if not final_answer.strip(): # Check for empty or whitespace-only response
                    final_answer = "I could not generate a coherent answer based on the provided information."
                    print(f"[{self.name}] LLM response was empty or unparseable.")

                print(f"[{self.name}] LLM generated response (first 100 chars): '{final_answer[:100]}...'")

                # Send FINAL_RESPONSE message back to the UI via the Coordinator
                final_response_payload: FinalResponsePayload = {
                    "answer": final_answer,
                    # Pass back only the 'content' of source chunks for display in UI
                    "source_chunks": retrieved_context_contents
                }
                self.send_message(
                    receiver="UI", # Destination is the UI via Coordinator
                    msg_type="FINAL_RESPONSE",
                    payload=final_response_payload,
                    trace_id=trace_id
                )

            except Exception as e:
                print(f"[{self.name}] Error processing RETRIEVAL_RESULT or calling LLM: {e}")
                # Send an error message back to the sender (CoordinatorAgent/UI)
                self.send_message(
                    receiver="CoordinatorAgent", # Send back to coordinator to pass to UI
                    msg_type="ERROR",
                    payload={"message": f"LLM response generation error: {e}", "original_trace_id": trace_id},
                    trace_id=trace_id
                )
        else:
            print(f"[{self.name}] Unhandled message type: {message['type']} (Trace ID: {message['trace_id']})")


if __name__ == '__main__':
    # --- Standalone Test for LLMResponseAgent with Multi-Turn and Robust Prompt ---

    print("--- Testing LLMResponseAgent standalone with multi-turn capability ---")

    # Dummy Message Broker for testing purposes. It captures messages sent by the agent.
    received_messages_by_broker = []
    def dummy_message_broker_llm_response(msg: MCPMessage):
        print(f"[Dummy Broker] Received message: {msg['type']} from {msg['sender']} to {msg['receiver']} (Trace ID: {msg['trace_id']})")
        received_messages_by_broker.append(msg)

    # Initialize the LLMResponseAgent
    llm_agent = LLMResponseAgent(dummy_message_broker_llm_response)

    # --- Simulate a RETRIEVAL_RESULT message (Turn 1: Initial Query) ---
    print("\n--- Simulating RETRIEVAL_RESULT message for Turn 1 (Initial Query) ---")
    
    sample_query_1 = "What was the revenue increase and profit margin in the latest report?"
    sample_retrieved_chunks_1 = [
        {"id": "rep_chunk_1", "content": "Annual financial report summary: Revenue up by 15%, profit margin at 10%. Key initiatives for next quarter include expanding into new markets and optimizing existing sales channels."},
        {"id": "rep_chunk_2", "content": "The Q4 earnings call discussed strong growth in SaaS subscriptions leading to the revenue increase. Operating expenses were well-managed."}
    ]
    
    retrieval_result_msg_1 = create_mcp_message(
        sender="RetrievalAgent",
        receiver="LLMResponseAgent",
        msg_type="RETRIEVAL_RESULT",
        payload={
            "query": sample_query_1,
            "retrieved_chunks": sample_retrieved_chunks_1, # Note: payload now expects full chunks from RetrievalAgent
            "conversation_history": [] # No history for the first turn
        }
    )
    llm_agent.process_message(retrieval_result_msg_1)
    time.sleep(2) # Give LLM time to respond

    # Verify response for Turn 1
    if received_messages_by_broker:
        final_response_1 = next((msg for msg in received_messages_by_broker if msg['type'] == 'FINAL_RESPONSE' and msg['trace_id'] == retrieval_result_msg_1['trace_id']), None)
        if final_response_1:
            print(f"\nAssistant (Turn 1): {final_response_1['payload']['answer']}")
            # Update history for next turn
            current_history = [
                {"role": "user", "content": sample_query_1},
                {"role": "assistant", "content": final_response_1['payload']['answer']}
            ]
        else:
            print("\nError: No FINAL_RESPONSE received for Turn 1.")
            current_history = [{"role": "user", "content": sample_query_1}, {"role": "assistant", "content": "Error in response."}]
    else:
        print("\nNo messages received by broker yet.")
        current_history = [{"role": "user", "content": sample_query_1}]

    # Clear broker's received messages for the next turn
    received_messages_by_broker.clear()


    # --- Simulate a RETRIEVAL_RESULT message (Turn 2: Follow-up) ---
    print("\n--- Simulating RETRIEVAL_RESULT message for Turn 2 (Follow-up Query) ---")

    sample_query_2 = "What about the initiatives for next quarter?"
    # Assume RetrievalAgent still finds relevant chunks, potentially including the first one
    sample_retrieved_chunks_2 = [
        {"id": "rep_chunk_1", "content": "Annual financial report summary: Revenue up by 15%, profit margin at 10%. Key initiatives for next quarter include expanding into new markets and optimizing existing sales channels."},
        {"id": "roadmap_chunk_1", "content": "Project roadmap: Phase 1 (Discovery) completed. Phase 2 (Development) involves implementing agentic architecture and Model Context Protocol for inter-agent communication."}
    ]
    
    retrieval_result_msg_2 = create_mcp_message(
        sender="RetrievalAgent",
        receiver="LLMResponseAgent",
        msg_type="RETRIEVAL_RESULT",
        payload={
            "query": sample_query_2,
            "retrieved_chunks": sample_retrieved_chunks_2,
            "conversation_history": current_history # Pass the history
        }
    )
    llm_agent.process_message(retrieval_result_msg_2)
    time.sleep(2) # Give LLM time to respond

    # Verify response for Turn 2
    if received_messages_by_broker:
        final_response_2 = next((msg for msg in received_messages_by_broker if msg['type'] == 'FINAL_RESPONSE' and msg['trace_id'] == retrieval_result_msg_2['trace_id']), None)
        if final_response_2:
            print(f"\nAssistant (Turn 2): {final_response_2['payload']['answer']}")
            # Further update history if needed for more turns
        else:
            print("\nError: No FINAL_RESPONSE received for Turn 2.")
    else:
        print("\nNo messages received by broker for Turn 2.")

    print("\nStandalone LLMResponseAgent test complete with multi-turn simulation.")