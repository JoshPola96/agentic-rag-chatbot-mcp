# agents/base_agent.py

from abc import ABC, abstractmethod
from typing import Callable
from agents.mcp_messages import MCPMessage

class Agent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    
    Provides common functionality for agent identification and message passing
    through a central message broker (typically the CoordinatorAgent).
    Concrete agents must implement the 'process_message' method to define
    their specific behavior upon receiving messages.
    """

    def __init__(self, name: str, message_broker: Callable[[MCPMessage], None]):
        """
        Initializes a new Agent instance.

        Args:
            name (str): The unique name of the agent.
            message_broker (Callable[[MCPMessage], None]): A callable (function or method)
                                                            that acts as the central message
                                                            broker, through which this agent
                                                            can send messages to other agents.
                                                            This is typically the send_message
                                                            method of the CoordinatorAgent.
        """
        self.name = name
        self.message_broker = message_broker # This will be the Coordinator's send_message method

    def send_message(self, receiver: str, msg_type: str, payload: dict, trace_id: str = None):
        """
        Sends a message to another agent via the central message broker.

        Args:
            receiver (str): The name of the target agent or system component (e.g., "UI", "RetrievalAgent").
            msg_type (str): The type of message being sent (e.g., "QUERY_REQUEST", "FINAL_RESPONSE").
            payload (dict): The data payload of the message.
            trace_id (str, optional): An identifier to trace the message flow through the system.
                                      If None, the message broker (Coordinator) should generate one.
        """
        msg: MCPMessage = {
            "sender": self.name,
            "receiver": receiver,
            "type": msg_type,
            "trace_id": trace_id,
            "payload": payload
        }
        print(f"[{self.name}] Sending {msg_type} to {receiver} (Trace ID: {trace_id or 'N/A'})")
        self.message_broker(msg)

    @abstractmethod
    def process_message(self, message: MCPMessage):
        """
        Abstract method to be implemented by concrete agents to process incoming messages.
        
        Args:
            message (MCPMessage): The incoming message dictionary to be processed.
        """
        pass