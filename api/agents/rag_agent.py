"""RAG Agent with tool belt for appliance manual queries.

This agent uses retrieve_information and tavily_tool to answer
questions about appliance manuals and provide web search capabilities.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph

# Import tools from the tools module
sys.path.append(str(Path(__file__).parent.parent))
from tools.tools import retrieve_information, tavily_tool


logger = logging.getLogger(__name__)


# Handyman assistant system prompt
HANDYMAN_SYSTEM_PROMPT = """You are a handyman assistant who helps users with appliance-related questions.

Your workflow:
1. Use the tavily_tool to search the web for the manual for the appliance
1. For appliance-specific questions, use the retrieve_information tool to search through the appliance manuals
2. If additional information is needed, use the llm for web search
3. Provide clear, step-by-step instructions when explaining how to use or fix appliances
4. Always cite which manual or source your information comes from

Be helpful, practical, and safety-conscious in your responses."""


def create_rag_agent(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0
) -> Any:
    """Create a RAG agent with retrieve_information and tavily_tool using LangGraph.
    
    This agent uses LangGraph's ReAct pattern to decide when to use tools and when
    to respond directly. It's designed to be a helpful handyman assistant for
    appliance-related questions.
    
    The agent uses a compiled graph that can be extended with:
    - Memory/persistence
    - Human-in-the-loop
    - Custom routing logic
    - Multiple agents
    
    Args:
        model_name: OpenAI model name to use for the agent (default: gpt-4o-mini)
        temperature: Temperature for response generation (0.0 = deterministic)
        
    Returns:
        Compiled LangGraph agent ready to invoke
        
    Example:
        >>> agent = create_rag_agent()
        >>> result = agent.invoke({
        ...     "messages": [HumanMessage(content="How do I use the ice maker?")]
        ... })
        >>> # Access the final response
        >>> print(result["messages"][-1].content)
    """
    logger.info(f"Creating RAG agent with model: {model_name}")
    
    # Initialize the LLM
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    
    # Define the tool belt - both tools available
    tools: List[BaseTool] = [retrieve_information, tavily_tool]
    
    # Create the ReAct agent using LangGraph's prebuilt function
    # The prompt is passed via a system message in the state_modifier parameter
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_schema=None,  # Use default state
        # Note: System prompt will be added when invoking the agent
    )
    
    logger.info("RAG agent created successfully with LangGraph")
    return agent


__all__ = [
    "create_rag_agent",
    "HANDYMAN_SYSTEM_PROMPT"
]

