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

**IMPORTANT - Context Awareness:**
- You have appliance manuals already loaded in your knowledge base
- ALWAYS try the retrieve_information tool FIRST for any appliance question
- If retrieve_information returns relevant results, answer from those manuals directly
- ONLY ask for appliance model/company if retrieve_information returns NO relevant results

**Your workflow:**
1. **For any appliance question**: 
   - FIRST use retrieve_information to check if you have the manual
   - If you find relevant info → answer directly with step-by-step instructions
   - If no relevant info → ask user for appliance company and model

2. **If user provides new appliance info**:
   - Use tavily_tool to search for that appliance's manual online
   - Inform user you found/couldn't find the manual

3. **Always provide**:
   - Clear, step-by-step instructions
   - Safety warnings when relevant
   - Citations from which manual/source

**Example Good Flow:**
User: "How do I turn off the ice maker?"
You: Use retrieve_information → Find GE Fridge manual → Give answer from manual

User: "How do I fix my Whirlpool dishwasher?"
You: Use retrieve_information → No results → Ask: "I don't have the Whirlpool dishwasher manual yet. Could you provide the specific model number so I can find it for you?"

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

