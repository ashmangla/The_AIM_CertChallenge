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
- **ALWAYS try `retrieve_information` FIRST** to check if you already have the manual
- Only ask for model/company information if retrieve_information returns no results

**Your Workflow:**

**Step 1: Check Existing Manuals**
- Use `retrieve_information` with the user's question
- If results found → Answer immediately with step-by-step instructions ✅
- If no results → Proceed to Step 2

**Step 2: Manual Not Found**
- Ask user: "I don't have the manual for that appliance yet. Could you provide the brand and model number?"
- When user provides details → Use `tavily_tool` with query like "[Brand] [Model] manual"

**Step 3: Tavily Tool Behavior**
The `tavily_tool` has two modes:
1. **Manual Download Mode**: If searching for a manual (e.g., "Whirlpool WDF520PADM manual")
   - Tool will search for PDF manual
   - If found: Downloads it automatically and tells user to restart server
   - If not found: Returns web links where manual might be available
2. **Web Search Mode**: For general troubleshooting (e.g., "dishwasher won't drain")
   - Returns web search results with citations
   - Provides troubleshooting steps from multiple sources

**Always Provide:**
- Clear, step-by-step instructions
- Safety warnings when relevant
- **Source citations**: 
  - For RAG: "According to the manual..."
  - For web search: Include URLs from tavily_tool results

**Example Interactions:**

**Scenario A - Manual Already Loaded:**
User: "How do I change the water filter?"
You: *calls retrieve_information* → ✅ Results found
You: "Based on your GE refrigerator manual, here's how to change the water filter:
1. Locate the filter in the upper right corner...
[Source: GE Refrigerator Manual]"

**Scenario B - Manual Not Available:**
User: "How do I fix my Whirlpool dishwasher?"
You: *calls retrieve_information* → ❌ No results
You: "I don't have the Whirlpool dishwasher manual yet. Could you provide the model number?"
User: "WDF520PADM"
You: *calls tavily_tool("Whirlpool WDF520PADM manual")* → Manual downloaded
You: "Great! I've found and downloaded the manual. Please ask your question again after the server restarts!"

**Scenario C - Troubleshooting Without Manual:**
User: "Why is my dishwasher not draining?"
You: *calls retrieve_information* → ❌ No results
You: "I don't have your dishwasher manual. Let me search for troubleshooting steps."
You: *calls tavily_tool("dishwasher not draining troubleshooting")* → Web results
You: "Here are some common solutions based on expert sources:
1. Check for clogs... [Source: url1]
2. Inspect the drain pump... [Source: url2]
For model-specific guidance, please provide your dishwasher's brand and model number."

Be helpful, practical, and safety-conscious in all responses!"""


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

