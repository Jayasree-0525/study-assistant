"""LangGraph study assistant agent with three retrieval tools."""

import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from retrieval.tools import ALL_TOOLS
from typing_extensions import TypedDict

load_dotenv()

# overall goal of this file is to

SYSTEM_PROMPT = """You are a helpful study assistant for a university course
on Markets and Competitive Strategy (JRE410).

You help students understand their lecture notes on topics including:
- Porter's Five Forces model (buyer power, supplier power, rivalry,
  threat of entry, threat of substitutes)
- External industry analysis and the 7-question framework
- Strategic group maps and competitive positioning
- Key success factors (KSFs) of an industry
- Industry driving forces and how they reshape competition
- Return on equity and return on assets across industries
- Real-world examples: wine industry, engineering consulting,
  specialty clothing retailers (Zara, H&M, Gap), SNC Lavalin

You have access to three tools:
1. semantic_search — searches the student's uploaded lecture notes
2. query_tables — queries structured/tabular data from the notes
   (e.g. ROA by industry, entry barrier lists, KSF tables)
3. web_search — searches the web for additional real-world context
   beyond what the lecture covers

Strategy:
- ALWAYS call semantic_search first for any question
- If the question involves numbers, industry comparisons, or data
  tables (e.g. ROA figures), also call query_tables
- If semantic_search returns low scores (below 0.3) or the student
  asks for current real-world examples not in the notes, call web_search
- Always cite your sources: mention the file name and page number
- Be concise and focused on what the student is asking
- Connect answers back to the Five Forces framework where relevant

Never make up information. If you don't know, say so."""


class AgentState(TypedDict):
    """State that flows through the agent graph."""

    messages: Annotated[list, add_messages]


def _should_continue(state: AgentState) -> str:
    """Decide whether to call a tool or end the conversation."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def _call_model(state: AgentState) -> AgentState:
    """Call the LLM with the current conversation state."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


'''def build_agent(use_memory: bool = True):
    """Build and compile the LangGraph agent with optional memory."""
    from langgraph.checkpoint.memory import MemorySaver

    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)

    graph.add_node("agent", _call_model)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    if use_memory:
        memory = MemorySaver()
        return graph.compile(checkpointer=memory)

    return graph.compile()'''


def build_agent(use_memory: bool = True):
    """Build and compile the LangGraph agent with optional memory."""
    from langgraph.checkpoint.memory import MemorySaver

    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)

    graph.add_node("agent", _call_model)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    if use_memory:
        memory = MemorySaver()
        return graph.compile(checkpointer=memory)

    return graph.compile()


'''def ask(
    question: str,
    agent=None,
    thread_id: str = "default",
) -> str:
    """
    Ask the agent a question and get a response.
    Uses thread_id to maintain conversation memory.

    Args:
        question: The student's question.
        agent: Optional pre-built agent.
        thread_id: Session ID for memory continuity.

    Returns:
        The agent's response as a string.
    """
    if agent is None:
        agent = build_agent()

    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )

    last_message = result["messages"][-1]
    if isinstance(last_message, AIMessage):
        return last_message.content

    return str(last_message.content)'''


def ask(
    question: str,
    agent=None,
    thread_id: str = "default",
) -> str:
    """
    Ask the agent a question and get a response.
    Uses thread_id to maintain conversation memory.

    Args:
        question: The student's question.
        agent: Optional pre-built agent.
        thread_id: Session ID for memory continuity.

    Returns:
        The agent's response as a string.
    """
    if agent is None:
        agent = build_agent()

    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )

    last_message = result["messages"][-1]
    if isinstance(last_message, AIMessage):
        return last_message.content

    return str(last_message.content)


if __name__ == "__main__":
    print("Building agent...")
    agent = build_agent()
    print("Agent ready. Running test questions...\n")

    questions = [
        "What are the Five Forces in Porter's model?",
        "When is the bargaining power of buyers stronger?",
        "What are the most powerful barriers to entry? Give examples from the notes.",
    ]

    for q in questions:
        print(f"Q: {q}")
        answer = ask(q, agent)
        print(f"A: {answer}\n")
        print("-" * 60 + "\n")
