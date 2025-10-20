# src/agent_runner.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import hub
from src.rag_tool import rag_answer

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")


def rag_tool_fn(query: str) -> str:
    """RAG tool function to search movies"""
    out = rag_answer(query, top_k=5)
    s = out["answer"] + "\n\nSources:\n"
    for src in out["sources"]:
        meta = src["meta"]
        s += f"- {meta.get('title','')} ({meta.get('year','')}) — {meta.get('genre','')}\n"
    return s


def create_agent():
    """Create LangChain agent with RAG tool"""
    llm = ChatOpenAI(
        model=LLM_MODEL, 
        api_key=OPENAI_API_KEY, 
        temperature=0.2
    )
    
    # Define tools
    tools = [
        Tool(
            name="RAG_Movie_Search",
            func=rag_tool_fn,
            description="Use this tool to answer movie-related questions from the IMDB Top 1000 dataset. Input should be a question about movies."
        )
    ]
    
    # Get the prompt template
    prompt = hub.pull("hwchase17/react")
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor


if __name__ == "__main__":
    print("🎬 IMDB Movie Agent")
    print("Type 'exit' or 'quit' to stop\n")
    
    agent = create_agent()
    
    while True:
        q = input("Q: ")
        if q.strip().lower() in ("exit", "quit"):
            print("Goodbye! 👋")
            break
        
        try:
            result = agent.invoke({"input": q})
            print(f"\nA: {result['output']}\n")
        except Exception as e:
            print(f"Error: {e}\n")