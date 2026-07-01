# main.py - Python 3.14 compatible version
# Uses modern LangChain patterns instead of deprecated agent API

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage

# ─────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

search_tool = TavilySearchResults(max_results=2)

# ─────────────────────────────────────────────────────────
# SIMPLE AGENT FUNCTION
# Instead of LangChain's agent framework, we manually implement
# the same think → search → answer loop
# This works on any Python version
# ─────────────────────────────────────────────────────────
def run_agent(question: str) -> str:

    # Step 1: Ask the LLM if it needs to search
    decision_messages = [
        SystemMessage(content="""You are a helpful assistant. 
When given a question, decide if you need current/real-time information to answer it.

Reply with ONLY one of these two formats:
- If you need to search: SEARCH: <your search query>
- If you can answer directly: ANSWER: <your answer>

Examples:
Q: What is 5 + 5? → ANSWER: 10
Q: Who won the latest IPL? → SEARCH: latest IPL winner 2026
Q: What is the capital of France? → ANSWER: Paris
Q: What is the current price of Bitcoin? → SEARCH: Bitcoin price today"""),
        HumanMessage(content=question)
    ]

    decision = llm.invoke(decision_messages).content.strip()

    # Step 2: If it decided to search, do the search
    if decision.startswith("SEARCH:"):
        query = decision.replace("SEARCH:", "").strip()
        print(f"🔍 Searching for: {query}")

        search_results = search_tool.invoke(query)

        # Format search results into readable text
        results_text = ""
        for r in search_results:
            results_text += f"Source: {r.get('url', '')}\n"
            results_text += f"Content: {r.get('content', '')}\n\n"

        # Step 3: Ask LLM to answer using the search results
        answer_messages = [
            SystemMessage(content="""You are a helpful assistant.
You will be given a question and search results.
Answer the question directly and concisely using only the search results provided.
Always state the winner/result clearly at the start of your answer.
Use only the most recent information when multiple time periods appear.
If the search results describe a match or competition, always explicitly state who WON."""),
            HumanMessage(content=f"""Here is the question you must answer: {question}

Here are the search results to use:
{results_text}

Now answer the question: {question}""")
        ]

        answer = llm.invoke(answer_messages).content.strip()
        return answer

    # Step 4: If it decided to answer directly
    elif decision.startswith("ANSWER:"):
        answer = decision.replace("ANSWER:", "").strip()
        print(f"💡 Answered directly without searching")
        return answer

    # Fallback: just return what the LLM said
    else:
        print(f"⚠️ Unexpected format, returning raw response")
        return decision


# ─────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────
app = FastAPI(title="Research Agent API")


class Question(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "Research Agent API is running!"}


@app.post("/research")
def research(question: Question):
    answer = run_agent(question.text)
    return {
        "question": question.text,
        "answer": answer
    }