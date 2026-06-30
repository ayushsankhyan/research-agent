# main.py
# A research agent: FastAPI + LangChain + Groq + Tavily

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

# ─────────────────────────────────────────────────────────
# SET UP THE AGENT'S BRAIN AND TOOLS
# ─────────────────────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

# Write the ReAct prompt ourselves — no hub dependency needed
# This is the exact structure the agent uses to reason:
# Thought -> Action -> Observation -> repeat -> Final Answer
react_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT RULES:
- If the question can be answered directly using reasoning, math, or 
  knowledge you already have (such as arithmetic, definitions, or 
  logic), DO NOT use a tool. Instead, go straight to:
  Thought: I can answer this directly without searching.
  Final Answer: [your answer]
- Only use a tool when you genuinely need current, real-world, or 
  external information that you cannot reliably know on your own.
- NEVER write "Action: None". If you don't need a tool, skip directly 
  to "Final Answer:".
- When search results contain information from MULTIPLE different 
  years or dates, always identify and use ONLY the most recent, 
  current result. Explicitly look for dates in the search results 
  and compare them before answering. Ignore older results unless 
  the question specifically asks about history.

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

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
    result = agent_executor.invoke({"input": question.text})
    return {
        "question": question.text,
        "answer": result["output"]
    }