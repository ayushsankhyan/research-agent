# test_groq.py
# Quick test to confirm Groq is working

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

response = llm.invoke("Say hello and confirm you're working!")
print(response.content)