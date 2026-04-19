from fastapi import FastAPI
from langserve import add_routes
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import uvicorn 
import os
from dotenv import load_dotenv

load_dotenv()

# Get environment variable using os
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

app = FastAPI()

# LLMs
llama_model = ChatGroq(model_name="llama-3.3-70b-versatile")
openai_model = ChatGroq(model_name="openai/gpt-oss-120b")

# Prompt Templates for LLMs
llama_prompt = ChatPromptTemplate.from_template('Write me a best organized essay of about 150 words on this topic: {topic}')
openai_prompt = ChatPromptTemplate.from_template('Write me a best organized and a little bit funny poem on this topic: {topic}')

# Different routes for different LLMs (Every LLM is used for specific task in which they are good at)
add_routes(
  app,
  llama_prompt|llama_model,
  path='/llama/essay'
)

add_routes(
  app,
  openai_prompt|openai_model,
  path='/openai/poem'
)

if __name__ == "__main__":
  uvicorn.run(app, host="localhost", port=8000)

