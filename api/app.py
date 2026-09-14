from fastapi import FastAPI
from langserve import add_routes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import uvicorn 
import os
from dotenv import load_dotenv

load_dotenv()

# Get environment variable using os
os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')

app = FastAPI()

# LLMs
llama_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
openai_model = ChatGoogleGenerativeAI(model_name="gemini-3.5-flash-lite")

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

