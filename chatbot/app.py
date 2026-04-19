from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')

# Prompt Template
prompt_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are an ai assisstant which is master of all field. Respond smartly to all the user queries and generate answers effeciently."),
    ("user", "{question}")
  ]
)

# Streamlit 
st.title('Langchain first project (Chatbot) using Llama Model')
input_text = st.text_input("Search any topic of your choice.")

# Llama LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile")
output_parser = StrOutputParser()

# Chaining everything together
chain = prompt_template|llm|output_parser

if input_text:
  st.write(chain.invoke({'question': input_text}))