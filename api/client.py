import requests
import streamlit as st

def get_llama_response(input_text):
  response = requests.post('http://localhost:8000/llama/essay/invoke',
    json={'input': {'topic': input_text}}
  )

  return response.json()['output']['content']

def get_openai_response(input_text):
  response = requests.post('http://localhost:8000/openai/poem/invoke',
    json={'input': {'topic': input_text}}                         
  )

  return response.json()["output"]["content"]


st.title("Langserve demo with llama and open ai api")

llama_input_text = st.text_input("Write your desired essay topic.")
openai_input_text = st.text_input("Write your desired poem topic.")

if llama_input_text:
  st.write(get_llama_response(llama_input_text))

if openai_input_text:
  st.write(get_openai_response(openai_input_text))