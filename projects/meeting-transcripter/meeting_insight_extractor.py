import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders.text import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from utils.text_cleaner import clean_transcript

load_dotenv()

loader = TextLoader(
  file_path='./transcript/transcript.txt'
)
loaded_text = loader.load()
raw_transcript = loaded_text[0].page_content
transcript = clean_transcript(raw_transcript)

llm = ChatGroq(
  model_name='llama-3.3-70b-versatile'
)

prompt = """You are an expert meeting transcription insight extractor. 
Like if someone gives you any meeting transcript, you extract summary, action items, deadlines, decisions, project owners, and all that important stuff from the meeting in a clean format which can then be used by teams inside the organization. 
Meeting Transcript: {transcript}"""

prompt_template = PromptTemplate(
  input_variables=['transcript'],
  template=prompt
)

chain = prompt_template | llm
response = chain.invoke({'transcript': transcript})
result = response.content
print(result)