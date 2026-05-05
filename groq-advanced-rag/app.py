import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import time

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

if "vector" not in st.session_state:
  st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  st.session_state.loader = WebBaseLoader('https://docs.smith.langchain.com/')
  st.session_state.docs = st.session_state.loader.load()
  st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
  st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:5])
  st.session_state.vector_db = Chroma.from_documents(st.session_state.final_documents, st.session_state.embeddings)


st.title('Advanced RAG Chatbot Demo (Using Groq)')
llm = ChatGroq(model_name='llama-3.3-70b-versatile')

rag_prompt = ChatPromptTemplate.from_template(
  """
    Answer the questions based only on the provided context. Don't hallucinate. Make sure you give the correct and the most accurate response. Understand the input query or questions deeply and thoroughly and then use your full power and then generate the most accurate response by using your full power.
    The context is <context> {context} </context>
    The question is: {input}
  """
)

document_chain = create_stuff_documents_chain(llm, rag_prompt)
retriever = st.session_state.vector_db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input('Enter your questions here...')

if prompt: 
  start = time.process_time()
  response = retrieval_chain.invoke({"input": prompt})
  print("Response time:", time.process_time() - start)
  st.write(response['answer'])

  with st.expander("Document Similarity Search"):
    for i, doc in enumerate(response['context']):
      st.write(doc.page_content)
      st.write("------------")


