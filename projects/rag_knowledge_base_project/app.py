import os
from dotenv import load_dotenv

import gradio as gr

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_classic.chains.combine_documents import create_stuff_documents_chain 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

embeddings = HuggingFaceEmbeddings(
  model_name = 'sentence-transformers/all-MiniLM-L6-v2',
  model_kwargs = {"device": "cpu"},
  encode_kwargs = {"batch_size": 8}
)
print("embeddings model loaded")

loader = PyPDFDirectoryLoader(r'C:\Users\user\OneDrive\Desktop\langchain_tutorial\projects\rag_knowledge_base_project\knowledge')
docs = loader.load()
print(f"Loaded {len(docs)} pages.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
print("documents splitting has done")

db = FAISS.from_documents(documents, embeddings)
print("embeddings stored in vector store")

llm = ChatGroq(
  model_name="llama-3.3-70b-versatile"
)
print("LLM done")

rag_prompt = ChatPromptTemplate.from_template(
  """
    Answer the questions based only on the provided context. Don't hallucinate. Don't reply to anything else that is not in the context. Make sure you provide the most accurate answer and I will give you praise when you will provide the best accurate answer which the user finds helpful. Don't give any reply to the abusive inputs. Must stay in your limits. If the user asks anything that is not in the context or it is abusive then politely say that you can answer only through the provided context. If the user praises you after he/she finds your response helpful, then say thank you like words or say glad, you liked it and then say let me know how can I help you with the context.
    <context> {context} </context>
    Question: {input}
  """
)
print("prompt done")

def load_file_by_extension(file_path):
  ext = os.path.splitext(file_path)[-1].lower()
  try:
    if ext == ".pdf":
      return PyPDFLoader(file_path).load()
    elif ext == ".docx":
      return Docx2txtLoader(file_path).load()
    elif ext == ".txt":
      return TextLoader(file_path).load()
    elif ext == ".csv":
      return CSVLoader(file_path).load()
    elif ext == ".pptx":
      return UnstructuredPowerPointLoader(file_path).load()
    else:
            print(f"Unsupported file type: {ext}")
            return []
  except Exception as e:
    print(f"Error loading {file_path}: {e}")
    return []

def response(message, history):
  user_text = message["text"]
  user_files = message["files"]

  if user_files:
    new_docs = []
    for file in user_files:
      file_path = file["path"] if isinstance(file, dict) else file
      loaded = load_file_by_extension(file_path)
      new_docs.extend(loaded)
    
    split_new_docs = text_splitter.split_documents(new_docs)
    db.add_documents(split_new_docs)
  
  retriever = db.as_retriever()
  document_chain = create_stuff_documents_chain(llm, rag_prompt)
  retrieval_chain = create_retrieval_chain(retriever, document_chain)

  response = retrieval_chain.invoke({"input": user_text})
  answer = response["answer"]

  return answer

print("response function done")

with gr.Blocks(title='RAG Knowledge Based Chatbot') as demo:
  gr.Markdown('RAG Chatbot')
  gr.Markdown('Ask anything about your uploaded documents and about mr chips questions and answers')

  chatbot = gr.ChatInterface(
    fn=response,
    multimodal=True,
    chatbot=gr.Chatbot(height=420, show_label=False),
    textbox=gr.MultimodalTextbox(
      placeholder='Ask anything about your uploaded files',
      file_types=[".pdf", ".txt", ".docx", ".csv", ".xlsx", "image"],
      container=False,
      scale=8
    ),
    examples=[
      "What kind of house was that Mr. Chips rented from Mrs. Wicket?",
      "What services did Brookfield render during the war?",
      "When and how did Katherine die?"
    ]
  )

demo.launch()