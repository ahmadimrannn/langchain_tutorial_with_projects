import os
from dotenv import load_dotenv

import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# Loading the embeddings model
embeddings = HuggingFaceEmbeddings(
  model_name='sentence-transformers/all-MiniLM-L6-v2',
  model_kwargs = {"device": "cpu"},
  encode_kwargs = {"batch_size": 8}
)

# Loading source data, embed, and storing it in the vector store
loader = PyPDFLoader('./mr-chips-info/Novel-Goodbye-Mr.-Chips.pdf')
docs = loader.load()
print(docs[:2])
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 750, chunk_overlap = 150)
final_docs = text_splitter.split_documents(docs)
vectordb = FAISS.from_documents(final_docs, embeddings)

# Loading the LLM and Prompt

llm = ChatGroq(
  model_name = "llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_template(
  """
    Answer the questions based only on the following context. Don't hallucinate. Always give accurate answers. Think before giving the answer and first question your thoughts and think that are you giving the correct answer or you are just hallucinating. Always try to give the most accurate answer. 
    The context is: <context> {context} </context>
    The user's question is: {input}. 
  """
)

# Retriever, Documents Chain, and Retrieval chain
retriever = vectordb.as_retriever()
documents_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, documents_chain)

def chat(user_message, history):
  response = retrieval_chain.invoke({"input": user_message})
  answer = response['answer']

  return answer

with gr.Blocks(title='Mr. Chips Chatbot (Groq + Mr.Chips question answers)') as demo:
  gr.Markdown('🤖💪🏻 Mr. Chips Chatbot...')
  gr.Markdown("Ask anything about **Mr Chips (Chapter 1-18) Question answers** — powered by Groq's `llama-3.3-70b-versatile`")

  chatbot = gr.ChatInterface(
    fn=chat,
    chatbot=gr.Chatbot(height=420, show_label=False),
    textbox=gr.Textbox(
      placeholder='Ask anything about US census data',
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

