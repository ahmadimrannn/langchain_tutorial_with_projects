import os
from dotenv import load_dotenv

import gradio as gr

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ── 1. Load everything ONCE at startup (not per request) ────────────────────
print("Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 8}  # process in small batches
)

print("Loading documents...")
loader = WebBaseLoader('https://docs.smith.langchain.com/')
docs = loader.load()
print("Chunking documents...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
print('Getting final docs ready...')
final_docs = splitter.split_documents(docs)
print("Building vector db")
vector_db = FAISS.from_documents(final_docs, embeddings)

print("Loading LLM and building chain...")
llm = ChatGroq(
    groq_api_key=os.environ['GROQ_API_KEY'],
    model_name='llama-3.3-70b-versatile'
)

rag_prompt = ChatPromptTemplate.from_template("""
    Answer the questions based only on the provided context.
    Don't hallucinate. Give the most accurate response possible.
    <context> {context} </context>
    Question: {input}
""")

doc_chain = create_stuff_documents_chain(llm, rag_prompt)
retriever = vector_db.as_retriever()
chain = create_retrieval_chain(retriever, doc_chain)

print("✅ All resources loaded. Starting Gradio...")

# ── 2. Chat function (runs per message, chain already built) ─────────────────
def chat(user_message, history):
    response = chain.invoke({"input": user_message})
    answer = response['answer']

    return answer

# ── 3. Gradio UI ─────────────────────────────────────────────────────────────
with gr.Blocks(title="RAG Chatbot (Groq + LangSmith Docs)") as demo:
    gr.Markdown("# 🤖 RAG Chatbot Demo")
    gr.Markdown("Ask anything about **LangSmith** — powered by Groq's `llama-3.3-70b-versatile`")

    chatbot = gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(height=400, show_label=False),
        textbox=gr.Textbox(
            placeholder="Ask a question about LangSmith...",
            container=False,
            scale=7
        ),
        examples=[
            "What is LangSmith?",
            "How do I set up tracing in LangSmith?",
            "What are the main features of LangSmith?",
        ],
    )

demo.launch()