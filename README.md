# 🦜🔗 LangChain Tutorial with Projects
 
A beginner-friendly repository featuring hands-on LangChain tutorials and real-world projects to help developers build LLM-powered applications using chains, agents, memory, and tools.
 
Built while following the **Agentic AI Roadmap** — Step 4: Gen AI Fundamentals.
 
---
 
## 🚀 Projects
 
### 1. 🤖 Simple Chatbot
A conversational AI chatbot built with LangChain, Llama LLM, GroqCloud, and Streamlit.
 
**Tech Stack:**
- [LangChain](https://python.langchain.com/) — LLM application framework
- [Llama (via GroqCloud)](https://console.groq.com/) — LLM for fast inference
- [Streamlit](https://streamlit.io/) — Web UI framework
**Features:**
- Natural conversational responses powered by Llama LLM
- Clean, interactive UI built with Streamlit
- Fast inference via GroqCloud API
---
 
### 2. 🔌 Multi-LLM API Router
A unified API layer that routes requests from any Web App or Mobile App to multiple LLM providers — OpenAI, Gemini, Groq, and Claude — through a single interface.
 
**Architecture:**
 
```
Web App / Mobile App  →  API  →  Routes  →  OpenAI
                                         →  Gemini
                                         →  Groq
                                         →  Claude
```
 
**Tech Stack:**
- [LangChain](https://python.langchain.com/) — unified LLM interface
- [FastAPI](https://fastapi.tiangolo.com/) — API framework
- [OpenAI](https://platform.openai.com/) — GPT models
- [Google Gemini](https://ai.google.dev/) — Gemini models (not added in the project but the approach will be same)
- [GroqCloud](https://console.groq.com/) — Groq/Llama models
- [Anthropic Claude](https://www.anthropic.com/) — Claude models (not added in the project but the approach will be same)
**Features:**
- Single API endpoint to interact with multiple LLM providers
- Easily switch between models without changing your app code
- Extensible architecture — add new LLM providers as needed
- Designed as a reusable base for integrating AI into any web or mobile app
---
 
## 🛠️ Setup & Installation
 
### Prerequisites
- Python 3.8+
- API keys for the LLM providers you want to use:
  - [GroqCloud](https://console.groq.com/) *(free)*
  - [OpenAI](https://platform.openai.com/)
  - [Google Gemini](https://ai.google.dev/)
  - [Anthropic Claude](https://console.anthropic.com/)
### 1. Clone the Repository
```bash
git clone https://github.com/ahmadimrannn/langchain_tutorial_with_projects.git
cd langchain_tutorial_with_projects
```
 
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
 
### 3. Set Up Environment Variables
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```
 
### 4. Run the Chatbot
```bash
streamlit run app.py
```
 
### 5. Run the API
```bash
uvicorn main:app --reload
```
 
---
 
## 📁 Project Structure
 
```
langchain_tutorial_with_projects/
│
├── chatbot/
│   ├── app.py               # Streamlit UI
│   └── chain.py             # LangChain logic
│
├── api/
│   ├── main.py              # FastAPI app
│   └── routes.py            # LLM routing logic
│
├── requirements.txt         # Project dependencies
├── .env.example             # Environment variables template
└── README.md
```
 
---
 
## 📦 Dependencies
 
```txt
langchain
langchain-groq
langchain-openai
langchain-google-genai
langchain-anthropic
fastapi
uvicorn
streamlit
python-dotenv
```
 
---
 
## 🧠 What I Learned
 
- How LangChain chains work (Prompt → Model → Output)
- Integrating multiple LLMs through a unified LangChain interface
- Building a routing API to switch between LLM providers
- Building a Streamlit UI for an AI application
- Managing environment variables and API keys securely
---
 
## 🗺️ Learning Roadmap
 
This project is part of my Agentic AI learning journey:
 
- [x] Python Programming
- [x] NLP Foundations
- [x] Gen AI Fundamentals ← *You are here*
- [ ] Gen AI Projects
- [ ] Agentic AI Fundamentals
- [ ] Agentic AI Hands-on Practice
---
 
## 🤝 Contributing
 
This repo is meant to be beginner-friendly. If you find a bug, have a suggestion, or want to add a tutorial — feel free to open an issue or submit a pull request.
 
---
 
## 📬 Connect
 
**Ahmad Imran**
[GitHub](https://github.com/ahmadimrannn) • [LinkedIn](https://www.linkedin.com/in/ahmadimrannn)
 
---
 
⭐ If this helped you, consider giving the repo a star!
