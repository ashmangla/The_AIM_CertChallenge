<p align = "center" draggable="false" ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

# 🔧 HandyAssist - Your AI-Powered Appliance Manual Assistant

> **Smart, context-aware help for all your appliance questions!**

HandyAssist is an intelligent RAG (Retrieval Augmented Generation) agent that helps homeowners get instant answers from their appliance manuals. No more digging through PDF manuals or confusing diagrams - just ask a question and get clear, step-by-step instructions!

## 🌟 What Makes HandyAssist Special?

- **🧠 Context-Aware Intelligence**: The agent knows what manuals it has and only asks for details when needed
- **📊 Data-Driven Optimization**: Uses RAGAS evaluation to automatically select the best retrieval strategy
- **🎯 Production-Ready**: Semantic chunking + Cohere reranking for superior answer quality
- **🔍 Smart Retrieval**: Automatically searches existing manuals first, then helps find new ones via web search
- **💬 Natural Conversation**: Designed for everyday homeowners, not technicians

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+
- OpenAI API key
- Tavily API key (for web search)
- Cohere API key (for reranking)

### Installation

4. **Configure API Keys** ⚠️ **REQUIRED - System Won't Work Without This!**
   
   **Step 1: Copy the template**
   ```bash
   cd api
   cp .env.example .env
   ```
   
   **Step 2: Add YOUR real API keys to `api/.env`**
   
   Open `api/.env` in your text editor and **replace the placeholders** with your actual keys:
   
   ```bash
   # BEFORE (placeholder values - won't work!)
   OPENAI_API_KEY=your_openai_key_here
   TAVILY_API_KEY=your_tavily_key_here
   COHERE_API_KEY=your_cohere_key_here
   
   # AFTER (your real keys - will work!)
   OPENAI_API_KEY=sk-proj-abc123...  # Your actual OpenAI key
   TAVILY_API_KEY=tvly-dev-xyz789... # Your actual Tavily key
   COHERE_API_KEY=f7W084Lh...        # Your actual Cohere key
   ```
   
   **Where to Get API Keys:**
   - OpenAI: https://platform.openai.com/api-keys
   - Tavily: https://tavily.com (sign up for free)
   - Cohere: https://dashboard.cohere.ai/api-keys
   
   **Security Notes**:
   - ✅ `.env` is already in `.gitignore` - it will **never** be committed to Git
   - ✅ For production, set these as environment variables in your hosting platform
   - ❌ **Never** hardcode API keys in source files or commit them to Git
   - 💡 The system reads from `.env` automatically on startup

5. **Add your appliance manuals**
   
   Place PDF manuals in `api/data/` directory

6. **Start the servers**
   ```bash
   # Use the convenience script (Mac/Linux)
   bash restart-servers.sh
   
   # Or manually:
   # Terminal 1 - Backend
   cd api && uvicorn app:app --reload --port 8000
   
   # Terminal 2 - Frontend
   cd frontend && npm run dev
   ```

7. **Open your browser**
   
   Navigate to `http://localhost:3000` and start chatting!
   Sample ques : 
   (1) how do i change the water filter, 
   (2) can you also tell me how to turn on my airfryer generics model

## 🏗️ Architecture

### Backend (`/api`)

```
api/
├── app.py                 # FastAPI application
├── agents/
│   └── rag_agent.py      # LangGraph ReAct agent
├── tools/
│   └── tools.py          # Retrieval + Tavily search tools
├── evaluate_rag/
│   └── evaluate.py       # RAGAS evaluation suite
├── config/
│   └── retrieval_config.json  # Best retrieval config (auto-generated)
└── data/
    └── *.pdf             # Your appliance manuals
```

### Frontend (`/frontend`)

- **Next.js 14** with App Router
- **Tailwind CSS** for styling
- **Streaming responses** for real-time chat
- **Modern, responsive UI**

## 🎯 How It Works

### 1. **Intelligent Agent Workflow**

```
User Question
    ↓
Agent checks retrieve_information tool
    ↓
Manual found? → Answer immediately ✅
    ↓
No manual? → Ask for appliance details → Use Tavily to search web
```

### 2. **Optimized RAG Pipeline**

HandyAssist uses a **data-driven approach** to retrieval:

1. **Evaluation**: Run `api/evaluate_rag/evaluate.py` to test different strategies
2. **Optimization**: RAGAS scores 6 combinations (2 chunking × 3 retrieval methods)
3. **Auto-Config**: Best config is saved to `config/retrieval_config.json`
4. **Dynamic Loading**: `tools.py` reads config and uses winning strategy

**Current Champion**: Semantic Chunking + Cohere Rerank (0.825 avg score)

### 3. **RAGAS Evaluation Metrics**

We evaluate on 7 metrics:
- `context_precision`: Are retrieved chunks relevant?
- `context_recall`: Did we retrieve all necessary info?
- `faithfulness`: Is the answer grounded in context?
- `answer_relevancy`: Does the answer address the question?
- `answer_correctness`: Is the answer factually correct?
- `coherence`: Is the answer logically consistent?
- `conciseness`: Is the answer brief and efficient?

## 📊 Running Evaluations

To optimize your RAG system for your specific manuals:

     ```bash
cd api
python evaluate_rag/evaluate.py
```

This will:
1. Generate realistic homeowner questions (persona-driven SDG)
2. Test all 6 chunking/retrieval combinations
3. Evaluate with RAGAS metrics
4. Save best config to `config/retrieval_config.json`
5. Display results table

**Next time you restart the backend, it will automatically use the winning strategy!**

## 🔧 Configuration

### Chunking Strategies

- **Recursive**: Fixed-size chunks (750 chars, 50 overlap)
- **Semantic**: Meaning-based boundaries (percentile breakpoint)

### Retrieval Methods

- **Naive**: Standard vector similarity (k=5)
- **Cohere Rerank**: Re-ranks top 5 → best 3
- **Multi-Query**: Generates multiple query variations

### Language Filtering

By default, only English documents are used.

```python
docs, rag_documents = load_and_prepare_documents(
    language_filter='en',  # Change to None for all languages
    chunking_strategy='semantic'
)
```

## 🎨 Customization

### Change Agent Behavior

Edit `api/agents/rag_agent.py`:

```python
HANDYMAN_SYSTEM_PROMPT = """You are a handyman assistant..."""
```

### Adjust Evaluation Size

Edit `api/evaluate_rag/evaluate.py`:

```python
testset_size=5  # Increase for more test questions
```


## 🛠️ Tech Stack

### **Document Processing Layer**
- **PDF Extraction**: PyMuPDF (`pymupdf`)
- **Text Chunking**: 
  - LangChain `SemanticChunker` (meaning-based boundaries)
  - `RecursiveCharacterTextSplitter` (fixed-size with overlap)
- **Tokenization**: `tiktoken` (OpenAI tokenizer)
- **Language Detection**: `langdetect` (filter English-only docs)

### **Embedding & Retrieval Layer**
- **Embedding Model**: OpenAI `text-embedding-3-small`
- **Vector Database**: Qdrant (in-memory for development)
- **Retrieval Methods**: 
  - **Naive**: Standard vector similarity (k=5)
  - **Cohere Rerank**: Re-ranks top-k results for better precision
  - **Multi-Query**: Generates query variations for better recall

### **Agent & Orchestration Layer**
- **Agent Framework**: LangGraph (ReAct pattern)
- **LLM**: OpenAI `gpt-4o-mini` (primary reasoning engine)
- **Tools**: 
  - `retrieve_information` - RAG tool for manual search
  - `tavily_tool` - Web search for unknown appliances (Tavily API)
- **Context Management**: Dynamic system prompts with homeowner persona

### **Evaluation & Optimization Layer** 🆕
- **Evaluation Framework**: RAGAS 0.2.10
- **Metrics** (7 total):
  - `context_precision` - Relevance of retrieved chunks
  - `context_recall` - Completeness of retrieval
  - `faithfulness` - Answer grounded in context
  - `answer_relevancy` - Addresses the question
  - `answer_correctness` - Factual accuracy
  - `coherence` - Logical consistency
  - `conciseness` - Brevity and efficiency
- **Synthetic Data Generation**: Persona-driven question generation
- **Stratified Sampling**: Efficient document sampling for SDG
- **Auto-Configuration**: JSON-based dynamic config loading

### **Backend API**
- **Framework**: FastAPI (Python)
- **Validation**: Pydantic models
- **File Handling**: `python-multipart`
- **Environment**: `python-dotenv`
- **Async Streaming**: `asyncio` for real-time responses
- **CORS**: Middleware for cross-origin requests

### **Frontend Layer**
- **Framework**: Next.js 14 (App Router)
- **UI Library**: React 18
- **Styling**: Tailwind CSS (utility-first)
- **Language**: TypeScript (type safety)
- **HTTP Client**: Native `fetch` API

### **Infrastructure & DevOps**
- **Deployment**: Local development (localhost)
- **Storage**: Local filesystem (`api/data/`)
- **Monitoring**: RAGAS for RAG performance tracking
- **Process Management**: Custom restart script (`restart-servers.sh`)
- **Port Management**: Backend (8000), Frontend (3000)

## 📁 Project Structure

```
The_AIM_CertChallenge/
├── api/                          # Backend
│   ├── app.py                   # FastAPI app
│   ├── agents/                  # LangGraph agents
│   ├── tools/                   # RAG + web search tools
│   ├── evaluate_rag/            # RAGAS evaluation
│   │   ├── evaluate.py
│   │   ├── golden_dataset/      # Test questions
│   │   └── results/             # Evaluation results
│   ├── config/                  # Dynamic config
│   │   └── retrieval_config.json
│   └── data/                    # PDF manuals
├── frontend/                     # Next.js frontend
│   ├── app/
│   │   ├── page.tsx            # Landing page
│   │   └── chat/
│   │       └── page.tsx        # Chat interface
│   └── package.json
├── requirements.txt             # Python dependencies
├── restart-servers.sh           # Convenience script
└── README.md                    # You are here!
```

## 🧪 Example Questions

Try these with the GE Fridge manual:

- "How do I change the water filter?"
- "What's the filter capacity?"
- "How do I prevent water leakage?"
- "What are the installation requirements?"
- "How often should I replace the filter?"


## 📚 Learn More

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)

## 🎉 Acknowledgments

Built with ❤️ at [AI Makerspace](https://github.com/AI-Maker-Space)

## 📜 License

This project is open source and available under the MIT License.

---

**Made with 🔧 by AI Engineers, for Homeowners**

Questions? Issues? Open a GitHub issue or reach out to the AI Makerspace community!
