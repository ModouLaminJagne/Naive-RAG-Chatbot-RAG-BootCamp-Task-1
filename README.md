# -Naive-RAG-Chatbot-RAG-BootCamp-Task-1
Build and deploy your own Naive Retrieval-Augmented Generation (RAG) system that can answer user questions based on custom documents of your choice.
# 🤖 Naive RAG Chatbot

A comprehensive Retrieval-Augmented Generation (RAG) chatbot built with LangChain that can answer questions based on Wikipedia articles.

## 🎯 Project Overview

This project implements a complete RAG pipeline that:
- Loads documents from Wikipedia
- Chunks them intelligently 
- Creates vector embeddings
- Stores them in a FAISS vector database
- Retrieves relevant context for user queries
- Generates answers using OpenAI's GPT models

## 🛠️ Tech Stack

- **Framework**: LangChain + Python
- **Document Source**: Wikipedia API
- **Embedding Model**: HuggingfaceEmbedding
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **LLM**: OpenAI GPT2
- **Interface**: Streamlit web app
- **Reranking**: Cosine similarity-based reranking

## 🚀 Features

### Core RAG Components ✅
- ✅ **Document Ingestion**: Wikipedia loader with configurable topics
- ✅ **Intelligent Chunking**: Recursive character splitter with overlap
- ✅ **Vector Embeddings**: OpenAI embeddings with FAISS storage
- ✅ **Retrieval System**: Similarity search with optional reranking
- ✅ **Generation**: Structured prompts with context injection
- ✅ **User Interface**: Interactive Streamlit web app

### Enhanced Features 🌟
- 🔍 **Enhanced Retrieval**: Semantic reranking for better results
- 📊 **Chunk Visualization**: See exactly what context was retrieved
- ⚙️ **Configurable Topics**: Load any Wikipedia topics you want
- 🎯 **Similarity Scores**: Transparency in retrieval quality
- 💡 **Example Queries**: Pre-built questions to get started

## 📋 Requirements

```
langchain>=0.1.0
langchain-community>=0.1.0
langchain-huggingface>=0.1.0
gradio>=4.15.0
streamlit>=1.28.0
openai>=1.3.0
faiss-cpu>=1.7.4
wikipedia>=1.4.0
scikit-learn>=1.3.0
numpy>=1.24.0
transformers>=4.30.0
sentence-transformers>=2.2.2
torch>=2.0.0
python-dotenv>=1.0.0
tiktoken>=0.5.0
beautifulsoup4>=4.12.0
requests>=2.31.0
```

## 🔧 Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd rag-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Get OpenAI API Key
1. Visit [OpenAI API Keys](https://platform.openai.com/api-keys)(If you are using OpenAI Key)
2. Create a new API key
3. Keep it secure - you'll enter it in the app

### 5. Run the Application
```bash
streamlit run hfr.py
```

### 6. Access the App
- Open your browser to `http://localhost:8501`
- Live Demo Available at (https://https://mlj-nrchatbot.streamlit.app/)

## 💻 Usage Guide

### I am not usin an OpenAI Key
<!-- ### Step 1: Configure API Key -->
<!-- - Enter your OpenAI API key in the sidebar
- The app will validate and initialize the RAG system -->

### Step 2: Load Documents
- Choose topics in the sidebar (default: AI, ML, NLP)
- Adjust max documents per topic (1-5)
- Click "Load Documents" 
- Wait for processing (typically 30-60 seconds)

### Step 3: Ask Questions
- Type your question in the main input box
- Optional: Enable "Enhanced Retrieval" for reranking
- View the generated answer and retrieved context chunks

## 🔍 Example Queries

Try these questions once you've loaded the default topics:

- **"What is artificial intelligence?"**
- **"How does machine learning work?"**  
- **"What are the applications of natural language processing?"**
- **"What is the difference between AI and ML?"**
- **"How do neural networks function?"**

## 📊 How It Works

### 1. Document Ingestion
```python
# Load Wikipedia articles
loader = WikipediaLoader(query=topic, load_max_docs=max_docs_per_topic)
docs = loader.load()
```

### 2. Chunking Strategy
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # ~1000 characters per chunk
    chunk_overlap=200,    # 200 character overlap
    separators=["\n\n", "\n", " ", ""]  # Smart splitting
)
```

### 3. Vector Storage
```python
# Create FAISS vector store
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=HuggingFaceEmbeddings()
    # embedding=OpenAIEmbeddings()
)
```

### 4. Retrieval Chain
```python
# LangChain Expression Language (LCEL)
retrieval_chain = (
    RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | prompt_template
    | llm
    | StrOutputParser()
)
```

### 5. Enhanced Retrieval (Optional)
- Retrieves more chunks initially (k=10)
- Reranks using cosine similarity
- Returns top k=4 most relevant chunks

## 🏗️ Architecture

```
User Query → Embedding → Vector Search → Context Retrieval → LLM → Answer
                ↓
            FAISS Store ← Document Chunks ← Text Splitter ← Wikipedia Loader
```

## 🎯 Evaluation Criteria Met

- ✅ **Core Functionality**: All 5 components implemented
- ✅ **Code Clarity**: Well-structured, commented code
- ✅ **LangChain Usage**: Proper use of loaders, embeddings, chains
- ✅ **Creativity**: Wikipedia integration + enhanced retrieval
- ✅ **User Interface**: Complete Streamlit web app

## 🚧 Known Limitations

1. **API Costs**: I Use HuggingFace Embedding so no need for an API cost
2. **Memory Usage**: FAISS stores all vectors in memory
3. **Topic Dependency**: Limited to Wikipedia content quality
4. **No Persistence**: Vector store recreated each session
5. **Rate Limits**: No rate limits

## 🔮 Future Enhancements

- [ ] **Persistent Storage**: Save vector store to disk
- [ ] **Multiple Sources**: PDF, web scraping, documents
- [ ] **Advanced Reranking**: CrossEncoder models
- [ ] **Conversation Memory**: Multi-turn conversations
- [ ] **Source Citations**: Link back to original content
- [ ] **Evaluation Metrics**: RAGAS integration
- [ ] **Deployment**: Docker + cloud hosting

## 🛡️ Security Notes

- Never commit API keys to version control (If you are using any)
- Use environment variables for production
- Implement proper rate limiting for production use
- Consider input sanitization for user queries

## 📈 Performance Tips

- **Chunking**: Experiment with chunk sizes (500-2000 chars)
- **Retrieval**: Adjust k value based on query complexity
- **Reranking**: Use for complex/ambiguous queries
- **Caching**: Consider caching embeddings for repeated queries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain**: For the excellent RAG framework
- **HuggingFace**: For free and powerful embedding and language models
- **Streamlit**: For rapid web app development
- **FAISS**: For efficient similarity search
- **Wikipedia**: For freely available knowledge

---

**Built with ❤️ by ML Jagne for the NSK.AI RAG BootCamp Project-1**
