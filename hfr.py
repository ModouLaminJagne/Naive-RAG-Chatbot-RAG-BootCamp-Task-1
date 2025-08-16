#!/usr/bin/env python3
"""
Free RAG Chatbot using Hugging Face models (no API keys required)
"""

import os
import streamlit as st
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List

class FreeRAGChatbot:
    def __init__(self):
        """Initialize the Free RAG Chatbot with Hugging Face models"""
        
        # Initialize embeddings (completely free)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Use 'cuda' if you have GPU
        )
        
        # Initialize LLM (free but requires download)
        self.llm = self._setup_free_llm()
        self.vector_store = None
        self.qa_chain = None
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following context to answer the question. If you cannot find the answer in the context, say "I don't have enough information to answer this question."

Context: {context}

Question: {question}

Answer:"""
        )
    
    def _setup_free_llm(self):
        """Setup a free local LLM"""
        try:
            # Option 1: Microsoft DialoGPT (conversational)
            # model_name = "microsoft/DialoGPT-medium"
            
            # Option 2: Google Flan-T5 (better for Q&A)
            model_name = "google/flan-t5-small"
            
            # Option 3: GPT-2 (lightweight)
            # model_name = "gpt2"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=200,
                # max_length=512,
                temperature=0.7,
                do_sample=True,
                device=0  # Use CPU (-1) or 0 for GPU
            )
            
            return HuggingFacePipeline(pipeline=pipe)
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            # Fallback to a very simple model
            return self._setup_fallback_llm()
    
    def _setup_fallback_llm(self):
        """Fallback to simplest possible model"""
        pipe = pipeline(
            "text-generation",
            model="gpt2",
            max_length=200,
            temperature=0.7,
            device=0
        )
        return HuggingFacePipeline(pipeline=pipe)
    
    def load_documents(self, topics: List[str], max_docs_per_topic: int = 2):
        """Load and process documents (same as before)"""
        st.info("📚 Loading documents from Wikipedia...")
        
        all_documents = []
        
        for topic in topics:
            try:
                loader = WikipediaLoader(query=topic, load_max_docs=max_docs_per_topic)
                docs = loader.load()
                all_documents.extend(docs)
                st.success(f"✅ Loaded {len(docs)} documents for topic: {topic}")
            except Exception as e:
                st.error(f"❌ Error loading topic {topic}: {str(e)}")
        
        if not all_documents:
            st.error("No documents were loaded!")
            return False
        
        # Split documents
        st.info("🔪 Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(all_documents)
        st.success(f"✅ Created {len(chunks)} document chunks")
        
        # Create vector store (this takes a moment for embeddings)
        st.info("🔮 Creating embeddings and vector store...")
        with st.spinner("Computing embeddings (this may take a minute)..."):
            self.vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
        st.success("✅ Vector store created successfully!")
        
        # Setup QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": self.prompt_template}
        )
        
        return True
    
    def answer_question(self, question: str) -> str:
        """Answer a question using the free RAG system"""
        if not self.qa_chain:
            return "❌ RAG system not initialized. Please load documents first."
        
        try:
            with st.spinner("🤖 Generating answer..."):
                result = self.qa_chain({"query": question})
                return result["result"]
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}"

def main():
    st.set_page_config(
        page_title="🤖 Naive RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Naive RAG Chatbot")
    st.markdown("*A Naive Retrieval-Augmented Generation (RAG) system that can answer user questions based on custom documents of your choice.*")
    st.markdown("*No API keys required - uses free Hugging Face models*")
    st.markdown("---")
    
    # Important notice
    st.info("""
    🔋 **Performance Note**: This Chatbot uses free, local models which are:
    - ✅ Completely free forever
    - ✅ No API keys required  
    - ✅ Privacy-friendly (runs locally)
    - ⚠️ Slower than OpenAI models
    - ⚠️ May require model downloads (first time)
    """)
    
    # Initialize chatbot
    if "free_chatbot" not in st.session_state:
        with st.spinner("🔄 Initializing free models (first time setup)..."):
            st.session_state.free_chatbot = FreeRAGChatbot()
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Topic selection
    default_topics = ["Artificial Intelligence", "Machine Learning", "Python Programming"]
    
    topics_input = st.sidebar.text_area(
        "Topics to load (one per line)",
        value="\n".join(default_topics),
        help="Enter Wikipedia topics to load"
    )
    
    topics = [topic.strip() for topic in topics_input.split("\n") if topic.strip()]
    max_docs = st.sidebar.slider("Max documents per topic", 1, 3, 2)
    
    # Load documents button
    if st.sidebar.button("🔄 Load Documents"):
        success = st.session_state.free_chatbot.load_documents(topics, max_docs)
        if success:
            st.session_state.documents_loaded = True
            st.sidebar.success("✅ Documents loaded successfully!")
        else:
            st.sidebar.error("❌ Failed to load documents")
    
    # Main interface
    st.header("💬 Ask Questions")
    
    # Check if documents are loaded
    if not hasattr(st.session_state, 'documents_loaded'):
        st.info("👈 Please load documents using the sidebar first.")
        st.markdown("""
        ### 🚀 Quick Start:
        1. Choose topics in the sidebar (default ones work great)
        2. Click "Load Documents" 
        3. Wait for models to download and process (2-5 minutes first time)
        4. Ask questions!
        """)
        return
    
    # Question input
    question = st.text_input(
        "🤔 Your Question:",
        placeholder="e.g., What is machine learning?",
        help="Ask any question about the loaded documents"
    )
    
    if question:
        answer = st.session_state.free_chatbot.answer_question(question)
        
        # Display answer
        st.subheader("🎯 Answer:")
        st.write(answer)
    
    # Example queries
    st.markdown("---")
    st.subheader("💡 Example Questions")
    
    example_cols = st.columns(3)
    examples = [
        "What is artificial intelligence?",
        "How does machine learning work?", 
        "What is Python used for?"
    ]
    
    for i, example in enumerate(examples):
        if example_cols[i].button(example, key=f"ex_{i}"):
            # This would set the question (simplified for demo)
            st.info(f"Try asking: {example}")
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Models Used:**
        - **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
        - **LLM**: microsoft/DialoGPT-medium or similar
        - **Vector Store**: FAISS (local, no cloud)
        
        **First Time Setup:**
        - Downloads ~500MB of models
        - Takes 2-5 minutes depending on internet speed
        - Models cached locally for future use
        
        **Performance:**
        - Embeddings: ~10 seconds for typical document set
        - Question answering: ~5-15 seconds per query
        - No usage limits or costs!
        """)

if __name__ == "__main__":
    main()