#!/usr/bin/env python3
"""
Enhanced Free RAG Chatbot with Reranker and Multiple Document Sets
"""

import os
import streamlit as st
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List, Dict, Tuple
import pickle
from datetime import datetime
from sentence_transformers import CrossEncoder
import numpy as np

class RerankerRetriever:
    """Custom retriever that uses reranking to improve retrieval quality"""
    
    def __init__(self, vector_store, reranker_model_name="cross-encoder/ms-marco-MiniLM-L-2-v2", k=20, final_k=4):
        self.vector_store = vector_store
        self.k = k  # Number of documents to retrieve initially
        self.final_k = final_k  # Number of documents to return after reranking
        
        # Initialize reranker
        try:
            self.reranker = CrossEncoder(reranker_model_name)
            self.reranker_available = True
            st.success(f"✅ Reranker loaded: {reranker_model_name}")
        except Exception as e:
            st.warning(f"⚠️ Reranker not available: {e}. Using basic retrieval.")
            self.reranker = None
            self.reranker_available = False
    
    def get_relevant_documents(self, query: str):
        """Retrieve and rerank documents"""
        # Initial retrieval with higher k
        initial_docs = self.vector_store.similarity_search(query, k=self.k)
        
        if not self.reranker_available or len(initial_docs) <= self.final_k:
            return initial_docs[:self.final_k]
        
        # Prepare query-document pairs for reranking
        query_doc_pairs = [(query, doc.page_content) for doc in initial_docs]
        
        # Get reranking scores
        scores = self.reranker.predict(query_doc_pairs)
        
        # Sort documents by reranking scores
        doc_score_pairs = list(zip(initial_docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top documents after reranking
        reranked_docs = [doc for doc, score in doc_score_pairs[:self.final_k]]
        
        return reranked_docs

class DocumentSet:
    """Class to manage individual document sets"""
    
    def __init__(self, name: str, topics: List[str], vector_store=None, created_at=None):
        self.name = name
        self.topics = topics
        self.vector_store = vector_store
        self.created_at = created_at or datetime.now()
        self.doc_count = 0
    
    def to_dict(self):
        return {
            'name': self.name,
            'topics': self.topics,
            'created_at': self.created_at.isoformat(),
            'doc_count': self.doc_count
        }

class EnhancedRAGChatbot:
    def __init__(self):
        """Initialize the Enhanced RAG Chatbot"""
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize LLM
        self.llm = self._setup_free_llm()
        
        # Document sets management
        self.document_sets: Dict[str, DocumentSet] = {}
        self.active_document_set: str = None
        self.qa_chain = None
        self.retriever = None
        
        # Load existing document sets
        self._load_document_sets()
        
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
            # Using Flan-T5 for better Q&A performance
            model_name = "google/flan-t5-small"  # Better for Q&A than GPT-2
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            pipe = pipeline(
                "text2text-generation",  # Changed for T5
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                device=-1  # CPU
            )
            
            return HuggingFacePipeline(pipeline=pipe)
            
        except Exception as e:
            st.warning(f"Error loading Flan-T5: {e}. Falling back to GPT-2")
            return self._setup_fallback_llm()
    
    def _setup_fallback_llm(self):
        """Fallback to GPT-2"""
        pipe = pipeline(
            "text-generation",
            model="gpt2",
            max_new_tokens=200,
            temperature=0.7,
            device=-1
        )
        return HuggingFacePipeline(pipeline=pipe)
    
    def _load_document_sets(self):
        """Load existing document sets from disk"""
        if os.path.exists("document_sets.pkl"):
            try:
                with open("document_sets.pkl", "rb") as f:
                    saved_sets = pickle.load(f)
                    for name, data in saved_sets.items():
                        doc_set = DocumentSet(
                            name=data['name'],
                            topics=data['topics'],
                            created_at=datetime.fromisoformat(data['created_at'])
                        )
                        doc_set.doc_count = data.get('doc_count', 0)
                        
                        # Try to load vector store
                        if os.path.exists(f"vector_stores/{name}"):
                            try:
                                doc_set.vector_store = FAISS.load_local(
                                    f"vector_stores/{name}",
                                    self.embeddings,
                                    allow_dangerous_deserialization=True
                                )
                            except Exception as e:
                                st.warning(f"Could not load vector store for {name}: {e}")
                        
                        self.document_sets[name] = doc_set
                st.success(f"✅ Loaded {len(self.document_sets)} existing document sets")
            except Exception as e:
                st.warning(f"Could not load existing document sets: {e}")
    
    def _save_document_sets(self):
        """Save document sets metadata to disk"""
        os.makedirs("vector_stores", exist_ok=True)
        
        # Save metadata
        save_data = {name: doc_set.to_dict() for name, doc_set in self.document_sets.items()}
        with open("document_sets.pkl", "wb") as f:
            pickle.dump(save_data, f)
        
        # Save vector stores
        for name, doc_set in self.document_sets.items():
            if doc_set.vector_store is not None:
                doc_set.vector_store.save_local(f"vector_stores/{name}")
    
    def create_document_set(self, name: str, topics: List[str], max_docs_per_topic: int = 2) -> bool:
        """Create a new document set"""
        if name in self.document_sets:
            st.error(f"Document set '{name}' already exists!")
            return False
        
        st.info(f"📚 Creating document set: {name}")
        
        # Load documents
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
        
        # Create vector store
        st.info("🔮 Creating embeddings and vector store...")
        with st.spinner("Computing embeddings..."):
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
        
        # Create document set
        doc_set = DocumentSet(name=name, topics=topics, vector_store=vector_store)
        doc_set.doc_count = len(chunks)
        self.document_sets[name] = doc_set
        
        # Save to disk
        self._save_document_sets()
        
        st.success(f"✅ Document set '{name}' created successfully!")
        return True
    
    def switch_document_set(self, name: str) -> bool:
        """Switch to a different document set"""
        if name not in self.document_sets:
            st.error(f"Document set '{name}' not found!")
            return False
        
        doc_set = self.document_sets[name]
        if doc_set.vector_store is None:
            st.error(f"Vector store for '{name}' is not available!")
            return False
        
        self.active_document_set = name
        
        # Create enhanced retriever with reranker
        self.retriever = RerankerRetriever(
            vector_store=doc_set.vector_store,
            k=20,  # Retrieve more documents initially
            final_k=4  # Return top 4 after reranking
        )
        
        # Setup QA chain with enhanced retriever
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt_template}
        )
        
        st.success(f"✅ Switched to document set: {name}")
        return True
    
    def delete_document_set(self, name: str) -> bool:
        """Delete a document set"""
        if name not in self.document_sets:
            return False
        
        # Remove from memory
        del self.document_sets[name]
        
        # Remove vector store files
        import shutil
        if os.path.exists(f"vector_stores/{name}"):
            shutil.rmtree(f"vector_stores/{name}")
        
        # Save updated metadata
        self._save_document_sets()
        
        # Reset active set if it was deleted
        if self.active_document_set == name:
            self.active_document_set = None
            self.qa_chain = None
            self.retriever = None
        
        return True
    
    def answer_question(self, question: str) -> str:
        """Answer a question using the enhanced RAG system"""
        if not self.qa_chain:
            return "❌ No document set is active. Please select or create a document set first."
        
        try:
            with st.spinner("🤖 Generating answer with reranking..."):
                result = self.qa_chain({"query": question})
                return result["result"]
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}"

def main():
    st.set_page_config(
        page_title="Enhanced RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🚀 Enhanced RAG Chatbot")
    st.markdown("*Advanced Retrieval-Augmented Generation with Reranking and Multiple Document Sets*")
    st.markdown("---")
    
    # Initialize chatbot
    if "enhanced_chatbot" not in st.session_state:
        with st.spinner("🔄 Initializing enhanced models..."):
            st.session_state.enhanced_chatbot = EnhancedRAGChatbot()
    
    chatbot = st.session_state.enhanced_chatbot
    
    # Sidebar for document set management
    st.sidebar.header("📚 Document Sets")
    
    # Display existing document sets
    if chatbot.document_sets:
        st.sidebar.subheader("Available Sets:")
        for name, doc_set in chatbot.document_sets.items():
            col1, col2, col3 = st.sidebar.columns([3, 1, 1])
            
            with col1:
                is_active = name == chatbot.active_document_set
                status = "🟢" if is_active else "⚪"
                if st.button(f"{status} {name}", key=f"select_{name}"):
                    chatbot.switch_document_set(name)
                    st.rerun()
            
            with col2:
                if st.button("📝", key=f"info_{name}", help="Info"):
                    st.sidebar.write(f"Topics: {', '.join(doc_set.topics)}")
                    st.sidebar.write(f"Chunks: {doc_set.doc_count}")
                    st.sidebar.write(f"Created: {doc_set.created_at.strftime('%Y-%m-%d')}")
            
            with col3:
                if st.button("🗑️", key=f"delete_{name}", help="Delete"):
                    if chatbot.delete_document_set(name):
                        st.sidebar.success(f"Deleted {name}")
                        st.rerun()
    
    # Create new document set
    st.sidebar.subheader("➕ Create New Set")
    
    new_set_name = st.sidebar.text_input("Set Name", placeholder="e.g., AI Research")
    
    topics_input = st.sidebar.text_area(
        "Topics (one per line)",
        value="Artificial Intelligence\nMachine Learning\nDeep Learning",
        help="Enter Wikipedia topics"
    )
    
    topics = [topic.strip() for topic in topics_input.split("\n") if topic.strip()]
    max_docs = st.sidebar.slider("Max docs per topic", 1, 5, 2)
    
    if st.sidebar.button("🔧 Create Document Set"):
        if new_set_name and topics:
            success = chatbot.create_document_set(new_set_name, topics, max_docs)
            if success:
                st.rerun()
        else:
            st.sidebar.error("Please provide a name and topics!")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        if chatbot.active_document_set:
            active_set = chatbot.document_sets[chatbot.active_document_set]
            st.info(f"📖 Active Set: **{chatbot.active_document_set}** | Topics: {', '.join(active_set.topics)}")
            
            question = st.text_input(
                "🤔 Your Question:",
                placeholder="Ask anything about the active document set...",
            )
            
            if question:
                answer = chatbot.answer_question(question)
                st.subheader("🎯 Answer:")
                st.write(answer)
                
                # Show retrieval info
                if chatbot.retriever and chatbot.retriever.reranker_available:
                    st.caption("✨ Enhanced with reranking for better accuracy")
        else:
            st.info("👈 Please select or create a document set first.")
    
    with col2:
        st.header("ℹ️ System Info")
        
        # Active set info
        if chatbot.active_document_set:
            active_set = chatbot.document_sets[chatbot.active_document_set]
            st.metric("Document Chunks", active_set.doc_count)
            st.metric("Topics", len(active_set.topics))
        
        # Available sets
        st.metric("Total Sets", len(chatbot.document_sets))
        
        # Reranker status
        if chatbot.retriever:
            reranker_status = "✅ Active" if chatbot.retriever.reranker_available else "❌ Disabled"
        else:
            reranker_status = "⚪ Not initialized"
        st.metric("Reranker", reranker_status)
    
    # Example questions
    if chatbot.active_document_set:
        st.markdown("---")
        st.subheader("💡 Example Questions")
        
        examples = [
            "What are the main types of machine learning?",
            "How does deep learning differ from traditional ML?",
            "What are the applications of artificial intelligence?"
        ]
        
        cols = st.columns(len(examples))
        for i, example in enumerate(examples):
            if cols[i].button(example, key=f"example_{i}"):
                st.info(f"Try asking: {example}")
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Enhanced Features:**
        - **Reranker**: Uses cross-encoder/ms-marco-MiniLM-L-2-v2 for better retrieval
        - **Multiple Document Sets**: Switch between different knowledge bases
        - **Persistent Storage**: Document sets saved to disk
        - **Better LLM**: Flan-T5 for improved Q&A performance
        
        **Retrieval Pipeline:**
        1. Initial retrieval: Get top 20 similar documents
        2. Reranking: Use cross-encoder to rerank by relevance
        3. Final selection: Return top 4 most relevant documents
        4. Answer generation: Use selected context to generate answer
        
        **Performance Notes:**
        - First time setup downloads ~1GB of models
        - Reranking adds ~2-3 seconds but improves accuracy significantly
        - Document sets are cached for quick switching
        """)

if __name__ == "__main__":
    main()