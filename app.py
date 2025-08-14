import os
import streamlit as st
import openai
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RAGChatbot:
    def __init__(self, openai_api_key: str):
        """
        Initialize the RAG Chatbot with OpenAI API key
        
        Args:
            openai_api_key: Your OpenAI API key
        """
        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo")
        self.vector_store = None
        self.retrieval_chain = None
        
        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided context.
Use the context below to answer the user's question. If the answer cannot be found 
in the context, say "I don't have enough information in the provided context to answer this question."

Context:
{context}

Question: {question}

Answer: Please provide a comprehensive answer based on the context above.
        """)
    
    def load_and_process_documents(self, topics: List[str], max_docs_per_topic: int = 2) -> List:
        """
        Load documents from Wikipedia and process them
        
        Args:
            topics: List of Wikipedia topics to load
            max_docs_per_topic: Maximum number of documents per topic
            
        Returns:
            List of processed document chunks
        """
        st.info("📚 Loading documents from Wikipedia...")
        
        all_documents = []
        
        for topic in topics:
            try:
                # Load Wikipedia articles
                loader = WikipediaLoader(query=topic, load_max_docs=max_docs_per_topic)
                docs = loader.load()
                all_documents.extend(docs)
                st.success(f"✅ Loaded {len(docs)} documents for topic: {topic}")
            except Exception as e:
                st.error(f"❌ Error loading topic {topic}: {str(e)}")
        
        if not all_documents:
            st.error("No documents were loaded!")
            return []
        
        # Split documents into chunks
        st.info("🔪 Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Each chunk is ~1000 characters
            chunk_overlap=200,  # 200 character overlap between chunks
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Split on paragraphs first, then sentences
        )
        
        chunks = text_splitter.split_documents(all_documents)
        st.success(f"✅ Created {len(chunks)} document chunks")
        
        return chunks
    
    def create_vector_store(self, documents: List) -> None:
        """
        Create vector store from documents
        
        Args:
            documents: List of document chunks to vectorize
        """
        st.info("🔮 Creating embeddings and vector store...")
        
        try:
            # Create vector store with FAISS
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            st.success("✅ Vector store created successfully!")
        except Exception as e:
            st.error(f"❌ Error creating vector store: {str(e)}")
            raise e
    
    def setup_retrieval_chain(self, k: int = 4) -> None:
        """
        Setup the retrieval chain for RAG
        
        Args:
            k: Number of documents to retrieve
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please load documents first.")
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Create the retrieval chain using LCEL (LangChain Expression Language)
        self.retrieval_chain = (
            RunnableParallel({
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough()
            })
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        st.success("✅ Retrieval chain setup complete!")
    
    def _format_docs(self, docs) -> str:
        """Format retrieved documents into a single string"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def enhanced_retrieval(self, query: str, k_initial: int = 10, k_final: int = 4) -> List:
        """
        Enhanced retrieval with reranking
        
        Args:
            query: User query
            k_initial: Initial number of documents to retrieve
            k_final: Final number of documents after reranking
            
        Returns:
            List of reranked documents
        """
        if not self.vector_store:
            return []
        
        # Initial retrieval
        initial_docs = self.vector_store.similarity_search(query, k=k_initial)
        
        if len(initial_docs) <= k_final:
            return initial_docs
        
        # Rerank using embeddings similarity
        query_embedding = self.embeddings.embed_query(query)
        doc_embeddings = [self.embeddings.embed_query(doc.page_content) for doc in initial_docs]
        
        # Calculate cosine similarities
        similarities = []
        for doc_emb in doc_embeddings:
            sim = cosine_similarity([query_embedding], [doc_emb])[0][0]
            similarities.append(sim)
        
        # Sort by similarity and return top k_final
        doc_sim_pairs = list(zip(initial_docs, similarities))
        doc_sim_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in doc_sim_pairs[:k_final]]
    
    def answer_question(self, question: str, use_enhanced_retrieval: bool = False) -> str:
        """
        Answer a question using the RAG system
        
        Args:
            question: User question
            use_enhanced_retrieval: Whether to use enhanced retrieval with reranking
            
        Returns:
            Generated answer
        """
        if not self.retrieval_chain:
            return "❌ RAG system not properly initialized. Please load documents first."
        
        try:
            if use_enhanced_retrieval:
                # Use enhanced retrieval
                relevant_docs = self.enhanced_retrieval(question)
                context = self._format_docs(relevant_docs)
                
                # Use prompt template directly
                prompt = self.prompt_template.format(context=context, question=question)
                response = self.llm.invoke(prompt)
                return response.content
            else:
                # Use standard retrieval chain
                response = self.retrieval_chain.invoke(question)
                return response
                
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}"
    
    def get_relevant_chunks(self, question: str, k: int = 4) -> List[Dict]:
        """
        Get relevant document chunks for a question
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with metadata
        """
        if not self.vector_store:
            return []
        
        docs = self.vector_store.similarity_search_with_score(question, k=k)
        
        chunks = []
        for doc, score in docs:
            chunks.append({
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "title": doc.metadata.get("title", "Unknown"),
                "similarity_score": 1 - score  # Convert distance to similarity
            })
        
        return chunks

def main():
    st.set_page_config(
        page_title="🤖 RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Naive RAG Chatbot")
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "🔑 OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key to use the chatbot"
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
        st.info("💡 You can get your API key from: https://platform.openai.com/api-keys")
        return
    
    # Initialize chatbot
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot(api_key)
    
    # Topic selection
    st.sidebar.subheader("📚 Document Topics")
    default_topics = ["Artificial Intelligence", "Machine Learning", "Natural Language Processing"]
    
    topics_input = st.sidebar.text_area(
        "Topics to load (one per line)",
        value="\n".join(default_topics),
        help="Enter Wikipedia topics to load, one per line"
    )
    
    topics = [topic.strip() for topic in topics_input.split("\n") if topic.strip()]
    
    max_docs = st.sidebar.slider("Max documents per topic", 1, 5, 2)
    
    # Load documents button
    if st.sidebar.button("🔄 Load Documents"):
        with st.spinner("Loading and processing documents..."):
            # Load and process documents
            chunks = st.session_state.chatbot.load_and_process_documents(topics, max_docs)
            
            if chunks:
                # Create vector store
                st.session_state.chatbot.create_vector_store(chunks)
                
                # Setup retrieval chain
                st.session_state.chatbot.setup_retrieval_chain()
                
                st.session_state.documents_loaded = True
                st.sidebar.success("✅ Documents loaded successfully!")
            else:
                st.sidebar.error("❌ Failed to load documents")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Check if documents are loaded
        if not hasattr(st.session_state, 'documents_loaded'):
            st.info("👈 Please load documents using the sidebar first.")
            return
        
        # Question input
        question = st.text_input(
            "🤔 Your Question:",
            placeholder="e.g., What is artificial intelligence?",
            help="Ask any question about the loaded documents"
        )
        
        # Options
        col1a, col1b = st.columns(2)
        with col1a:
            use_enhanced = st.checkbox("🔍 Use Enhanced Retrieval", help="Uses reranking for better results")
        
        if question:
            with st.spinner("Thinking..."):
                # Get answer
                answer = st.session_state.chatbot.answer_question(question, use_enhanced)
                
                # Display answer
                st.subheader("🎯 Answer:")
                st.write(answer)
    
    with col2:
        st.header("📋 Document Chunks")
        
        if hasattr(st.session_state, 'documents_loaded') and question:
            with st.spinner("Finding relevant chunks..."):
                chunks = st.session_state.chatbot.get_relevant_chunks(question)
                
                if chunks:
                    st.subheader("🔍 Retrieved Context:")
                    for i, chunk in enumerate(chunks, 1):
                        with st.expander(f"Chunk {i} - {chunk['title']} (Score: {chunk['similarity_score']:.3f})"):
                            st.write(chunk['content'])
                            st.caption(f"Source: {chunk['source']}")
    
    # Example queries
    st.markdown("---")
    st.subheader("💡 Example Questions")
    example_questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the applications of natural language processing?",
        "What is the difference between AI and ML?",
        "How do neural networks function?"
    ]
    
    cols = st.columns(len(example_questions))
    for i, eq in enumerate(example_questions):
        if cols[i].button(f"Try: {eq}", key=f"example_{i}"):
            st.experimental_rerun()

if __name__ == "__main__":
    main()