#!/usr/bin/env python3
"""
CLI version of the RAG Chatbot for terminal usage
"""

import os
import sys
from typing import List
import argparse
from dotenv import load_dotenv

# Import our RAG components (assuming they're in the same directory)
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class CLIRAGChatbot:
    def __init__(self, openai_api_key: str):
        """Initialize the CLI RAG Chatbot"""
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
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
    
    def load_documents(self, topics: List[str], max_docs_per_topic: int = 2):
        """Load and process documents from Wikipedia"""
        print("📚 Loading documents from Wikipedia...")
        
        all_documents = []
        
        for topic in topics:
            try:
                print(f"  Loading: {topic}")
                loader = WikipediaLoader(query=topic, load_max_docs=max_docs_per_topic)
                docs = loader.load()
                all_documents.extend(docs)
                print(f"  ✅ Loaded {len(docs)} documents for {topic}")
            except Exception as e:
                print(f"  ❌ Error loading {topic}: {str(e)}")
        
        if not all_documents:
            print("❌ No documents were loaded!")
            return False
        
        print(f"\n🔪 Splitting {len(all_documents)} documents into chunks...")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(all_documents)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Create vector store
        print("\n🔮 Creating embeddings and vector store...")
        try:
            self.vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            print("✅ Vector store created!")
        except Exception as e:
            print(f"❌ Error creating vector store: {str(e)}")
            return False
        
        # Setup retrieval chain
        print("⚙️ Setting up retrieval chain...")
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        self.retrieval_chain = (
            RunnableParallel({
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough()
            })
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        print("✅ RAG system ready!\n")
        return True
    
    def _format_docs(self, docs) -> str:
        """Format retrieved documents into a single string"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def answer_question(self, question: str) -> str:
        """Answer a question using the RAG system"""
        if not self.retrieval_chain:
            return "❌ RAG system not initialized. Please load documents first."
        
        try:
            response = self.retrieval_chain.invoke(question)
            return response
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}"
    
    def show_relevant_chunks(self, question: str, k: int = 4):
        """Show relevant document chunks for debugging"""
        if not self.vector_store:
            print("❌ Vector store not initialized")
            return
        
        docs = self.vector_store.similarity_search_with_score(question, k=k)
        
        print(f"\n🔍 Retrieved {len(docs)} relevant chunks:")
        print("-" * 60)
        
        for i, (doc, score) in enumerate(docs, 1):
            title = doc.metadata.get("title", "Unknown")
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            similarity = 1 - score
            
            print(f"Chunk {i}: {title} (Similarity: {similarity:.3f})")
            print(f"Content: {content_preview}")
            print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="CLI RAG Chatbot")
    parser.add_argument("--topics", nargs="+", 
                       default=["Artificial Intelligence", "Machine Learning", "Natural Language Processing"],
                       help="Wikipedia topics to load")
    parser.add_argument("--max-docs", type=int, default=2,
                       help="Maximum documents per topic")
    parser.add_argument("--api-key", type=str,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--show-chunks", action="store_true",
                       help="Show retrieved chunks for each question")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key required!")
        print("Set OPENAI_API_KEY environment variable or use --api-key flag")
        print("Get your key from: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    # Initialize chatbot
    print("🤖 Initializing RAG Chatbot...")
    chatbot = CLIRAGChatbot(api_key)
    
    # Load documents
    success = chatbot.load_documents(args.topics, args.max_docs)
    if not success:
        sys.exit(1)
    
    # Interactive loop
    print("💬 RAG Chatbot ready! Type 'quit' to exit, 'help' for commands.")
    print("📚 Topics loaded:", ", ".join(args.topics))
    print("=" * 60)
    
    while True:
        try:
            # Get user input
            question = input("\n🤔 Your question: ").strip()
            
            # Handle commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif question.lower() == 'help':
                print("\n📖 Available commands:")
                print("  help       - Show this help message")
                print("  quit/exit  - Exit the chatbot")
                print("  topics     - Show loaded topics")
                print("  examples   - Show example questions")
                print("  Just type your question to get an answer!")
                continue
            elif question.lower() == 'topics':
                print(f"\n📚 Loaded topics: {', '.join(args.topics)}")
                continue
            elif question.lower() == 'examples':
                examples = [
                    "What is artificial intelligence?",
                    "How does machine learning work?",
                    "What are the applications of natural language processing?",
                    "What is the difference between AI and ML?",
                    "How do neural networks function?"
                ]
                print("\n💡 Example questions:")
                for i, ex in enumerate(examples, 1):
                    print(f"  {i}. {ex}")
                continue
            elif not question:
                continue
            
            # Show relevant chunks if requested
            if args.show_chunks:
                chatbot.show_relevant_chunks(question)
            
            # Get answer
            print("\n🤖 Thinking...")
            answer = chatbot.answer_question(question)
            
            # Display answer
            print(f"\n🎯 Answer:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()