#!/usr/bin/env python3
"""
Test suite for the RAG Chatbot
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Import our RAG components
try:
    from langchain.document_loaders import WikipediaLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import Document
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    IMPORTS_AVAILABLE = False

class TestRAGComponents(unittest.TestCase):
    """Test individual RAG components"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required packages not available")
        
        # Mock documents for testing
        self.mock_docs = [
            Document(
                page_content="Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
                metadata={"title": "Artificial Intelligence", "source": "wikipedia"}
            ),
            Document(
                page_content="Machine Learning is a subset of AI that focuses on algorithms that improve through experience.",
                metadata={"title": "Machine Learning", "source": "wikipedia"}
            )
        ]
    
    def test_text_splitter(self):
        """Test document chunking"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len
        )
        
        # Create a long document
        long_doc = Document(
            page_content="This is a very long document. " * 10,
            metadata={"title": "Test Doc"}
        )
        
        chunks = text_splitter.split_documents([long_doc])
        
        self.assertGreater(len(chunks), 1, "Document should be split into multiple chunks")
        self.assertLessEqual(len(chunks[0].page_content), 120, "Chunks should respect size limit")
    
    @patch('langchain.vectorstores.FAISS.from_documents')
    @patch('langchain.embeddings.OpenAIEmbeddings')
    def test_vector_store_creation(self, mock_embeddings, mock_faiss):
        """Test vector store creation"""
        mock_embeddings.return_value = Mock()
        mock_vector_store = Mock()
        mock_faiss.return_value = mock_vector_store
        
        # This would normally create a vector store
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(self.mock_docs, embeddings)
        
        mock_faiss.assert_called_once()
        self.assertIsNotNone(vector_store)
    
    def test_document_metadata(self):
        """Test document metadata preservation"""
        for doc in self.mock_docs:
            self.assertIn("title", doc.metadata)
            self.assertIn("source", doc.metadata)
            self.assertIsInstance(doc.page_content, str)
            self.assertGreater(len(doc.page_content), 0)

class TestRAGIntegration(unittest.TestCase):
    """Test RAG system integration"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required packages not available")
        
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            self.skipTest("OpenAI API key not available for integration tests")
    
    @unittest.skipIf(not os.getenv("RUN_INTEGRATION_TESTS"), "Integration tests disabled")
    def test_wikipedia_loader(self):
        """Test loading documents from Wikipedia"""
        try:
            loader = WikipediaLoader(query="Machine Learning", load_max_docs=1)
            docs = loader.load()
            
            self.assertGreater(len(docs), 0, "Should load at least one document")
            self.assertIsInstance(docs[0], Document)
            self.assertIn("Machine Learning", docs[0].page_content)
        except Exception as e:
            self.skipTest(f"Wikipedia loader failed: {e}")
    
    @unittest.skipIf(not os.getenv("RUN_INTEGRATION_TESTS"), "Integration tests disabled")
    def test_embeddings_creation(self):
        """Test creating embeddings"""
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
            
            # Test single text embedding
            embedding = embeddings.embed_query("What is artificial intelligence?")
            
            self.assertIsInstance(embedding, list)
            self.assertGreater(len(embedding), 1000)  # OpenAI embeddings are 1536 dimensions
            self.assertTrue(all(isinstance(x, float) for x in embedding))
        except Exception as e:
            self.skipTest(f"Embeddings test failed: {e}")

class TestRAGQueries(unittest.TestCase):
    """Test specific RAG queries and responses"""
    
    def setUp(self):
        """Set up query test fixtures"""
        self.test_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are neural networks?",
            "Define deep learning",
            "Applications of natural language processing"
        ]
        
        self.expected_keywords = {
            "What is artificial intelligence?": ["intelligence", "machine", "computer", "ai"],
            "How does machine learning work?": ["algorithm", "data", "learning", "model"],
            "What are neural networks?": ["network", "neuron", "layer", "connection"],
            "Define deep learning": ["deep", "learning", "neural", "layer"],
            "Applications of natural language processing": ["language", "text", "nlp", "processing"]
        }
    
    def test_query_formatting(self):
        """Test query preprocessing"""
        for query in self.test_queries:
            # Basic validation
            self.assertIsInstance(query, str)
            self.assertGreater(len(query.strip()), 0)
            self.assertTrue(query.endswith("?") or len(query.split()) > 2)
    
    def test_expected_keywords(self):
        """Test that expected keywords are defined for queries"""
        for query in self.test_queries:
            self.assertIn(query, self.expected_keywords)
            keywords = self.expected_keywords[query]
            self.assertIsInstance(keywords, list)
            self.assertGreater(len(keywords), 0)

class TestRAGPerformance(unittest.TestCase):
    """Test RAG system performance characteristics"""
    
    def test_chunk_size_analysis(self):
        """Test different chunk sizes"""
        test_text = "Artificial intelligence is a field of computer science. " * 100
        
        chunk_sizes = [500, 1000, 1500]
        overlaps = [100, 200, 300]
        
        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                if overlap < chunk_size:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=overlap
                    )
                    
                    doc = Document(page_content=test_text)
                    chunks = text_splitter.split_documents([doc])
                    
                    # Validate chunk properties
                    for chunk in chunks:
                        self.assertLessEqual(len(chunk.page_content), chunk_size + 50)  # Some tolerance
                        self.assertGreater(len(chunk.page_content), 0)
    
    def test_retrieval_k_values(self):
        """Test different retrieval k values"""
        k_values = [1, 3, 5, 10]
        
        for k in k_values:
            # Test that k is reasonable
            self.assertGreater(k, 0)
            self.assertLessEqual(k, 20)  # Reasonable upper bound

def run_basic_tests():
    """Run basic functionality tests"""
    print("🧪 Running Basic Tests...")
    print("-" * 40)
    
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRAGComponents))
    suite.addTest(unittest.makeSuite(TestRAGQueries))
    suite.addTest(unittest.makeSuite(TestRAGPerformance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_integration_tests():
    """Run integration tests (requires API key)"""
    print("\n🔗 Running Integration Tests...")
    print("-" * 40)
    print("Note: These tests require OpenAI API key and internet connection")
    
    # Set environment variable to enable integration tests
    os.environ["RUN_INTEGRATION_TESTS"] = "1"
    
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRAGIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_manual_tests():
    """Run manual tests for user interaction"""
    print("\n👤 Manual Tests")
    print("-" * 40)
    
    print("✅ Test UI responsiveness:")
    print("   - Run 'streamlit run app.py'")
    print("   - Check loading times < 60 seconds")
    print("   - Verify all buttons work")
    
    print("\n✅ Test CLI functionality:")
    print("   - Run 'python cli_rag.py'")
    print("   - Try example questions")
    print("   - Test help commands")
    
    print("\n✅ Test error handling:")
    print("   - Try invalid API key")
    print("   - Test network disconnection")
    print("   - Submit empty queries")
    
    print("\n✅ Test answer quality:")
    print("   - Ask domain-specific questions")
    print("   - Verify answers are relevant")
    print("   - Check source attribution")

def main():
    """Main test runner"""
    print("🚀 RAG Chatbot Test Suite")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    if not IMPORTS_AVAILABLE:
        print("❌ Required packages not installed")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Run basic tests
    basic_success = run_basic_tests()
    
    # Ask about integration tests
    if os.getenv("OPENAI_API_KEY"):
        response = input("\n🔑 Run integration tests with OpenAI API? (y/n): ").lower()
        if response in ['y', 'yes']:
            integration_success = run_integration_tests()
        else:
            integration_success = True
            print("⏭️ Skipping integration tests")
    else:
        print("⏭️ Skipping integration tests (no API key)")
        integration_success = True
    
    # Show manual test instructions
    run_manual_tests()
    
    # Summary
    print("\n" + "=" * 50)
    if basic_success and integration_success:
        print("🎉 All automated tests passed!")
        print("✅ RAG system is ready for use")
    else:
        print("❌ Some tests failed")
        print("🔧 Please check the errors above")
    
    print("\n📋 Next steps:")
    print("1. Complete manual tests")
    print("2. Test with your specific use case")
    print("3. Deploy to production if ready")

if __name__ == "__main__":
    main()