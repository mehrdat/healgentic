"""
Medical knowledge base for document processing and retrieval
"""

import os
from typing import List
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document



class MedicalKnowledgeBase:
    """
    Handles medical textbook processing and retrieval
    
    This class provides:
    1. Medical textbook processing (PDF/EPUB)
    2. Vector database creation and management
    3. Semantic search capabilities
    4. Source attribution
    """
    
    def __init__(self, knowledge_dir: str = None):
        load_dotenv()
        
        # Set directories
        self.knowledge_dir = knowledge_dir or os.getenv(
            "MEDICAL_TEXTBOOKS_DIR", 
            "data/medical_textbooks"
        )
        self.vector_store_dir = os.getenv(
            "VECTOR_STORE_DIR", 
            "data/vector_store"
        )
        
        # Initialize components
        self.vector_store = None
        self._setup_embeddings()
        self._setup_text_splitter()
        
        # Create directories if they don't exist
        Path(self.knowledge_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector_store_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_embeddings(self):
        """Setup embedding model"""
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            print("⚠️  Warning: No Together AI API key found!")
            print("Running in demo mode with limited functionality.")
            self.embeddings = None
            self.demo_mode = True
            return
        
        try:
            # Try to use sentence-transformers for better embeddings
            from .embeddings import SentenceTransformerEmbeddings
            self.embeddings = SentenceTransformerEmbeddings()
            self.demo_mode = False
            print("✅ SentenceTransformer embeddings initialized")
        except ImportError:
            print("⚠️  sentence-transformers not available, using simple embeddings")
            try:
                from .embeddings import TogetherEmbeddings
                self.embeddings = TogetherEmbeddings()
                self.demo_mode = False
                print("✅ Simple embeddings initialized")
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize any embeddings: {e}")
                print("Running in demo mode with limited functionality.")
                self.embeddings = None
                self.demo_mode = True
    
    def _setup_text_splitter(self):
        """Setup text splitter for optimal medical context"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Optimal for medical context
            chunk_overlap=200,  # Preserve context between chunks
            separators=["\n\n", "\n", ". ", " "],
            length_function=len
        )
    
    def process_medical_textbooks(self) -> int:
        """Process all medical textbooks and create vector database"""
        if self.demo_mode:
            print("📚 Demo mode: Creating mock knowledge base...")
            return 100  # Mock document count
        
        print(f"Processing medical textbooks from: {self.knowledge_dir}")
        documents = []
        processed_files = 0

        # Get all TXT files in the directory
        textbook_files = list(Path(self.knowledge_dir).glob("*.txt"))
        
        if not textbook_files:
            print(f"⚠️  No TXT files found in {self.knowledge_dir}")
            return 0
        
        from langchain_community.document_loaders import TextLoader
        for file_path in textbook_files:
            try:
                print(f"Processing: {file_path.name}")
                
                # Load TXT

                loader = TextLoader(str(file_path),autodetect_encoding=True)
                docs = loader.load()
                
                # Add metadata
                if docs:
                    docs[0].metadata.update({
                        'source_book': file_path.name,
                        'book_type': 'medical_textbook',
                        'file_path': str(file_path)
                    })
                
                
                # Split documents into chunks
                split_docs = self.text_splitter.split_documents(docs)
                documents.extend(split_docs)
                processed_files += 1
                
                print(f"  ✅ Processed {file_path.name}: {len(split_docs)} chunks")
                
            except Exception as e:
                print(f"  ❌ Error processing {file_path.name}: {e}")
        
        # Create vector store
        if documents and not self.demo_mode:
            try:
                print("Creating vector database...")
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                
                # Save vector store
                vector_store_path = os.path.join(self.vector_store_dir, "medical_knowledge")
                self.vector_store.save_local(vector_store_path)
                
                print(f"✅ Vector database created with {len(documents)} chunks from {processed_files} books")
                
            except Exception as e:
                print(f"❌ Error creating vector store: {e}")
                return 0
        
        return len(documents)
    
    def load_vector_store(self) -> bool:
        """Load existing vector store"""
        if self.demo_mode:
            return True
            
        try:
            vector_store_path = os.path.join(self.vector_store_dir, "medical_knowledge")
            if os.path.exists(vector_store_path):
                self.vector_store = FAISS.load_local(vector_store_path, self.embeddings)
                print("✅ Vector store loaded successfully")
                return True
            else:
                print("⚠️  Vector store not found. Please run process_medical_textbooks() first.")
                return False
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            return False
    
    def search_medical_knowledge(self, query: str, k: int = 10) -> List[Document]:
        """Search medical knowledge base for relevant information"""
        if self.demo_mode:
            # Return mock documents for demo mode
            mock_doc = Document(
                page_content=f"Demo medical knowledge response for query: {query}",
                metadata={"source_book": "demo_textbook.pdf", "book_type": "medical_textbook"}
            )
            return [mock_doc]
        
        # Load vector store if not already loaded
        if not self.vector_store:
            if not self.load_vector_store():
                return []
        
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Handle different result formats
            relevant_docs = []
            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    # Standard format: (document, score)
                    doc, score = result
                    if score < 0.8:  # Filter by relevance score (lower is better for FAISS)
                        relevant_docs.append(doc)
                elif hasattr(result, 'page_content'):
                    # Document without score
                    relevant_docs.append(result)
                else:
                    # Fallback: treat as document
                    relevant_docs.append(result)
            
            return relevant_docs[:k]
            
        except Exception as e:
            print(f"❌ Error searching knowledge base: {e}")
            return []
    
    def get_statistics(self) -> dict:
        """Get knowledge base statistics"""
        if self.demo_mode:
            return {
                "mode": "demo",
                "total_documents": 100,
                "textbooks": ["demo_textbook.pdf"]
            }
        
        # Count textbooks in directory
        textbook_files = list(Path(self.knowledge_dir).glob("*.txt"))
        
        stats = {
            "mode": "production",
            "textbooks_directory": self.knowledge_dir,
            "available_textbooks": len(textbook_files),
            "textbook_files": [f.name for f in textbook_files],
            "vector_store_exists": self.vector_store is not None
        }
        
        return stats


