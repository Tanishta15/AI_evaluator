import logging
import uuid
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import pandas as pd

# ChromaDB imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

# Milvus imports
try:
    from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

class VectorStorage:
    """
    Abstract base class for vector storage implementations.
    """
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def store_embeddings(self, embeddings_data: List[Dict[str, Any]]) -> bool:
        """Store embeddings with metadata."""
        raise NotImplementedError
        
    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        raise NotImplementedError
        
    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Get embeddings by IDs."""
        raise NotImplementedError

class ChromaDBStorage(VectorStorage):
    """
    ChromaDB implementation for vector storage.
    """
    
    def __init__(self, config=None, collection_name: str = "ai_evaluator_embeddings"):
        super().__init__(config)
        
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not available. Install with: pip install chromadb")
        
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            self.logger.info("🔄 Initializing ChromaDB...")
            
            # Use in-memory client instead of persistent to avoid creating vector_db directory
            self.client = chromadb.Client(
                Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(self.collection_name)
                self.logger.info(f"✅ Connected to existing ChromaDB collection: {self.collection_name}")
            except:
                # Create new collection without specifying embedding function
                # This allows it to accept whatever dimension we provide
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "AI Idea Evaluator embeddings with metadata"}
                )
                self.logger.info(f"✅ Created new ChromaDB collection: {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ChromaDB: {e}")
            raise e
    
    def store_embeddings(self, embeddings_data: List[Dict[str, Any]]) -> bool:
        """Store embeddings with metadata in ChromaDB."""
        try:
            self.logger.info(f"🔄 Storing {len(embeddings_data)} embeddings in ChromaDB...")
            
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for data in embeddings_data:
                # Generate unique ID
                slide_id = str(uuid.uuid4())
                ids.append(slide_id)
                
                # Extract embedding
                embedding = data.get('embedding', [])
                if isinstance(embedding, list):
                    embeddings.append(embedding)
                else:
                    embeddings.append(embedding.tolist())
                
                # Prepare metadata (ChromaDB requires string values)
                metadata = {
                    'slide_number': str(data.get('slide_number', 0)),
                    'title': data.get('title', '')[:500],  # Limit length
                    'total_pages': str(data.get('total_pages', 0)),
                    'images_present': str(data.get('images_present', False)),
                    'primary_criterion': data.get('primary_criterion', ''),
                    'primary_criterion_score': str(data.get('primary_criterion_score', 0)),
                    'embedding_model': data.get('embedding_model', ''),
                    'processing_timestamp': data.get('processing_timestamp', ''),
                    'document_id': data.get('document_id', '')
                }
                metadatas.append(metadata)
                
                # Document text for full-text search
                content = data.get('content', '')
                documents.append(f"{data.get('title', '')} {content}"[:1000])  # Limit length
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            self.logger.info(f"✅ Successfully stored {len(embeddings_data)} embeddings in ChromaDB")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store embeddings in ChromaDB: {e}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5, criteria: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings in ChromaDB."""
        try:
            # Prepare query
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Build where clause for filtering
            where_clause = None
            if criteria:
                where_clause = {"primary_criterion": criteria}
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause,
                include=['embeddings', 'metadatas', 'documents', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'document': results['documents'][0][i],
                    'embedding': results['embeddings'][0][i] if 'embeddings' in results else None
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to search ChromaDB: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                'total_vectors': count,
                'collection_name': self.collection_name,
                'storage_type': 'ChromaDB'
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to get ChromaDB stats: {e}")
            return {}

class MilvusStorage(VectorStorage):
    """
    Milvus implementation for vector storage.
    """
    
    def __init__(self, config=None, collection_name: str = "ai_evaluator_embeddings", 
                 host: str = "localhost", port: str = "19530"):
        super().__init__(config)
        
        if not MILVUS_AVAILABLE:
            raise ImportError("Milvus is not available. Install with: pip install pymilvus")
        
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.collection = None
        
        self._initialize_milvus()
    
    def _initialize_milvus(self):
        """Initialize Milvus connection and collection."""
        try:
            self.logger.info("🔄 Initializing Milvus connection...")
            
            # Connect to Milvus
            connections.connect("default", host=self.host, port=self.port)
            
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.logger.info(f"✅ Connected to existing Milvus collection: {self.collection_name}")
            else:
                self._create_collection()
                self.logger.info(f"✅ Created new Milvus collection: {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Milvus: {e}")
            raise e
    
    def _create_collection(self):
        """Create a new Milvus collection."""
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # Adjust dimension as needed
            FieldSchema(name="slide_number", dtype=DataType.INT64),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="total_pages", dtype=DataType.INT64),
            FieldSchema(name="images_present", dtype=DataType.BOOL),
            FieldSchema(name="primary_criterion", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="primary_criterion_score", dtype=DataType.FLOAT),
        ]
        
        schema = CollectionSchema(fields, "AI Evaluator embeddings collection")
        self.collection = Collection(self.collection_name, schema)
        
        # Create index
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        self.collection.create_index("embedding", index_params)
    
    def store_embeddings(self, embeddings_data: List[Dict[str, Any]]) -> bool:
        """Store embeddings with metadata in Milvus."""
        try:
            self.logger.info(f"🔄 Storing {len(embeddings_data)} embeddings in Milvus...")
            
            # Prepare data for Milvus
            entities = {
                "id": [],
                "embedding": [],
                "slide_number": [],
                "title": [],
                "content": [],
                "total_pages": [],
                "images_present": [],
                "primary_criterion": [],
                "primary_criterion_score": []
            }
            
            for data in embeddings_data:
                entities["id"].append(str(uuid.uuid4()))
                entities["embedding"].append(data.get('embedding', []))
                entities["slide_number"].append(data.get('slide_number', 0))
                entities["title"].append(data.get('title', '')[:500])
                entities["content"].append(data.get('content', '')[:1000])
                entities["total_pages"].append(data.get('total_pages', 0))
                entities["images_present"].append(data.get('images_present', False))
                entities["primary_criterion"].append(data.get('primary_criterion', ''))
                entities["primary_criterion_score"].append(data.get('primary_criterion_score', 0.0))
            
            # Insert data
            self.collection.insert(entities)
            self.collection.flush()
            
            self.logger.info(f"✅ Successfully stored {len(embeddings_data)} embeddings in Milvus")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store embeddings in Milvus: {e}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5, criteria: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Milvus."""
        try:
            # Load collection
            self.collection.load()
            
            # Prepare search parameters
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            # Build expression for filtering
            expr = None
            if criteria:
                expr = f'primary_criterion == "{criteria}"'
            
            # Search
            results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=k,
                expr=expr,
                output_fields=["slide_number", "title", "content", "primary_criterion", "primary_criterion_score"]
            )
            
            # Format results
            formatted_results = []
            for hit in results[0]:
                result = {
                    'id': hit.id,
                    'distance': hit.distance,
                    'slide_number': hit.entity.get('slide_number'),
                    'title': hit.entity.get('title'),
                    'content': hit.entity.get('content'),
                    'primary_criterion': hit.entity.get('primary_criterion'),
                    'primary_criterion_score': hit.entity.get('primary_criterion_score')
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to search Milvus: {e}")
            return []

class VectorStorageManager:
    """
    Manager class that handles both ChromaDB and Milvus storage options.
    """
    
    def __init__(self, storage_type: str = "chromadb", config=None, **kwargs):
        """
        Initialize vector storage manager.
        
        Args:
            storage_type: Either "chromadb" or "milvus"
            config: Configuration object
            **kwargs: Additional arguments for storage initialization
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.storage_type = storage_type.lower()
        
        if self.storage_type == "chromadb":
            if not CHROMADB_AVAILABLE:
                raise ImportError("ChromaDB is not available. Install with: pip install chromadb")
            self.storage = ChromaDBStorage(config, **kwargs)
        elif self.storage_type == "milvus":
            if not MILVUS_AVAILABLE:
                raise ImportError("Milvus is not available. Install with: pip install pymilvus")
            self.storage = MilvusStorage(config, **kwargs)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
        
        self.logger.info(f"✅ Initialized {storage_type} vector storage")
    
    def store_document_embeddings(self, embeddings_data: List[Dict[str, Any]], 
                                document_id: str, filename: str) -> bool:
        """
        Store embeddings for a complete document.
        
        Args:
            embeddings_data: List of embedding data with metadata
            document_id: Unique document identifier
            filename: Original filename
            
        Returns:
            Success status
        """
        try:
            # Add document-level metadata
            enriched_data = []
            for data in embeddings_data:
                enriched_item = data.copy()
                enriched_item.update({
                    'document_id': document_id,
                    'filename': filename,
                    'storage_timestamp': datetime.now().isoformat()
                })
                enriched_data.append(enriched_item)
            
            # Store in vector database
            success = self.storage.store_embeddings(enriched_data)
            
            if success:
                self.logger.info(f"✅ Stored embeddings for document: {filename}")
            else:
                self.logger.error(f"❌ Failed to store embeddings for document: {filename}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error storing document embeddings: {e}")
            return False
    
    def search_by_criteria(self, criteria: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for content by evaluation criteria."""
        # This would need a reference embedding for the criteria
        # For now, return empty list
        return []
    
    def search_similar_content(self, query_embedding: np.ndarray, 
                             criteria: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content."""
        return self.storage.search_similar(query_embedding, k, criteria)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = self.storage.get_collection_stats() if hasattr(self.storage, 'get_collection_stats') else {}
        stats['storage_type'] = self.storage_type
        return stats
