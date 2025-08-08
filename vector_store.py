#!/usr/bin/env python3
"""
Vector Search Module for CET-Mentor v2.0
Handles embedding, indexing, and retrieval of college data using FAISS and sentence-transformers
"""

import json
import os
import pickle
import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

# Vector and ML libraries
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Data class for search results"""
    college: str
    branch: str
    category: str
    closing_rank: int
    fees: str
    city: str
    naac_rating: str
    similarity_score: float
    metadata: Dict[str, Any]

class CollegeVectorStore:
    """
    Vector store for college data with embedding, indexing, and search capabilities
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 vector_store_path: str = 'college_vector_store',
                 use_gpu: bool = False):
        """
        Initialize the vector store
        
        Args:
            model_name: Sentence transformer model name
            vector_store_path: Path to save/load vector store
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.vector_store_path = Path(vector_store_path)
        self.use_gpu = use_gpu
        
        # Initialize components
        self.model = None
        self.index = None
        self.documents = []
        self.metadata = []
        self.embeddings = None
        self.scaler = StandardScaler()
        
        # Create directory if it doesn't exist
        self.vector_store_path.mkdir(exist_ok=True)
        
        # File paths
        self.index_path = self.vector_store_path / 'faiss.index'
        self.metadata_path = self.vector_store_path / 'metadata.pkl'
        self.embeddings_path = self.vector_store_path / 'embeddings.npy'
        self.model_config_path = self.vector_store_path / 'config.json'
        
        logger.info(f"Vector store initialized with model: {model_name}")

    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            if self.model is None:
                device = 'cuda' if self.use_gpu else 'cpu'
                self.model = SentenceTransformer(self.model_name, device=device)
                logger.info(f"Loaded model {self.model_name} on {device}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _create_document_text(self, college_data: Dict) -> str:
        """
        Create searchable text from college data
        
        Args:
            college_data: College information dictionary
            
        Returns:
            Formatted text for embedding
        """
        # Create comprehensive text representation
        text_parts = []
        
        # Core information
        if college_data.get('college'):
            text_parts.append(f"College: {college_data['college']}")
        
        if college_data.get('branch'):
            text_parts.append(f"Branch: {college_data['branch']}")
            
        if college_data.get('category'):
            text_parts.append(f"Category: {college_data['category']}")
            
        if college_data.get('city'):
            text_parts.append(f"City: {college_data['city']}")
            
        if college_data.get('naac_rating'):
            text_parts.append(f"NAAC Rating: {college_data['naac_rating']}")
            
        # Rank information
        if college_data.get('closing_rank'):
            rank = college_data['closing_rank']
            text_parts.append(f"Closing Rank: {rank}")
            
            # Add rank categories for better matching
            if rank <= 1000:
                text_parts.append("Top tier excellent rank")
            elif rank <= 5000:
                text_parts.append("Good rank competitive")
            elif rank <= 15000:
                text_parts.append("Moderate rank accessible")
            else:
                text_parts.append("High rank easy admission")
        
        # Fees information
        if college_data.get('fees'):
            fees = college_data['fees']
            text_parts.append(f"Fees: {fees}")
            
            # Add fee categories
            if any(term in fees.lower() for term in ['free', '0', 'government']):
                text_parts.append("Low cost affordable government")
            elif any(term in fees for term in ['1,00,000', '2,00,000']):
                text_parts.append("Moderate fees reasonable cost")
            else:
                text_parts.append("Higher fees premium institute")
        
        # Additional searchable terms based on branch
        branch_keywords = {
            'computer science': ['CS', 'IT', 'software', 'programming', 'technology'],
            'information technology': ['IT', 'CS', 'software', 'programming', 'technology'],
            'mechanical': ['mech', 'automotive', 'manufacturing', 'design'],
            'electrical': ['EE', 'power', 'electronics', 'circuits'],
            'civil': ['construction', 'infrastructure', 'building', 'structural'],
            'electronics': ['ECE', 'communication', 'circuits', 'embedded'],
            'chemical': ['process', 'chemistry', 'materials', 'industrial']
        }
        
        branch_lower = college_data.get('branch', '').lower()
        for key, keywords in branch_keywords.items():
            if key in branch_lower:
                text_parts.extend(keywords)
                break
        
        return ' '.join(text_parts)

    def load_college_data(self, json_file: str = 'mht_cet_data.json') -> List[Dict]:
        """
        Load college data from JSON file
        
        Args:
            json_file: Path to JSON file containing college data
            
        Returns:
            List of college data dictionaries
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} college records from {json_file}")
            return data
            
        except FileNotFoundError:
            logger.error(f"File {json_file} not found")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {e}")
            raise

    def create_embeddings(self, college_data: List[Dict]) -> np.ndarray:
        """
        Create embeddings for college data
        
        Args:
            college_data: List of college data dictionaries
            
        Returns:
            Array of embeddings
        """
        self._load_model()
        
        logger.info("Creating document texts...")
        documents = []
        metadata = []
        
        for data in college_data:
            try:
                # Create searchable text
                doc_text = self._create_document_text(data)
                documents.append(doc_text)
                
                # Store metadata
                metadata.append({
                    'college': data.get('college', ''),
                    'branch': data.get('branch', ''),
                    'category': data.get('category', ''),
                    'closing_rank': data.get('closing_rank', 0),
                    'fees': data.get('fees', ''),
                    'city': data.get('city', ''),
                    'naac_rating': data.get('naac_rating', ''),
                    'original_data': data
                })
                
            except Exception as e:
                logger.warning(f"Error processing college data: {e}")
                continue
        
        self.documents = documents
        self.metadata = metadata
        
        logger.info(f"Creating embeddings for {len(documents)} documents...")
        
        # Create embeddings in batches to manage memory
        batch_size = 64
        all_embeddings = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_docs,
                convert_to_numpy=True,
                show_progress_bar=True if i == 0 else False,
                normalize_embeddings=True  # Important for cosine similarity
            )
            all_embeddings.append(batch_embeddings)
            
        self.embeddings = np.vstack(all_embeddings)
        logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
        
        return self.embeddings

    def build_index(self, embeddings: np.ndarray):
        """
        Build FAISS index for fast similarity search
        
        Args:
            embeddings: Array of document embeddings
        """
        logger.info("Building FAISS index...")
        
        # Use IndexFlatIP for cosine similarity (with normalized vectors)
        # This gives exact results and is suitable for our use case
        dimension = embeddings.shape[1]
        
        # For larger datasets, consider IndexIVFFlat or IndexHNSWFlat
        if len(embeddings) > 10000:
            # Use IVF index for larger datasets
            nlist = min(100, len(embeddings) // 100)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Train the index
            logger.info("Training IVF index...")
            index.train(embeddings.astype(np.float32))
            index.nprobe = 10  # Number of clusters to search
        else:
            # Use flat index for smaller datasets
            index = faiss.IndexFlatIP(dimension)
        
        # Add embeddings to index
        index.add(embeddings.astype(np.float32))
        
        self.index = index
        logger.info(f"Built FAISS index with {index.ntotal} vectors")

    def save_vector_store(self):
        """Save vector store components to disk"""
        try:
            logger.info("Saving vector store...")
            
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            # Save embeddings
            if self.embeddings is not None:
                np.save(self.embeddings_path, self.embeddings)
            
            # Save configuration
            config = {
                'model_name': self.model_name,
                'num_documents': len(self.metadata),
                'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else None
            }
            
            with open(self.model_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Vector store saved to {self.vector_store_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise

    def load_vector_store(self) -> bool:
        """
        Load vector store from disk
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            if not all(path.exists() for path in [
                self.index_path, self.metadata_path, 
                self.embeddings_path, self.model_config_path
            ]):
                logger.info("Vector store files not found")
                return False
            
            logger.info("Loading vector store...")
            
            # Load configuration
            with open(self.model_config_path, 'r') as f:
                config = json.load(f)
            
            # Verify model compatibility
            if config.get('model_name') != self.model_name:
                logger.warning(f"Model mismatch: {config.get('model_name')} vs {self.model_name}")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Load embeddings
            self.embeddings = np.load(self.embeddings_path)
            
            # Load model
            self._load_model()
            
            logger.info(f"Loaded vector store with {len(self.metadata)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False

    def retrieve_relevant_colleges(self, 
                                 query: str, 
                                 top_k: int = 5,
                                 min_similarity: float = 0.1) -> List[Dict]:
        """
        Retrieve relevant colleges based on query
        
        Args:
            query: Search query
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant college dictionaries with similarity scores
        """
        if self.model is None or self.index is None:
            raise ValueError("Vector store not initialized. Call build_from_data() first.")
        
        # Create query embedding
        query_embedding = self.model.encode(
            [query], 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search using FAISS
        similarities, indices = self.index.search(
            query_embedding.astype(np.float32), 
            min(top_k * 2, len(self.metadata))  # Get more results to filter
        )
        
        results = []
        seen_colleges = set()
        
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx == -1 or similarity < min_similarity:
                continue
                
            metadata = self.metadata[idx]
            
            # Avoid duplicate colleges (prefer higher similarity)
            college_key = f"{metadata['college']}_{metadata['branch']}_{metadata['category']}"
            if college_key in seen_colleges:
                continue
            seen_colleges.add(college_key)
            
            # Create result dictionary
            result = {
                'college': metadata['college'],
                'branch': metadata['branch'],
                'category': metadata['category'],
                'closing_rank': metadata['closing_rank'],
                'fees': metadata['fees'],
                'city': metadata['city'],
                'naac_rating': metadata['naac_rating'],
                'similarity_score': float(similarity),
                'metadata': metadata['original_data']
            }
            
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        logger.info(f"Retrieved {len(results)} relevant colleges for query: '{query}'")
        return results

    def build_from_data(self, json_file: str = 'mht_cet_data.json', force_rebuild: bool = False):
        """
        Build complete vector store from data file
        
        Args:
            json_file: Path to college data JSON file
            force_rebuild: Whether to force rebuild even if saved store exists
        """
        # Try to load existing vector store first
        if not force_rebuild and self.load_vector_store():
            logger.info("Using existing vector store")
            return
        
        logger.info("Building new vector store...")
        
        # Load college data
        college_data = self.load_college_data(json_file)
        
        # Create embeddings
        embeddings = self.create_embeddings(college_data)
        
        # Build FAISS index
        self.build_index(embeddings)
        
        # Save for future use
        self.save_vector_store()
        
        logger.info("Vector store build completed")

    def search_with_filters(self, 
                          query: str, 
                          top_k: int = 5,
                          city_filter: Optional[str] = None,
                          branch_filter: Optional[str] = None,
                          category_filter: Optional[str] = None,
                          max_rank: Optional[int] = None,
                          min_similarity: float = 0.1) -> List[Dict]:
        """
        Search with additional filters
        
        Args:
            query: Search query
            top_k: Number of results
            city_filter: Filter by city
            branch_filter: Filter by branch
            category_filter: Filter by category
            max_rank: Maximum closing rank
            min_similarity: Minimum similarity threshold
            
        Returns:
            Filtered search results
        """
        # Get initial results
        results = self.retrieve_relevant_colleges(query, top_k * 3, min_similarity)
        
        # Apply filters
        filtered_results = []
        
        for result in results:
            # Apply filters
            if city_filter and city_filter.lower() not in result['city'].lower():
                continue
                
            if branch_filter and branch_filter.lower() not in result['branch'].lower():
                continue
                
            if category_filter and result['category'] != category_filter:
                continue
                
            if max_rank and result['closing_rank'] > max_rank:
                continue
            
            filtered_results.append(result)
            
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results

    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if not self.metadata:
            return {}
        
        df = pd.DataFrame(self.metadata)
        
        stats = {
            'total_documents': len(self.metadata),
            'unique_colleges': df['college'].nunique(),
            'unique_branches': df['branch'].nunique(),
            'unique_cities': df['city'].nunique(),
            'categories': df['category'].value_counts().to_dict(),
            'avg_closing_rank': df['closing_rank'].mean(),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else None
        }
        
        return stats

# Convenience functions for easy usage
def build_vector_store(json_file: str = 'mht_cet_data.json', 
                      model_name: str = 'all-MiniLM-L6-v2',
                      force_rebuild: bool = False) -> CollegeVectorStore:
    """
    Build and return a vector store
    
    Args:
        json_file: Path to college data JSON
        model_name: Sentence transformer model name
        force_rebuild: Whether to force rebuild
        
    Returns:
        Initialized CollegeVectorStore
    """
    store = CollegeVectorStore(model_name=model_name)
    store.build_from_data(json_file, force_rebuild=force_rebuild)
    return store

def retrieve_relevant_colleges(query: str, 
                             top_k: int = 5,
                             json_file: str = 'mht_cet_data.json') -> List[Dict]:
    """
    Convenience function to retrieve relevant colleges
    
    Args:
        query: Search query
        top_k: Number of results
        json_file: Path to college data JSON
        
    Returns:
        List of relevant colleges
    """
    store = CollegeVectorStore()
    
    # Try to load existing store, otherwise build new one
    if not store.load_vector_store():
        store.build_from_data(json_file)
    
    return store.retrieve_relevant_colleges(query, top_k)

def main():
    """Example usage"""
    # Build vector store
    store = build_vector_store('mht_cet_data.json', force_rebuild=False)
    
    # Example searches
    test_queries = [
        "computer science college in Mumbai",
        "top engineering college with good placement",
        "mechanical engineering low fees",
        "VJTI electronics branch",
        "college with rank under 5000"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = store.retrieve_relevant_colleges(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['college']} - {result['branch']}")
            print(f"   Rank: {result['closing_rank']}, City: {result['city']}")
            print(f"   Similarity: {result['similarity_score']:.3f}")
    
    # Print statistics
    print("\nVector Store Statistics:")
    stats = store.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
