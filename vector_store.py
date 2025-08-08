#!/usr/bin/env python3
"""
Vector Store for MHT-CET College RAG System

Purpose: Build and manage FAISS vector index for semantic retrieval of college data
Supports both local sentence-transformers and OpenAI/OpenRouter embeddings
Includes evaluation utilities to measure retrieval performance (F1 score)

Usage:
    # Build index
    python vector_store.py --build --data mht_cet_data.json
    
    # Test retrieval
    python vector_store.py --test "vjti mumbai computer science"

Environment Variables:
    EMBEDDINGS_MODE=local|openai       # Embedding provider
    EMBEDDINGS_MODEL=all-MiniLM-L6-v2  # Model name
    OPENAI_API_KEY=your_key            # If using OpenAI embeddings
    VECTOR_INDEX_PATH=index.faiss      # Index save location
    VECTOR_METADATA_PATH=metadata.json # Metadata save location

Dependencies:
    pip install sentence-transformers faiss-cpu numpy scikit-learn
    # Optional: openai for hosted embeddings
"""

import json
import logging
import os
import pickle
import time
from typing import List, Dict, Any, Optional, Tuple
import argparse
from dataclasses import dataclass

import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support

# Embedding providers
try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBEDDINGS = True
except ImportError:
    LOCAL_EMBEDDINGS = False
    logging.warning("sentence-transformers not available - local embeddings disabled")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class RetrievalResult:
    """Single retrieval result with metadata"""
    record_id: int
    score: float
    college: str
    branch: str
    category: str
    closing_rank: Optional[int]
    city: str
    source_url: str
    full_record: Dict[str, Any]


class VectorStore:
    """FAISS-based vector store for college data semantic retrieval"""
    
    def __init__(self, 
                 embeddings_mode: str = "local",
                 embeddings_model: str = "all-MiniLM-L6-v2",
                 index_path: str = "index.faiss",
                 metadata_path: str = "metadata.json"):
        """
        Initialize vector store
        
        Args:
            embeddings_mode: "local" or "openai"
            embeddings_model: Model name/path
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load record metadata
        """
        self.embeddings_mode = embeddings_mode
        self.embeddings_model = embeddings_model
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # Initialize embedding provider
        self.embedder = None
        self.embedding_dim = None
        self.index = None
        self.metadata = {}  # id -> record mapping
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize embedding model based on configuration"""
        if self.embeddings_mode == "local":
            if not LOCAL_EMBEDDINGS:
                raise ImportError("sentence-transformers required for local embeddings")
            
            self.logger.info(f"Loading local embedding model: {self.embeddings_model}")
            self.embedder = SentenceTransformer(self.embeddings_model)
            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
            
        elif self.embeddings_mode == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package required for OpenAI embeddings")
            
            # Set up OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable required")
            
            openai.api_key = api_key
            self.embedding_dim = 1536  # OpenAI ada-002 dimension
            self.logger.info("Using OpenAI embeddings")
            
        else:
            raise ValueError(f"Unsupported embeddings_mode: {self.embeddings_mode}")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings [n_texts, embedding_dim]
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
            
        if self.embeddings_mode == "local":
            return self._embed_local(texts, batch_size)
        elif self.embeddings_mode == "openai":
            return self._embed_openai(texts, batch_size)

    def _embed_local(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using local sentence-transformers"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedder.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
            
            if len(embeddings) % 10 == 0:
                self.logger.info(f"Processed {len(embeddings) * batch_size} texts")
        
        result = np.vstack(embeddings)
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / (norms + 1e-8)
        
        return result

    def _embed_openai(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = openai.Embedding.create(
                    input=batch,
                    model="text-embedding-ada-002"
                )
                
                batch_embeddings = [item['embedding'] for item in response['data']]
                embeddings.extend(batch_embeddings)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"OpenAI embedding error: {e}")
                # Fallback to zeros
                embeddings.extend([[0.0] * self.embedding_dim] * len(batch))
        
        result = np.array(embeddings)
        
        # Normalize embeddings
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / (norms + 1e-8)
        
        return result

    def _create_searchable_text(self, record: Dict[str, Any]) -> str:
        """Convert college record to searchable text representation"""
        components = []
        
        # College name and location
        if record.get('college'):
            components.append(record['college'])
        if record.get('city'):
            components.append(record['city'])
        if record.get('state'):
            components.append(record['state'])
            
        # Branch information
        if record.get('branch') and record['branch'] != 'Not Specified':
            components.append(record['branch'])
            
        # Category
        if record.get('category'):
            components.append(f"{record['category']} category")
            
        # Ranking information
        if record.get('closing_rank'):
            components.append(f"closing rank {record['closing_rank']}")
            
        # Additional metadata
        if record.get('naac_rating'):
            components.append(f"NAAC {record['naac_rating']}")
            
        # Create comprehensive searchable text
        searchable_text = " ".join(components)
        
        # Add common search variations
        college_name = record.get('college', '').lower()
        if 'vjti' in college_name:
            searchable_text += " veermata jijabai technological institute"
        elif 'iit' in college_name:
            searchable_text += " indian institute of technology"
        elif 'nit' in college_name:
            searchable_text += " national institute of technology"
        
        # Add branch variations
        branch = record.get('branch', '').lower()
        if 'computer' in branch:
            searchable_text += " computer science engineering CSE IT information technology"
        elif 'mechanical' in branch:
            searchable_text += " mechanical engineering MECH"
        elif 'electronics' in branch:
            searchable_text += " electronics communication ECE EXTC"
        elif 'civil' in branch:
            searchable_text += " civil engineering CE"
        elif 'electrical' in branch:
            searchable_text += " electrical engineering EE"
            
        return searchable_text.strip()

    def build_index(self, data_path: str, save_index: bool = True) -> Tuple[faiss.Index, Dict[int, Dict]]:
        """
        Build FAISS index from college data
        
        Args:
            data_path: Path to JSON data file
            save_index: Whether to save index to disk
            
        Returns:
            Tuple of (FAISS index, metadata dict)
        """
        self.logger.info(f"Building index from {data_path}")
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
            
        self.logger.info(f"Loaded {len(records)} records")
        
        # Create searchable texts
        texts = []
        metadata = {}
        
        for idx, record in enumerate(records):
            searchable_text = self._create_searchable_text(record)
            texts.append(searchable_text)
            metadata[idx] = record
            
        # Generate embeddings
        self.logger.info("Generating embeddings...")
        embeddings = self.embed_texts(texts)
        
        # Build FAISS index
        self.logger.info("Building FAISS index...")
        index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        index.add(embeddings.astype(np.float32))
        
        self.index = index
        self.metadata = metadata
        
        # Save if requested
        if save_index:
            self.save_index()
            
        self.logger.info(f"Index built successfully with {index.ntotal} vectors")
        return index, metadata

    def save_index(self):
        """Save FAISS index and metadata to disk"""
        if self.index is None:
            raise ValueError("No index to save")
            
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save metadata
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Index saved to {self.index_path}, metadata to {self.metadata_path}")

    def load_index(self) -> bool:
        """
        Load FAISS index and metadata from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load FAISS index
            if not os.path.exists(self.index_path):
                self.logger.error(f"Index file not found: {self.index_path}")
                return False
                
            self.index = faiss.read_index(self.index_path)
            
            # Load metadata
            if not os.path.exists(self.metadata_path):
                self.logger.error(f"Metadata file not found: {self.metadata_path}")
                return False
                
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                # Convert string keys back to integers
                metadata_raw = json.load(f)
                self.metadata = {int(k): v for k, v in metadata_raw.items()}
                
            self.logger.info(f"Index loaded successfully with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading index: {e}")
            return False

    def retrieve_relevant_colleges(self, 
                                 query: str, 
                                 top_k: int = 5, 
                                 score_threshold: float = 0.2) -> List[RetrievalResult]:
        """
        Retrieve most relevant colleges for query
        
        Args:
            query: Search query string
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of RetrievalResult objects
        """
        if self.index is None:
            if not self.load_index():
                raise ValueError("No index available")
        
        # Generate query embedding
        query_embedding = self.embed_texts([query])
        
        # Search index
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= score_threshold and idx in self.metadata:
                record = self.metadata[idx]
                result = RetrievalResult(
                    record_id=idx,
                    score=float(score),
                    college=record.get('college', 'Unknown'),
                    branch=record.get('branch', 'Unknown'),
                    category=record.get('category', 'Open'),
                    closing_rank=record.get('closing_rank'),
                    city=record.get('city', 'Unknown'),
                    source_url=record.get('source_url', ''),
                    full_record=record
                )
                results.append(result)
                
        return results

    def evaluate_retrieval(self, test_queries_path: str) -> Dict[str, float]:
        """
        Evaluate retrieval performance using test queries
        
        Args:
            test_queries_path: Path to JSON file with test queries and expected results
            
        Returns:
            Dictionary with precision, recall, F1 scores
        """
        if not os.path.exists(test_queries_path):
            self.logger.error(f"Test queries file not found: {test_queries_path}")
            return {}
            
        with open(test_queries_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            
        if 'queries' not in test_data:
            self.logger.error("Test data must contain 'queries' key")
            return {}
            
        all_precisions = []
        all_recalls = []
        all_f1s = []
        
        for test_case in test_data['queries']:
            query = test_case['query']
            expected_colleges = set(test_case.get('expected_colleges', []))
            
            # Retrieve results
            results = self.retrieve_relevant_colleges(query, top_k=10)
            retrieved_colleges = set(result.college.lower() for result in results)
            expected_colleges_lower = set(college.lower() for college in expected_colleges)
            
            # Calculate metrics
            if expected_colleges_lower:
                tp = len(retrieved_colleges.intersection(expected_colleges_lower))
                fp = len(retrieved_colleges - expected_colleges_lower)
                fn = len(expected_colleges_lower - retrieved_colleges)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1s.append(f1)
                
                self.logger.info(f"Query: '{query}' | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}")
        
        # Calculate averages
        avg_precision = np.mean(all_precisions) if all_precisions else 0.0
        avg_recall = np.mean(all_recalls) if all_recalls else 0.0
        avg_f1 = np.mean(all_f1s) if all_f1s else 0.0
        
        metrics = {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'num_queries': len(all_f1s)
        }
        
        self.logger.info(f"Overall Metrics - P: {avg_precision:.3f} | R: {avg_recall:.3f} | F1: {avg_f1:.3f}")
        
        return metrics

    def get_similar_colleges(self, college_name: str, top_k: int = 5) -> List[RetrievalResult]:
        """Find colleges similar to a given college"""
        query = f"{college_name} similar colleges"
        return self.retrieve_relevant_colleges(query, top_k=top_k)

    def search_by_criteria(self, 
                          branch: Optional[str] = None,
                          city: Optional[str] = None,
                          max_rank: Optional[int] = None,
                          category: Optional[str] = None,
                          top_k: int = 10) -> List[RetrievalResult]:
        """Search colleges by specific criteria"""
        query_parts = []
        
        if branch:
            query_parts.append(branch)
        if city:
            query_parts.append(city)
        if category:
            query_parts.append(f"{category} category")
        if max_rank:
            query_parts.append(f"rank under {max_rank}")
            
        query = " ".join(query_parts)
        results = self.retrieve_relevant_colleges(query, top_k=top_k * 2)  # Get more initially
        
        # Apply hard filters
        filtered_results = []
        for result in results:
            if max_rank and result.closing_rank and result.closing_rank > max_rank:
                continue
            if category and result.category.lower() != category.lower():
                continue
            if len(filtered_results) >= top_k:
                break
            filtered_results.append(result)
            
        return filtered_results


def create_sample_test_queries() -> Dict[str, Any]:
    """Create sample test queries for evaluation"""
    return {
        "queries": [
            {
                "query": "vjti mumbai computer science",
                "expected_colleges": ["Veermata Jijabai Technological Institute", "VJTI"]
            },
            {
                "query": "best engineering colleges mumbai",
                "expected_colleges": ["IIT Bombay", "VJTI", "COEP"]
            },
            {
                "query": "mechanical engineering pune",
                "expected_colleges": ["COEP", "VIT Pune"]
            },
            {
                "query": "computer science low cutoff",
                "expected_colleges": ["Generic Engineering College"]
            }
        ]
    }


def main():
    """CLI interface for vector store operations"""
    parser = argparse.ArgumentParser(description="MHT-CET Vector Store Operations")
    parser.add_argument("--build", action="store_true", help="Build new index")
    parser.add_argument("--data", default="mht_cet_data.json", help="Data file path")
    parser.add_argument("--test", type=str, help="Test query for retrieval")
    parser.add_argument("--evaluate", type=str, help="Path to test queries file")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    # Initialize vector store with environment variables
    embeddings_mode = os.getenv("EMBEDDINGS_MODE", "local")
    embeddings_model = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
    index_path = os.getenv("VECTOR_INDEX_PATH", "index.faiss")
    metadata_path = os.getenv("VECTOR_METADATA_PATH", "metadata.json")
    
    vector_store = VectorStore(
        embeddings_mode=embeddings_mode,
        embeddings_model=embeddings_model,
        index_path=index_path,
        metadata_path=metadata_path
    )
    
    if args.build:
        print(f"Building index from {args.data}...")
        vector_store.build_index(args.data)
        print("Index built successfully!")
        
    if args.test:
        print(f"Testing retrieval for: '{args.test}'")
        results = vector_store.retrieve_relevant_colleges(args.test, top_k=args.top_k)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.college}")
            print(f"   Branch: {result.branch}")
            print(f"   City: {result.city}")
            print(f"   Category: {result.category}")
            if result.closing_rank:
                print(f"   Closing Rank: {result.closing_rank}")
            print(f"   Score: {result.score:.3f}")
            
    if args.evaluate:
        print(f"Evaluating retrieval performance...")
        if not os.path.exists(args.evaluate):
            print("Creating sample test queries file...")
            sample_queries = create_sample_test_queries()
            with open(args.evaluate, 'w') as f:
                json.dump(sample_queries, f, indent=2)
            print(f"Sample test queries saved to {args.evaluate}")
            
        metrics = vector_store.evaluate_retrieval(args.evaluate)
        print(f"\nRetrieval Performance:")
        print(f"Precision: {metrics.get('precision', 0):.3f}")
        print(f"Recall: {metrics.get('recall', 0):.3f}")
        print(f"F1 Score: {metrics.get('f1', 0):.3f}")


if __name__ == "__main__":
    main()
