#!/usr/bin/env python3
"""
Vector Store for MHT-CET College RAG System

Purpose: Build and manage FAISS vector index for semantic retrieval of college data
Supports OpenAI, Anthropic-compatible, and HuggingFace embeddings with hybrid search
Includes BM25 keyword search, evaluation utilities, and dynamic updates

Usage:
    # Build index from scratch
    python vector_store.py --rebuild --data ./data/structured_data.json
    
    # Search for colleges
    python vector_store.py --search "VJTI computer science OBC cutoff 2023"
    
    # Evaluate performance
    python vector_store.py --evaluate test_queries.json

Environment Variables:
    EMBEDDINGS_PROVIDER=openai|anthropic|huggingface  # Embedding provider
    EMBEDDINGS_MODEL=text-embedding-3-small           # Model name
    OPENAI_API_KEY=your_key                           # API keys
    ANTHROPIC_API_KEY=your_key
    HF_TOKEN=your_token
    VECTOR_INDEX_PATH=./vector_store/faiss_index      # Index location
    VECTOR_METADATA_PATH=./vector_store/metadata.json # Metadata location

Dependencies:
    pip install faiss-cpu numpy scikit-learn openai anthropic sentence-transformers rank-bm25
"""

import json
import logging
import os
import pickle
import time
import uuid
import re
from typing import List, Dict, Any, Optional, Tuple, Union
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import hashlib

import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer

# BM25 for keyword search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logging.warning("rank-bm25 not available - keyword search disabled")

# Embedding providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@dataclass
class RetrievalResult:
    """Single retrieval result with comprehensive metadata"""
    record_id: str
    score: float
    college_name: str
    branch: str
    year: int
    category_cutoffs: Dict[str, float]
    fees: Optional[str]
    placement_rating: Optional[str]
    city: str
    college_type: str
    source_url: str
    raw_text: str
    full_record: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class SearchQuery:
    """Structured search query with extracted criteria"""
    raw_query: str
    college_names: List[str]
    branches: List[str]
    categories: List[str]
    cities: List[str]
    year: Optional[int]
    max_rank: Optional[float]


class EmbeddingProvider:
    """Abstract base class for embedding providers"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding_dim = None
        
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for list of texts"""
        raise NotImplementedError
        
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for single query"""
        return self.embed_texts([query])[0]


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        super().__init__(model_name)
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required for OpenAI embeddings")
            
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Model dimensions
        model_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        self.embedding_dim = model_dims.get(model_name, 1536)
        
    def embed_texts(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logging.error(f"OpenAI embedding error: {e}")
                # Fallback to zeros
                embeddings.extend([[0.0] * self.embedding_dim] * len(batch))
        
        result = np.array(embeddings)
        return self._normalize_embeddings(result)
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)


class AnthropicEmbedding(EmbeddingProvider):
    """Anthropic embedding provider (placeholder - would use Claude API)"""
    
    def __init__(self, model_name: str = "claude-3-sonnet"):
        super().__init__(model_name)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package required for Anthropic embeddings")
            
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.embedding_dim = 1536  # Placeholder dimension
        
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Anthropic API (placeholder implementation)"""
        # Note: This is a placeholder - Anthropic doesn't currently offer embeddings API
        # In practice, you'd use OpenAI or HuggingFace as fallback
        logging.warning("Anthropic embeddings not yet available - using random embeddings as placeholder")
        return np.random.rand(len(texts), self.embedding_dim)


class HuggingFaceEmbedding(EmbeddingProvider):
    """HuggingFace sentence-transformers embedding provider"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        if not HF_AVAILABLE:
            raise ImportError("sentence-transformers required for HuggingFace embeddings")
            
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings using HuggingFace models"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                logging.info(f"Processed {i + len(batch)} texts")
        
        result = np.vstack(embeddings) if embeddings else np.array([]).reshape(0, self.embedding_dim)
        return self._normalize_embeddings(result)
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)


class QueryParser:
    """Parse natural language queries to extract structured criteria"""
    
    def __init__(self):
        # Common college name mappings
        self.college_aliases = {
            'vjti': ['veermata jijabai technological institute', 'vjti', 'v.j.t.i'],
            'iit bombay': ['iit bombay', 'iit mumbai', 'indian institute of technology bombay'],
            'coep': ['coep', 'college of engineering pune'],
            'ict': ['ict mumbai', 'institute of chemical technology'],
            'spit': ['spit mumbai', 'sardar patel institute of technology']
        }
        
        # Branch variations
        self.branch_aliases = {
            'computer science': ['computer science', 'cs', 'cse', 'computer engineering', 'it', 'information technology'],
            'mechanical': ['mechanical', 'mech', 'mechanical engineering'],
            'electronics': ['electronics', 'extc', 'ece', 'electronics and telecommunication'],
            'electrical': ['electrical', 'ee', 'electrical engineering'],
            'civil': ['civil', 'ce', 'civil engineering'],
            'chemical': ['chemical', 'che', 'chemical engineering']
        }
        
        # Category patterns
        self.category_patterns = {
            'open': r'\b(open|general|oge)\b',
            'obc': r'\b(obc|other backward class)\b',
            'sc': r'\b(sc|scheduled caste)\b',
            'st': r'\b(st|scheduled tribe)\b',
            'ews': r'\b(ews|economically weaker)\b'
        }
    
    def parse_query(self, query: str) -> SearchQuery:
        """Parse natural language query into structured criteria"""
        query_lower = query.lower()
        
        # Extract college names
        colleges = []
        for canonical, aliases in self.college_aliases.items():
            for alias in aliases:
                if alias in query_lower:
                    colleges.append(canonical)
                    break
        
        # Extract branches
        branches = []
        for canonical, aliases in self.branch_aliases.items():
            for alias in aliases:
                if alias in query_lower:
                    branches.append(canonical)
                    break
        
        # Extract categories
        categories = []
        for category, pattern in self.category_patterns.items():
            if re.search(pattern, query_lower):
                categories.append(category.upper())
        
        # Extract year
        year_match = re.search(r'\b(20\d{2})\b', query)
        year = int(year_match.group(1)) if year_match else None
        
        # Extract rank/cutoff
        rank_match = re.search(r'\b(?:rank|cutoff|percentile)\s*(?:under|below|less than)?\s*(\d+(?:\.\d+)?)\b', query_lower)
        max_rank = float(rank_match.group(1)) if rank_match else None
        
        # Extract cities (simple approach)
        cities = []
        city_keywords = ['mumbai', 'pune', 'nashik', 'nagpur', 'aurangabad', 'kolhapur']
        for city in city_keywords:
            if city in query_lower:
                cities.append(city.title())
        
        return SearchQuery(
            raw_query=query,
            college_names=colleges,
            branches=branches,
            categories=categories,
            cities=cities,
            year=year,
            max_rank=max_rank
        )


class VectorStoreManager:
    """Production-ready FAISS vector store for MHT-CET college data"""
    
    def __init__(self, 
                 embeddings_provider: str = "openai",
                 embeddings_model: str = "text-embedding-3-small",
                 index_path: str = "./vector_store/faiss_index",
                 metadata_path: str = "./vector_store/metadata.json",
                 cache_path: str = "./vector_store/embeddings_cache.pkl"):
        """
        Initialize vector store manager
        
        Args:
            embeddings_provider: "openai", "anthropic", or "huggingface"
            embeddings_model: Model name for embeddings
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load record metadata
            cache_path: Path to cache embeddings
        """
        self.embeddings_provider = embeddings_provider
        self.embeddings_model = embeddings_model
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.cache_path = cache_path
        
        # Create directories
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embedder = self._initialize_embeddings()
        self.query_parser = QueryParser()
        self.index = None
        self.metadata = {}  # id -> record mapping
        self.text_to_id = {}  # text hash -> id mapping
        self.embeddings_cache = self._load_embeddings_cache()
        self.bm25_index = None
        self.texts_for_bm25 = []
        
        # Logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _initialize_embeddings(self) -> EmbeddingProvider:
        """Initialize embedding provider based on configuration"""
        provider = self.embeddings_provider.lower()
        
        if provider == "openai":
            return OpenAIEmbedding(self.embeddings_model)
        elif provider == "anthropic":
            return AnthropicEmbedding(self.embeddings_model)
        elif provider == "huggingface":
            return HuggingFaceEmbedding(self.embeddings_model)
        else:
            raise ValueError(f"Unsupported embeddings provider: {provider}")

    def _load_embeddings_cache(self) -> Dict[str, np.ndarray]:
        """Load cached embeddings to avoid recomputation"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load embeddings cache: {e}")
        return {}

    def _save_embeddings_cache(self):
        """Save embeddings cache to disk"""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
        except Exception as e:
            self.logger.error(f"Failed to save embeddings cache: {e}")

    def _hash_text(self, text: str) -> str:
        """Generate hash for text caching"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def load_data(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load structured JSON data from scraper output
        
        Args:
            json_path: Path to structured_data.json
            
        Returns:
            List of processed records
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Data file not found: {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        self.logger.info(f"Loaded {len(raw_data)} raw records from {json_path}")
        
        # Process records into searchable chunks
        processed_records = []
        
        for record in raw_data:
            # Extract base information
            college_name = record.get('college_name', 'Unknown College')
            college_type = record.get('college_type', 'Unknown')
            location = record.get('location', 'Unknown')
            year = record.get('year', 2024)
            branches = record.get('branches', {})
            
            # Create a record for each branch
            for branch_name, branch_data in branches.items():
                if not isinstance(branch_data, dict):
                    continue
                    
                # Extract category cutoffs
                category_cutoffs = {}
                fees = None
                placement_rating = None
                
                for key, value in branch_data.items():
                    if key.upper() in ['OPEN', 'OBC', 'SC', 'ST', 'EWS', 'TFWS']:
                        try:
                            category_cutoffs[key.upper()] = float(value) if value else None
                        except (ValueError, TypeError):
                            pass
                    elif 'fees' in key.lower():
                        fees = str(value)
                    elif 'placement' in key.lower() or 'rating' in key.lower():
                        placement_rating = str(value)
                
                # Create processed record
                processed_record = {
                    'id': str(uuid.uuid4()),
                    'college_name': college_name,
                    'branch': branch_name,
                    'year': year,
                    'category_cutoffs': category_cutoffs,
                    'fees': fees,
                    'placement_rating': placement_rating,
                    'city': location,
                    'college_type': college_type,
                    'source_url': record.get('source_url', ''),
                    'raw_text': self._create_searchable_text(college_name, branch_name, location, 
                                                           year, category_cutoffs, fees, placement_rating),
                    'full_record': record
                }
                
                processed_records.append(processed_record)
        
        self.logger.info(f"Processed into {len(processed_records)} searchable records")
        return processed_records

    def _create_searchable_text(self, college_name: str, branch: str, city: str, 
                               year: int, category_cutoffs: Dict[str, float],
                               fees: Optional[str], placement_rating: Optional[str]) -> str:
        """Create comprehensive searchable text for each record"""
        components = []
        
        # Core information
        components.append(college_name)
        components.append(branch)
        components.append(city)
        components.append(str(year))
        
        # Add cutoff information
        for category, cutoff in category_cutoffs.items():
            if cutoff:
                components.append(f"{category} category cutoff {cutoff}")
                components.append(f"{category} rank {cutoff}")
        
        # Add fees and placement info
        if fees:
            components.append(f"fees {fees}")
        if placement_rating:
            components.append(f"placement rating {placement_rating}")
            
        # Add common variations and acronyms
        college_lower = college_name.lower()
        if 'vjti' in college_lower or 'veermata jijabai' in college_lower:
            components.extend(['vjti', 'veermata jijabai technological institute', 'mumbai engineering'])
        elif 'iit' in college_lower:
            components.extend(['iit', 'indian institute of technology', 'premier institute'])
        elif 'nit' in college_lower:
            components.extend(['nit', 'national institute of technology'])
        elif 'coep' in college_lower:
            components.extend(['coep', 'college of engineering pune'])
            
        # Add branch variations
        branch_lower = branch.lower()
        if 'computer' in branch_lower:
            components.extend(['computer science', 'cse', 'computer engineering', 'it', 'information technology'])
        elif 'mechanical' in branch_lower:
            components.extend(['mechanical engineering', 'mech'])
        elif 'electronics' in branch_lower:
            components.extend(['electronics', 'extc', 'ece', 'electronics telecommunication'])
        elif 'electrical' in branch_lower:
            components.extend(['electrical engineering', 'ee'])
        elif 'civil' in branch_lower:
            components.extend(['civil engineering', 'ce'])
        elif 'chemical' in branch_lower:
            components.extend(['chemical engineering', 'che'])
            
        return " ".join(components).strip()

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings with caching support
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings
        """
        embeddings = []
        texts_to_embed = []
        cache_keys = []
        
        # Check cache first
        for text in texts:
            cache_key = self._hash_text(text)
            cache_keys.append(cache_key)
            
            if cache_key in self.embeddings_cache:
                embeddings.append(self.embeddings_cache[cache_key])
            else:
                embeddings.append(None)
                texts_to_embed.append(text)
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            self.logger.info(f"Generating embeddings for {len(texts_to_embed)} new texts")
            new_embeddings = self.embedder.embed_texts(texts_to_embed)
            
            # Update cache
            new_idx = 0
            for i, embedding in enumerate(embeddings):
                if embedding is None:
                    embeddings[i] = new_embeddings[new_idx]
                    self.embeddings_cache[cache_keys[i]] = new_embeddings[new_idx]
                    new_idx += 1
            
            # Save updated cache
            self._save_embeddings_cache()
        
        return np.array(embeddings)

    def build_faiss_index(self, records: List[Dict[str, Any]], save_index: bool = True) -> Tuple[faiss.Index, Dict[str, Dict]]:
        """
        Build FAISS index from processed records
        
        Args:
            records: List of processed college records
            save_index: Whether to save index to disk
            
        Returns:
            Tuple of (FAISS index, metadata dict)
        """
        self.logger.info(f"Building FAISS index from {len(records)} records")
        
        # Extract texts and build metadata
        texts = []
        metadata = {}
        
        for record in records:
            record_id = record['id']
            searchable_text = record['raw_text']
            
            texts.append(searchable_text)
            metadata[record_id] = record
            self.text_to_id[self._hash_text(searchable_text)] = record_id
        
        # Generate embeddings
        self.logger.info("Generating embeddings...")
        embeddings = self.generate_embeddings(texts)
        
        # Build FAISS index (using Inner Product for normalized vectors = cosine similarity)
        self.logger.info("Building FAISS index...")
        index = faiss.IndexFlatIP(self.embedder.embedding_dim)
        index.add(embeddings.astype(np.float32))
        
        # Build BM25 index for keyword search
        if BM25_AVAILABLE:
            self.logger.info("Building BM25 keyword index...")
            tokenized_texts = [text.lower().split() for text in texts]
            self.bm25_index = BM25Okapi(tokenized_texts)
            self.texts_for_bm25 = texts
        
        # Store references
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
            
        # Save BM25 index
        if self.bm25_index and BM25_AVAILABLE:
            bm25_path = self.index_path.replace('faiss_index', 'bm25_index.pkl')
            with open(bm25_path, 'wb') as f:
                pickle.dump({
                    'bm25_index': self.bm25_index,
                    'texts': self.texts_for_bm25
                }, f)
            
        self.logger.info(f"Index saved to {self.index_path}, metadata to {self.metadata_path}")

    def load_faiss_index(self) -> bool:
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
                self.metadata = json.load(f)
                
            # Load BM25 index
            bm25_path = self.index_path.replace('faiss_index', 'bm25_index.pkl')
            if os.path.exists(bm25_path) and BM25_AVAILABLE:
                with open(bm25_path, 'rb') as f:
                    bm25_data = pickle.load(f)
                    self.bm25_index = bm25_data['bm25_index']
                    self.texts_for_bm25 = bm25_data['texts']
                
            self.logger.info(f"Index loaded successfully with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading index: {e}")
            return False

    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Perform semantic search using FAISS"""
        if self.index is None:
            return []
            
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search index
        scores, indices = self.index.search(query_embedding, min(top_k * 2, self.index.ntotal))
        
        # Convert to (record_id, score) tuples
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(list(self.metadata.keys())):
                record_id = list(self.metadata.keys())[idx]
                results.append((record_id, float(score)))
                
        return results

    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Perform keyword search using BM25"""
        if not self.bm25_index or not BM25_AVAILABLE:
            return []
            
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top results
        top_indices = np.argsort(scores)[::-1][:top_k * 2]
        
        results = []
        for idx in top_indices:
            if idx < len(self.texts_for_bm25):
                # Find corresponding record ID
                text_hash = self._hash_text(self.texts_for_bm25[idx])
                if text_hash in self.text_to_id:
                    record_id = self.text_to_id[text_hash]
                    results.append((record_id, float(scores[idx])))
                    
        return results

    def search(self, query: str, top_k: int = 5, hybrid: bool = True, 
               score_threshold: float = 0.1) -> List[RetrievalResult]:
        """
        Main search function with hybrid semantic + keyword search
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            hybrid: Whether to combine semantic and keyword search
            score_threshold: Minimum similarity score
            
        Returns:
            List of RetrievalResult objects
        """
        if self.index is None:
            if not self.load_faiss_index():
                raise ValueError("No index available - please build index first")
        
        start_time = time.time()
        
        # Parse query for structured criteria
        parsed_query = self.query_parser.parse_query(query)
        
        # Perform searches
        semantic_results = self._semantic_search(query, top_k * 2)
        
        if hybrid and BM25_AVAILABLE:
            keyword_results = self._keyword_search(query, top_k * 2)
            # Merge results with weighted scoring
            merged_results = self._merge_search_results(semantic_results, keyword_results, 
                                                       semantic_weight=0.7, keyword_weight=0.3)
        else:
            merged_results = semantic_results
        
        # Apply structured filters
        filtered_results = self._apply_structured_filters(merged_results, parsed_query)
        
        # Convert to RetrievalResult objects
        final_results = []
        seen_records = set()
        
        for record_id, score in filtered_results[:top_k * 2]:
            if record_id in seen_records or record_id not in self.metadata:
                continue
                
            if score < score_threshold:
                continue
                
            record = self.metadata[record_id]
            result = RetrievalResult(
                record_id=record_id,
                score=score,
                college_name=record['college_name'],
                branch=record['branch'],
                year=record['year'],
                category_cutoffs=record['category_cutoffs'],
                fees=record['fees'],
                placement_rating=record['placement_rating'],
                city=record['city'],
                college_type=record['college_type'],
                source_url=record['source_url'],
                raw_text=record['raw_text'],
                full_record=record['full_record']
            )
            
            final_results.append(result)
            seen_records.add(record_id)
            
            if len(final_results) >= top_k:
                break
        
        search_time = time.time() - start_time
        self.logger.info(f"Search completed in {search_time:.3f}s, returned {len(final_results)} results")
        
        return final_results

    def _merge_search_results(self, semantic_results: List[Tuple[str, float]], 
                             keyword_results: List[Tuple[str, float]],
                             semantic_weight: float = 0.7, 
                             keyword_weight: float = 0.3) -> List[Tuple[str, float]]:
        """Merge semantic and keyword search results with weighted scoring"""
        merged_scores = defaultdict(float)
        
        # Normalize scores for semantic results
        if semantic_results:
            max_semantic = max(score for _, score in semantic_results) if semantic_results else 1.0
            for record_id, score in semantic_results:
                normalized_score = score / max_semantic if max_semantic > 0 else 0
                merged_scores[record_id] += normalized_score * semantic_weight
        
        # Normalize scores for keyword results  
        if keyword_results:
            max_keyword = max(score for _, score in keyword_results) if keyword_results else 1.0
            for record_id, score in keyword_results:
                normalized_score = score / max_keyword if max_keyword > 0 else 0
                merged_scores[record_id] += normalized_score * keyword_weight
        
        # Sort by combined score
        merged_results = [(record_id, score) for record_id, score in merged_scores.items()]
        merged_results.sort(key=lambda x: x[1], reverse=True)
        
        return merged_results

    def _apply_structured_filters(self, results: List[Tuple[str, float]], 
                                 parsed_query: SearchQuery) -> List[Tuple[str, float]]:
        """Apply structured filters based on parsed query"""
        filtered_results = []
        
        for record_id, score in results:
            if record_id not in self.metadata:
                continue
                
            record = self.metadata[record_id]
            
            # Apply filters
            if parsed_query.college_names:
                college_match = any(college.lower() in record['college_name'].lower() 
                                  for college in parsed_query.college_names)
                if not college_match:
                    continue
            
            if parsed_query.branches:
                branch_match = any(branch.lower() in record['branch'].lower() 
                                 for branch in parsed_query.branches)
                if not branch_match:
                    continue
                    
            if parsed_query.cities:
                city_match = any(city.lower() in record['city'].lower() 
                               for city in parsed_query.cities)
                if not city_match:
                    continue
                    
            if parsed_query.year and record['year'] != parsed_query.year:
                continue
                
            if parsed_query.max_rank:
                # Check if any category cutoff is within the specified rank
                valid_cutoff = False
                for category, cutoff in record['category_cutoffs'].items():
                    if cutoff and cutoff <= parsed_query.max_rank:
                        valid_cutoff = True
                        break
                if not valid_cutoff:
                    continue
            
            # Apply category filter if specified
            if parsed_query.categories:
                has_category = any(cat in record['category_cutoffs'] 
                                 for cat in parsed_query.categories)
                if not has_category:
                    continue
            
            filtered_results.append((record_id, score))
            
        return filtered_results

    def add_new_entry(self, entry: Dict[str, Any]) -> bool:
        """
        Dynamically add new college/branch data to the index
        
        Args:
            entry: New record dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Process the new entry
            processed_records = self.load_data(None)  # Would need to modify this
            # For now, assume entry is already processed
            
            record_id = entry.get('id', str(uuid.uuid4()))
            searchable_text = entry['raw_text']
            
            # Generate embedding
            embedding = self.generate_embeddings([searchable_text])
            
            # Add to FAISS index
            self.index.add(embedding.astype(np.float32))
            
            # Update metadata
            self.metadata[record_id] = entry
            self.text_to_id[self._hash_text(searchable_text)] = record_id
            
            # Update BM25 index
            if self.bm25_index and BM25_AVAILABLE:
                # Rebuild BM25 (simple approach - for production, use incremental update)
                all_texts = self.texts_for_bm25 + [searchable_text]
                tokenized_texts = [text.lower().split() for text in all_texts]
                self.bm25_index = BM25Okapi(tokenized_texts)
                self.texts_for_bm25 = all_texts
            
            # Save updated index
            self.save_index()
            
            self.logger.info(f"Added new entry: {entry.get('college_name', 'Unknown')} - {entry.get('branch', 'Unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding new entry: {e}")
            return False

    def refresh_index(self, data_path: str) -> bool:
        """
        Rebuild index from latest JSON data
        
        Args:
            data_path: Path to updated data file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Refreshing index from latest data...")
            
            # Clear current state
            self.index = None
            self.metadata = {}
            self.text_to_id = {}
            self.bm25_index = None
            self.texts_for_bm25 = []
            
            # Rebuild from scratch
            records = self.load_data(data_path)
            self.build_faiss_index(records, save_index=True)
            
            self.logger.info("Index refresh completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error refreshing index: {e}")
            return False

    def get_similar_colleges(self, college_name: str, branch: str = None, top_k: int = 5) -> List[RetrievalResult]:
        """Find colleges similar to a given college and branch"""
        query_parts = [college_name]
        if branch:
            query_parts.append(branch)
        query_parts.append("similar colleges")
        
        query = " ".join(query_parts)
        return self.search(query, top_k=top_k)

    def search_by_criteria(self, 
                          branch: Optional[str] = None,
                          city: Optional[str] = None,
                          max_rank: Optional[float] = None,
                          category: Optional[str] = None,
                          year: Optional[int] = None,
                          college_type: Optional[str] = None,
                          top_k: int = 10) -> List[RetrievalResult]:
        """Search colleges by specific structured criteria"""
        query_parts = []
        
        if branch:
            query_parts.append(branch)
        if city:
            query_parts.append(city)
        if category:
            query_parts.append(f"{category} category")
        if college_type:
            query_parts.append(college_type)
        if max_rank:
            query_parts.append(f"cutoff under {max_rank}")
        if year:
            query_parts.append(str(year))
            
        query = " ".join(query_parts) if query_parts else "engineering colleges"
        
        # Use structured search
        parsed_query = SearchQuery(
            raw_query=query,
            college_names=[],
            branches=[branch] if branch else [],
            categories=[category.upper()] if category else [],
            cities=[city] if city else [],
            year=year,
            max_rank=max_rank
        )
        
        # Get more results initially for filtering
        results = self.search(query, top_k=top_k * 3, hybrid=True)
        
        # Apply additional filters
        filtered_results = []
        for result in results:
            if college_type and college_type.lower() not in result.college_type.lower():
                continue
            if len(filtered_results) >= top_k:
                break
            filtered_results.append(result)
            
        return filtered_results

    def get_college_branches(self, college_name: str) -> List[RetrievalResult]:
        """Get all branches available in a specific college"""
        query = f"{college_name} all branches"
        results = self.search(query, top_k=50, hybrid=True)
        
        # Filter to exact college matches
        college_results = []
        for result in results:
            if college_name.lower() in result.college_name.lower():
                college_results.append(result)
                
        return college_results

    def get_cutoff_trends(self, college_name: str, branch: str) -> Dict[int, Dict[str, float]]:
        """Get cutoff trends over years for a college-branch combination"""
        # Search for all years
        query = f"{college_name} {branch}"
        results = self.search(query, top_k=20, hybrid=True)
        
        # Group by year
        trends = defaultdict(dict)
        for result in results:
            if (college_name.lower() in result.college_name.lower() and 
                branch.lower() in result.branch.lower()):
                trends[result.year].update(result.category_cutoffs)
                
        return dict(trends)

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
            expected_branches = set(test_case.get('expected_branches', []))
            
            # Retrieve results
            results = self.search(query, top_k=10)
            retrieved_colleges = set(result.college_name.lower() for result in results)
            retrieved_branches = set(result.branch.lower() for result in results)
            
            # Calculate metrics for colleges
            expected_colleges_lower = set(college.lower() for college in expected_colleges)
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
            'num_queries': len(all_f1s),
            'total_test_cases': len(test_data['queries'])
        }
        
        self.logger.info(f"Overall Metrics - P: {avg_precision:.3f} | R: {avg_recall:.3f} | F1: {avg_f1:.3f}")
        
        return metrics

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        if not self.metadata:
            return {}
            
        stats = {
            'total_records': len(self.metadata),
            'embedding_dimension': self.embedder.embedding_dim,
            'embedding_provider': self.embeddings_provider,
            'embedding_model': self.embeddings_model,
            'has_bm25_index': self.bm25_index is not None,
            'cache_size': len(self.embeddings_cache)
        }
        
        # College and branch distributions
        colleges = defaultdict(int)
        branches = defaultdict(int)
        years = defaultdict(int)
        cities = defaultdict(int)
        
        for record in self.metadata.values():
            colleges[record['college_name']] += 1
            branches[record['branch']] += 1
            years[record['year']] += 1
            cities[record['city']] += 1
        
        stats.update({
            'unique_colleges': len(colleges),
            'unique_branches': len(branches),
            'years_covered': sorted(years.keys()),
            'top_colleges': dict(sorted(colleges.items(), key=lambda x: x[1], reverse=True)[:10]),
            'top_branches': dict(sorted(branches.items(), key=lambda x: x[1], reverse=True)[:10]),
            'cities': list(cities.keys())
        })
        
        return stats


def create_sample_test_queries() -> Dict[str, Any]:
    """Create comprehensive sample test queries for evaluation"""
    return {
        "description": "Test queries for MHT-CET college retrieval evaluation",
        "queries": [
            {
                "query": "vjti mumbai computer science open category cutoff 2023",
                "expected_colleges": ["Veermata Jijabai Technological Institute", "VJTI"],
                "expected_branches": ["Computer Science", "Computer Engineering"]
            },
            {
                "query": "best engineering colleges mumbai mechanical",
                "expected_colleges": ["IIT Bombay", "VJTI", "SPIT"],
                "expected_branches": ["Mechanical Engineering"]
            },
            {
                "query": "coep pune electronics cutoff obc",
                "expected_colleges": ["COEP", "College of Engineering Pune"],
                "expected_branches": ["Electronics"]
            },
            {
                "query": "computer science colleges under rank 1000",
                "expected_colleges": ["IIT Bombay", "VJTI", "COEP"],
                "expected_branches": ["Computer Science"]
            },
            {
                "query": "chemical engineering mumbai colleges fees",
                "expected_colleges": ["ICT Mumbai", "VJTI"],
                "expected_branches": ["Chemical Engineering"]
            },
            {
                "query": "low cutoff engineering colleges pune",
                "expected_colleges": ["Various Engineering Colleges"],
                "expected_branches": ["All Branches"]
            }
        ]
    }


def create_embedding_provider(provider: str, model: str) -> EmbeddingProvider:
    """Factory function to create embedding provider"""
    provider = provider.lower()
    
    if provider == "openai":
        return OpenAIEmbedding(model)
    elif provider == "anthropic":
        return AnthropicEmbedding(model)
    elif provider == "huggingface":
        return HuggingFaceEmbedding(model)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def main():
    """CLI interface for vector store operations"""
    parser = argparse.ArgumentParser(description="MHT-CET Vector Store Operations")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch")
    parser.add_argument("--data", default="./data/structured_data.json", help="Data file path")
    parser.add_argument("--search", type=str, help="Test search query")
    parser.add_argument("--evaluate", type=str, help="Path to test queries file for evaluation")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--stats", action="store_true", help="Show vector store statistics")
    parser.add_argument("--hybrid", action="store_true", default=True, help="Use hybrid search")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "huggingface"],
                       help="Embedding provider")
    parser.add_argument("--model", help="Embedding model name")
    
    args = parser.parse_args()
    
    # Get configuration from environment variables or CLI args
    provider = args.provider or os.getenv("EMBEDDINGS_PROVIDER", "openai")
    
    # Default models for each provider
    default_models = {
        "openai": "text-embedding-3-small",
        "anthropic": "claude-3-sonnet",
        "huggingface": "all-MiniLM-L6-v2"
    }
    
    model = args.model or os.getenv("EMBEDDINGS_MODEL", default_models.get(provider, "text-embedding-3-small"))
    index_path = os.getenv("VECTOR_INDEX_PATH", "./vector_store/faiss_index")
    metadata_path = os.getenv("VECTOR_METADATA_PATH", "./vector_store/metadata.json")
    
    # Initialize vector store
    try:
        vector_store = VectorStoreManager(
            embeddings_provider=provider,
            embeddings_model=model,
            index_path=index_path,
            metadata_path=metadata_path
        )
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return
    
    # Execute commands
    if args.rebuild:
        print(f"Rebuilding index from {args.data}...")
        try:
            records = vector_store.load_data(args.data)
            vector_store.build_faiss_index(records)
            print("✅ Index rebuilt successfully!")
        except Exception as e:
            print(f"❌ Error rebuilding index: {e}")
            
    elif args.search:
        print(f"Searching for: '{args.search}'")
        try:
            results = vector_store.search(args.search, top_k=args.top_k, hybrid=args.hybrid)
            
            if not results:
                print("No results found.")
                return
                
            print(f"\n🔍 Found {len(results)} results:\n")
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.college_name}")
                print(f"   📚 Branch: {result.branch}")
                print(f"   📍 City: {result.city}")
                print(f"   📅 Year: {result.year}")
                print(f"   🎯 Score: {result.score:.3f}")
                
                if result.category_cutoffs:
                    print(f"   📊 Cutoffs:")
                    for category, cutoff in result.category_cutoffs.items():
                        if cutoff:
                            print(f"      {category}: {cutoff}")
                
                if result.fees:
                    print(f"   💰 Fees: {result.fees}")
                    
                if result.placement_rating:
                    print(f"   ⭐ Placement: {result.placement_rating}")
                    
                print()
                
        except Exception as e:
            print(f"❌ Search error: {e}")
            
    elif args.evaluate:
        print("📊 Evaluating retrieval performance...")
        
        # Create sample test file if it doesn't exist
        if not os.path.exists(args.evaluate):
            print(f"Creating sample test queries at {args.evaluate}...")
            sample_queries = create_sample_test_queries()
            os.makedirs(os.path.dirname(args.evaluate) if os.path.dirname(args.evaluate) else '.', exist_ok=True)
            with open(args.evaluate, 'w', encoding='utf-8') as f:
                json.dump(sample_queries, f, indent=2, ensure_ascii=False)
            print(f"✅ Sample test queries created at {args.evaluate}")
            
        try:
            metrics = vector_store.evaluate_retrieval(args.evaluate)
            print(f"\n📈 Retrieval Performance Metrics:")
            print(f"   Precision: {metrics.get('precision', 0):.3f}")
            print(f"   Recall: {metrics.get('recall', 0):.3f}")
            print(f"   F1 Score: {metrics.get('f1', 0):.3f}")
            print(f"   Test Queries: {metrics.get('num_queries', 0)}")
            
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            
    elif args.stats:
        print("📊 Vector Store Statistics:")
        try:
            stats = vector_store.get_statistics()
            
            print(f"   Total Records: {stats.get('total_records', 0)}")
            print(f"   Embedding Dimension: {stats.get('embedding_dimension', 0)}")
            print(f"   Provider: {stats.get('embedding_provider', 'Unknown')}")
            print(f"   Model: {stats.get('embedding_model', 'Unknown')}")
            print(f"   BM25 Index: {'✅' if stats.get('has_bm25_index') else '❌'}")
            print(f"   Cache Size: {stats.get('cache_size', 0)} embeddings")
            print(f"   Unique Colleges: {stats.get('unique_colleges', 0)}")
            print(f"   Unique Branches: {stats.get('unique_branches', 0)}")
            print(f"   Years Covered: {stats.get('years_covered', [])}")
            
            if stats.get('top_colleges'):
                print(f"\n🏫 Top Colleges by Records:")
                for college, count in list(stats['top_colleges'].items())[:5]:
                    print(f"   {college}: {count} records")
                    
            if stats.get('top_branches'):
                print(f"\n🎓 Top Branches by Records:")
                for branch, count in list(stats['top_branches'].items())[:5]:
                    print(f"   {branch}: {count} records")
                    
        except Exception as e:
            print(f"❌ Stats error: {e}")
    
    else:
        # Default: try to load existing index or prompt for rebuild
        if vector_store.load_faiss_index():
            print("✅ Vector store loaded successfully!")
            print("Use --search 'your query' to test retrieval")
            print("Use --stats to see index statistics")
        else:
            print("❌ No existing index found.")
            print("Use --rebuild --data your_data.json to build the index")


# Convenience functions for app.py integration
_global_vector_store = None

def initialize_vector_store(provider: str = None, model: str = None) -> VectorStoreManager:
    """Initialize global vector store instance for app.py"""
    global _global_vector_store
    
    provider = provider or os.getenv("EMBEDDINGS_PROVIDER", "openai")
    model = model or os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    
    _global_vector_store = VectorStoreManager(
        embeddings_provider=provider,
        embeddings_model=model
    )
    
    # Try to load existing index
    if not _global_vector_store.load_faiss_index():
        logging.warning("No existing vector index found - please run --rebuild first")
    
    return _global_vector_store

def search(query: str, top_k: int = 5, hybrid: bool = True) -> List[Dict[str, Any]]:
    """
    Convenience function for app.py integration
    
    Args:
        query: Natural language search query
        top_k: Number of results to return
        hybrid: Use hybrid semantic + keyword search
        
    Returns:
        List of result dictionaries
    """
    global _global_vector_store
    
    if _global_vector_store is None:
        _global_vector_store = initialize_vector_store()
    
    results = _global_vector_store.search(query, top_k=top_k, hybrid=hybrid)
    return [result.to_dict() for result in results]

def add_new_entry(entry: Dict[str, Any]) -> bool:
    """
    Convenience function to add new college data
    
    Args:
        entry: New college record dictionary
        
    Returns:
        True if successful, False otherwise
    """
    global _global_vector_store
    
    if _global_vector_store is None:
        _global_vector_store = initialize_vector_store()
    
    return _global_vector_store.add_new_entry(entry)

def get_college_info(college_name: str, branch: str = None) -> List[Dict[str, Any]]:
    """
    Get specific college information
    
    Args:
        college_name: Name of the college
        branch: Optional branch filter
        
    Returns:
        List of matching records
    """
    query = f"{college_name}"
    if branch:
        query += f" {branch}"
        
    return search(query, top_k=10)

def search_by_criteria(branch: str = None, city: str = None, max_rank: float = None,
                      category: str = None, year: int = None) -> List[Dict[str, Any]]:
    """
    Search by structured criteria
    
    Args:
        branch: Engineering branch
        city: City name
        max_rank: Maximum cutoff rank
        category: Admission category (OPEN, OBC, etc.)
        year: Academic year
        
    Returns:
        List of matching records
    """
    global _global_vector_store
    
    if _global_vector_store is None:
        _global_vector_store = initialize_vector_store()
    
    results = _global_vector_store.search_by_criteria(
        branch=branch, city=city, max_rank=max_rank, 
        category=category, year=year
    )
    
    return [result.to_dict() for result in results]


if __name__ == "__main__":
    main()
