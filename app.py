#!/usr/bin/env python3
"""
CET-Mentor v2.1 Flask Backend

Purpose: Production-grade Flask API with RAG-powered MHT-CET assistance
Implements double approval workflow: RAG retrieval → LLM validation → streaming response
Supports Claude 3.5 Sonnet, GPT-4, and other OpenRouter models with SSE streaming

Features:
- Enhanced error handling and logging
- Robust session management with SQLite persistence
- Comprehensive rate limiting and security
- Production-ready deployment configuration
- Admin dashboard with analytics
- Real-time streaming responses
- Improved RAG integration with confidence scoring

Usage:
    python app.py

Environment Variables (see .env.example):
    FLASK_SECRET_KEY=your_secret_key
    OPENROUTER_API_KEY=your_openrouter_key
    OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
    EMBEDDINGS_MODE=local|openai
    EMBEDDINGS_MODEL=all-MiniLM-L6-v2
    VECTOR_INDEX_PATH=index.faiss
    VECTOR_METADATA_PATH=metadata.json
    LLM_MODEL=anthropic/claude-3.5-sonnet
    ADMIN_SECRET=your_admin_secret
    RATE_LIMIT_PER_MINUTE=60
    DATABASE_URL=sqlite:///cet_mentor.db

Endpoints:
    GET  /                    - Render chat interface
    POST /suggest            - College suggestions by rank/criteria
    POST /chat               - RAG-powered chat with streaming
    POST /feedback           - Submit user feedback
    GET  /health             - Health check endpoint
    GET  /stats              - Application statistics
    POST /admin/reindex      - Rebuild vector index (admin)
    GET  /admin/logs         - View chat logs (admin)
    GET  /admin/dashboard    - Admin analytics dashboard
"""

import csv
import json
import logging
import os
import sqlite3
import time
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generator, Tuple
import uuid
import re
from contextlib import contextmanager
from dataclasses import dataclass, asdict

from flask import Flask, render_template, request, jsonify, Response, session, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests
from dotenv import load_dotenv

# Import vector store with fallback
try:
    from vector_store import VectorStore, RetrievalResult
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    
    # Fallback classes
    @dataclass
    class RetrievalResult:
        record_id: int
        college: str
        branch: str
        city: str
        category: str
        closing_rank: Optional[int]
        score: float
        source_url: str
        full_record: Dict[str, Any]
    
    class VectorStore:
        def __init__(self, *args, **kwargs):
            pass
        
        def load_index(self) -> bool:
            return False
        
        def retrieve_relevant_colleges(self, query: str, top_k: int = 5, score_threshold: float = 0.3) -> List[RetrievalResult]:
            return []
        
        def build_index(self, data_file: str, save_index: bool = True):
            pass

# Load environment variables
load_dotenv()

# Configure comprehensive logging
def setup_logging():
    """Setup comprehensive logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Setup separate loggers for different components
    access_logger = logging.getLogger('access')
    access_handler = logging.FileHandler('logs/access.log')
    access_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    access_logger.addHandler(access_handler)
    access_logger.setLevel(logging.INFO)
    
    error_logger = logging.getLogger('errors')
    error_handler = logging.FileHandler('logs/errors.log')
    error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    error_logger.addHandler(error_handler)
    error_logger.setLevel(logging.ERROR)
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Application Configuration
@dataclass
class AppConfig:
    """Centralized application configuration"""
    # Flask settings
    SECRET_KEY: str = os.getenv('FLASK_SECRET_KEY', 'dev-secret-change-in-production')
    DEBUG: bool = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    PORT: int = int(os.getenv('PORT', 5000))
    
    # API settings
    OPENROUTER_API_KEY: str = os.getenv('OPENROUTER_API_KEY', '')
    OPENROUTER_BASE_URL: str = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
    LLM_MODEL: str = os.getenv('LLM_MODEL', 'anthropic/claude-3.5-sonnet')
    FALLBACK_MODEL: str = os.getenv('FALLBACK_MODEL', 'openai/gpt-4o')
    
    # Vector store settings
    EMBEDDINGS_MODE: str = os.getenv('EMBEDDINGS_MODE', 'local')
    EMBEDDINGS_MODEL: str = os.getenv('EMBEDDINGS_MODEL', 'all-MiniLM-L6-v2')
    VECTOR_INDEX_PATH: str = os.getenv('VECTOR_INDEX_PATH', 'data/index.faiss')
    VECTOR_METADATA_PATH: str = os.getenv('VECTOR_METADATA_PATH', 'data/metadata.json')
    
    # RAG settings
    RAG_SCORE_THRESHOLD: float = float(os.getenv('RAG_SCORE_THRESHOLD', '0.3'))
    RAG_MIN_RESULTS: int = int(os.getenv('RAG_MIN_RESULTS', '2'))
    RAG_MAX_CONTEXT_LENGTH: int = int(os.getenv('RAG_MAX_CONTEXT_LENGTH', '4000'))
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_PER_MINUTE', '30'))
    RATE_LIMIT_PER_HOUR: int = int(os.getenv('RATE_LIMIT_PER_HOUR', '200'))
    
    # Admin settings
    ADMIN_SECRET: str = os.getenv('ADMIN_SECRET', 'admin-secret-change-me')
    
    # Database
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'sqlite:///data/cet_mentor.db')
    
    # Session settings
    SESSION_TIMEOUT_HOURS: int = int(os.getenv('SESSION_TIMEOUT_HOURS', '24'))
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        if not self.OPENROUTER_API_KEY:
            issues.append("OPENROUTER_API_KEY not configured")
        
        if self.SECRET_KEY == 'dev-secret-change-in-production' and not self.DEBUG:
            issues.append("Default secret key in production")
        
        if self.ADMIN_SECRET == 'admin-secret-change-me':
            issues.append("Default admin secret")
        
        return issues

config = AppConfig()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# Enable CORS with proper configuration
CORS(app, 
     origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Add your frontend URLs
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)

# Initialize rate limiter with Redis fallback
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=[f"{config.RATE_LIMIT_PER_HOUR} per hour"],
    storage_uri=os.getenv('REDIS_URL', 'memory://')
)

# Global variables
vector_store: Optional[VectorStore] = None
db_lock = threading.Lock()

# Enhanced system prompt with better instructions
SYSTEM_PROMPT = """You are CET-Mentor, an expert assistant for MHT-CET (Maharashtra Common Entrance Test) college admissions in Maharashtra, India.

CRITICAL RESPONSE GUIDELINES:

1. VERIFIED CONTEXT USAGE:
   - If VERIFIED CONTEXT is provided, prioritize this official data
   - Cite specific colleges, ranks, and sources from the context
   - Always mention data limitations and encourage verification

2. NO VERIFIED CONTEXT:
   - Clearly state "I don't have specific cutoff data for this query"
   - Provide general MHT-CET guidance and processes
   - Suggest checking official sources (DTE Maharashtra, college websites)

3. RANK INTERPRETATION:
   - Remember: LOWER rank number = BETTER performance
   - Rank 1000 is better than rank 5000
   - Explain this clearly when discussing cutoffs

4. RESPONSE STRUCTURE:
   - Start with direct answer to the specific question
   - Provide supporting information from verified context if available
   - Include helpful general guidance
   - End with encouragement and next steps

5. TONE AND APPROACH:
   - Be helpful, accurate, and encouraging
   - Acknowledge data limitations honestly
   - Provide actionable guidance
   - Use simple, clear language

You can discuss general topics like study strategies, career guidance, and admission processes even without specific data."""

# Database Management
@contextmanager
def get_db_connection():
    """Thread-safe database connection context manager"""
    conn = None
    try:
        # Ensure data directory exists
        db_path = config.DATABASE_URL.replace('sqlite:///', '')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        with db_lock:
            conn = sqlite3.connect(db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def init_database():
    """Initialize database with all required tables"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    client_ip TEXT,
                    user_agent TEXT
                )
            ''')
            
            # Chat logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    query TEXT NOT NULL,
                    response_length INTEGER,
                    retrieval_ids TEXT,
                    llm_decision TEXT,
                    has_verified_context BOOLEAN,
                    response_time_ms INTEGER,
                    client_ip TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            # Feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rating TEXT NOT NULL CHECK (rating IN ('positive', 'negative')),
                    comment TEXT,
                    query_context TEXT,
                    client_ip TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            # Application stats table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS app_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE DEFAULT CURRENT_DATE,
                    metric_name TEXT NOT NULL,
                    metric_value INTEGER DEFAULT 0,
                    UNIQUE(date, metric_name)
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_logs_session ON chat_logs(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_logs_timestamp ON chat_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_app_stats_date ON app_stats(date)')
            
            conn.commit()
            logger.info("Database initialized successfully")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

def cleanup_old_sessions():
    """Clean up expired sessions and logs"""
    try:
        cutoff_time = datetime.now() - timedelta(hours=config.SESSION_TIMEOUT_HOURS)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Clean old sessions
            cursor.execute("DELETE FROM sessions WHERE last_activity < ?", (cutoff_time,))
            sessions_deleted = cursor.rowcount
            
            # Clean old chat logs (keep last 30 days)
            old_logs_cutoff = datetime.now() - timedelta(days=30)
            cursor.execute("DELETE FROM chat_logs WHERE timestamp < ?", (old_logs_cutoff,))
            logs_deleted = cursor.rowcount
            
            conn.commit()
            
            if sessions_deleted > 0 or logs_deleted > 0:
                logger.info(f"Cleanup: {sessions_deleted} sessions, {logs_deleted} logs removed")
                
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

# Session Management
def get_session_id() -> str:
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        
        # Store session in database
        user_data = {
            'conversation_history': [],
            'preferences': {},
            'created_at': datetime.now().isoformat(),
            'total_queries': 0
        }
        
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO sessions (session_id, user_data, client_ip, user_agent)
                    VALUES (?, ?, ?, ?)
                ''', (
                    session['session_id'],
                    json.dumps(user_data),
                    get_remote_address(),
                    request.headers.get('User-Agent', '')[:500]
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error creating session: {e}")
    
    return session['session_id']

def get_user_session(session_id: str) -> Dict[str, Any]:
    """Retrieve user session data"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_data FROM sessions WHERE session_id = ? AND last_activity > ?",
                (session_id, datetime.now() - timedelta(hours=config.SESSION_TIMEOUT_HOURS))
            )
            result = cursor.fetchone()
            
            if result:
                return json.loads(result['user_data'])
            else:
                # Session expired or doesn't exist
                return {
                    'conversation_history': [],
                    'preferences': {},
                    'created_at': datetime.now().isoformat(),
                    'total_queries': 0
                }
                
    except Exception as e:
        logger.error(f"Error retrieving session {session_id}: {e}")
        return {'conversation_history': [], 'preferences': {}, 'total_queries': 0}

def save_user_session(session_id: str, user_data: Dict[str, Any]):
    """Save user session data"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions 
                SET user_data = ?, last_activity = CURRENT_TIMESTAMP 
                WHERE session_id = ?
            ''', (json.dumps(user_data), session_id))
            
            if cursor.rowcount == 0:
                # Session doesn't exist, create it
                cursor.execute('''
                    INSERT INTO sessions (session_id, user_data, client_ip, user_agent)
                    VALUES (?, ?, ?, ?)
                ''', (
                    session_id,
                    json.dumps(user_data),
                    get_remote_address(),
                    request.headers.get('User-Agent', '')[:500]
                ))
            
            conn.commit()
            
    except Exception as e:
        logger.error(f"Error saving session {session_id}: {e}")

# Enhanced Vector Store Management
def initialize_vector_store() -> bool:
    """Initialize vector store with comprehensive error handling"""
    global vector_store
    
    if not VECTOR_STORE_AVAILABLE:
        logger.warning("Vector store module not available - RAG functionality disabled")
        return False
    
    try:
        # Ensure data directory exists
        os.makedirs(os.path.dirname(config.VECTOR_INDEX_PATH), exist_ok=True)
        
        vector_store = VectorStore(
            embeddings_mode=config.EMBEDDINGS_MODE,
            embeddings_model=config.EMBEDDINGS_MODEL,
            index_path=config.VECTOR_INDEX_PATH,
            metadata_path=config.VECTOR_METADATA_PATH
        )
        
        if vector_store.load_index():
            logger.info("Vector store initialized successfully")
            return True
        else:
            logger.warning("Could not load vector index - RAG functionality limited")
            return False
            
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        vector_store = None
        return False

# Enhanced LLM Integration
class LLMClient:
    """Enhanced LLM client with better error handling and monitoring"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = None
    
    def call_api(self, messages: List[Dict], model: str = None, stream: bool = True, 
                 temperature: float = 0.7, max_tokens: int = 1500) -> requests.Response:
        """Call LLM API with comprehensive error handling"""
        if not self.config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not configured")
        
        model = model or self.config.LLM_MODEL
        self.request_count += 1
        self.last_request_time = datetime.now()
        
        headers = {
            'Authorization': f"Bearer {self.config.OPENROUTER_API_KEY}",
            'Content-Type': 'application/json',
            'HTTP-Referer': request.headers.get('Referer', 'https://cet-mentor.com'),
            'X-Title': 'CET-Mentor v2.1'
        }
        
        payload = {
            'model': model,
            'messages': messages,
            'stream': stream,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': 0.9
        }
        
        try:
            response = requests.post(
                f"{self.config.OPENROUTER_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                stream=stream,
                timeout=60
            )
            
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                # Rate limit - try fallback
                if model != self.config.FALLBACK_MODEL:
                    logger.warning(f"Rate limit hit for {model}, trying fallback")
                    return self.call_api(messages, self.config.FALLBACK_MODEL, stream, temperature, max_tokens)
                else:
                    raise requests.exceptions.RequestException("Rate limit exceeded on all models")
            else:
                response.raise_for_status()
                
        except requests.exceptions.RequestException as e:
            self.error_count += 1
            logger.error(f"LLM API error with {model}: {e}")
            
            # Try fallback model if primary fails
            if model != self.config.FALLBACK_MODEL:
                logger.info(f"Trying fallback model: {self.config.FALLBACK_MODEL}")
                return self.call_api(messages, self.config.FALLBACK_MODEL, stream, temperature, max_tokens)
            
            raise
    
    def stream_response(self, messages: List[Dict]) -> Generator[str, None, None]:
        """Stream LLM response with proper error handling"""
        try:
            response = self.call_api(messages, stream=True)
            
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith('data: '):
                    try:
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            break
                        
                        json_data = json.loads(data)
                        delta = json_data.get('choices', [{}])[0].get('delta', {})
                        content = delta.get('content', '')
                        
                        if content:
                            yield content
                            
                    except json.JSONDecodeError:
                        continue
                    except KeyError:
                        continue
                        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\n\n[Error: Could not complete response - {str(e)}]"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM client statistics"""
        return {
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'success_rate': (self.request_count - self.error_count) / max(self.request_count, 1),
            'last_request': self.last_request_time.isoformat() if self.last_request_time else None
        }

llm_client = LLMClient(config)

# Enhanced Query Processing
class QueryProcessor:
    """Enhanced query processing with better intent detection"""
    
    @staticmethod
    def detect_intent(query: str) -> Dict[str, Any]:
        """Enhanced intent detection with better patterns"""
        query_lower = query.lower().strip()
        
        # Direct rank patterns
        rank_patterns = [
            r'^\s*(\d{1,7})\s*$',  # Pure number
            r'my\s+rank\s+is\s+(\d{4,7})',
            r'rank\s*:?\s*(\d{4,7})',
            r'(\d{4,7})\s+rank'
        ]
        
        extracted_data = {
            'type': 'general_chat',
            'rank': None,
            'category': None,
            'branch': None,
            'city': None,
            'confidence': 0.0
        }
        
        # Extract rank
        for pattern in rank_patterns:
            match = re.search(pattern, query_lower)
            if match:
                rank = int(match.group(1))
                if 1 <= rank <= 200000:  # Valid CET rank range
                    extracted_data['rank'] = rank
                    extracted_data['type'] = 'rank_suggestion'
                    extracted_data['confidence'] = 0.8
                break
        
        # Extract category with better patterns
        category_patterns = {
            'obc': [r'\bobc\b', r'other\s+backward'],
            'sc': [r'\bsc\b', r'scheduled\s+caste'],
            'st': [r'\bst\b', r'scheduled\s+tribe'],
            'open': [r'\bopen\b', r'\bgeneral\b', r'unreserved']
        }
        
        for category, patterns in category_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                extracted_data['category'] = category
                break
        
        # Extract branch with comprehensive patterns
        branch_patterns = {
            'computer science': [r'computer', r'\bcse\b', r'\bcs\b', r'information\s+technology', r'\bit\b'],
            'mechanical': [r'mechanical', r'\bmech\b'],
            'electronics': [r'electronics', r'\bextc\b', r'\bece\b', r'communication'],
            'civil': [r'\bcivil\b'],
            'electrical': [r'electrical', r'\bee\b', r'power'],
            'chemical': [r'chemical'],
            'biotechnology': [r'biotech', r'biotechnology'],
            'automobile': [r'automobile', r'automotive']
        }
        
        for branch, patterns in branch_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                extracted_data['branch'] = branch
                break
        
        # Extract city
        cities = ['mumbai', 'pune', 'nagpur', 'nashik', 'aurangabad', 'kolhapur', 'sangli', 'solapur', 'amravati']
        for city in cities:
            if city in query_lower:
                extracted_data['city'] = city
                break
        
        return extracted_data
    
    @staticmethod
    def build_search_query(intent: Dict[str, Any], original_query: str) -> str:
        """Build optimized search query for vector retrieval"""
        query_parts = []
        
        if intent.get('rank'):
            query_parts.append(f"rank {intent['rank']}")
        
        if intent.get('branch'):
            query_parts.append(intent['branch'])
        
        if intent.get('city'):
            query_parts.append(intent['city'])
        
        if intent.get('category') and intent['category'] != 'open':
            query_parts.append(f"{intent['category']} category")
        
        # If no specific components extracted, use original query
        if not query_parts:
            return original_query
        
        return " ".join(query_parts)

query_processor = QueryProcessor()

# Enhanced RAG Processing
def create_verified_context_block(results: List[RetrievalResult]) -> str:
    """Create enhanced VERIFIED CONTEXT block from retrieval results"""
    if not results:
        return ""
    
    context_lines = ["=== VERIFIED CONTEXT (Official MHT-CET Data) ==="]
    context_lines.append(f"Retrieved {len(results)} relevant records:\n")
    
    for i, result in enumerate(results, 1):
        context_lines.append(f"Record {i}:")
        context_lines.append(f"  College: {result.college}")
        context_lines.append(f"  Branch: {result.branch}")
        context_lines.append(f"  City: {result.city}")
        context_lines.append(f"  Category: {result.category}")
        
        if result.closing_rank:
            context_lines.append(f"  Closing Rank: {result.closing_rank:,}")
        
        # Add additional metadata if available
        if result.full_record.get('fees'):
            context_lines.append(f"  Fees: {result.full_record['fees']}")
        
        if result.full_record.get('naac_rating'):
            context_lines.append(f"  NAAC Rating: {result.full_record['naac_rating']}")
        
        if result.full_record.get('nirf_ranking'):
            context_lines.append(f"  NIRF Ranking: {result.full_record['nirf_ranking']}")
        
        context_lines.append(f"  Source: {result.source_url}")
        context_lines.append(f"  Relevance Score: {result.score:.3f}")
        context_lines.append("")
    
    context_lines.append("=== END VERIFIED CONTEXT ===\n")
    
    # Ensure context doesn't exceed max length
    context_text = "\n".join(context_lines)
    if len(context_text) > config.RAG_MAX_CONTEXT_LENGTH:
        # Truncate and add notice
        context_text = context_text[:config.RAG_MAX_CONTEXT_LENGTH]
        context_text += "\n\n[Context truncated due to length limit]"
    
    return context_text

def calculate_retrieval_confidence(results: List[RetrievalResult], intent: Dict[str, Any]) -> float:
    """Calculate enhanced confidence score from RAG results"""
    if not results:
        return 0.0
    
    # Base confidence from similarity scores
    scores = [result.score for result in results]
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    
    # Boost confidence based on intent matching
    intent_boost = 0.0
    if intent.get('rank'):
        # Check if any results have ranks close to query rank
        query_rank = intent['rank']
        for result in results:
            if result.closing_rank:
                rank_diff = abs(result.closing_rank - query_rank)
                if rank_diff < 5000:  # Within 5k ranks
                    intent_boost += 0.1
    
    # Final confidence calculation
    confidence = min((avg_score * 0.7 + max_score * 0.3 + intent_boost), 1.0)
    
    logger.info(f"Retrieval confidence: {confidence:.3f} (avg: {avg_score:.3f}, max: {max_score:.3f}, boost: {intent_boost:.3f})")
    return confidence

# Logging Functions
def log_chat_interaction(session_id: str, query: str, response: str, 
                        retrieval_ids: List[int], llm_decision: Dict[str, Any],
                        response_time_ms: int):
    """Log chat interaction to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO chat_logs 
                (session_id, query, response_length, retrieval_ids, llm_decision, 
                 has_verified_context, response_time_ms, client_ip)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                query,
                len(response),
                json.dumps(retrieval_ids),
                json.dumps(llm_decision),
                len(retrieval_ids) > 0,
                response_time_ms,
                get_remote_address()
            ))
            conn.commit()
            
    except Exception as e:
        logger.error(f"Error logging chat interaction: {e}")

def update_daily_stats(metric_name: str, increment: int = 1):
    """Update daily statistics"""
    try:
        today = datetime.now().date()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO app_stats (date, metric_name, metric_value)
                VALUES (?, ?, ?)
                ON CONFLICT(date, metric_name) 
                DO UPDATE SET metric_value = metric_value + ?
            ''', (today, metric_name, increment, increment))
            conn.commit()
            
    except Exception as e:
        logger.error(f"Error updating stats: {e}")

# Flask Routes
@app.before_request
def before_request():
    """Execute before each request"""
    g.start_time = time.time()
    
    # Log access
    access_logger = logging.getLogger('access')
    access_logger.info(f"{request.method} {request.path} - {get_remote_address()}")

@app.after_request
def after_request(response):
    """Execute after each request"""
    # Calculate response time
    if hasattr(g, 'start_time'):
        response_time = (time.time() - g.start_time) * 1000
        response.headers['X-Response-Time'] = f"{response_time:.2f}ms"
    
    # Add security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    return response

@app.route('/')
def index():
    """Render main chat interface"""
    try:
        # Check if we have a template file
        return render_template('index.html')
    except Exception:
        # Fallback to simple HTML response
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>CET-Mentor v2.1</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
                .status { color: #666; margin-bottom: 20px; }
                .success { color: #28a745; }
                .warning { color: #ffc107; }
                .error { color: #dc3545; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎓 CET-Mentor v2.1</h1>
                <p class="status">Production-ready Flask backend is running!</p>
                
                <h3>API Endpoints:</h3>
                <ul>
                    <li><strong>POST /chat</strong> - Main chatbot interface</li>
                    <li><strong>POST /suggest</strong> - College suggestions by rank</li>
                    <li><strong>POST /feedback</strong> - Submit feedback</li>
                    <li><strong>GET /health</strong> - Health check</li>
                    <li><strong>GET /stats</strong> - Application statistics</li>
                </ul>
                
                <h3>System Status:</h3>
                <p class="status">Vector Store: <span class="{'success' if vector_store else 'warning'}">{'Available' if vector_store else 'Limited'}</span></p>
                <p class="status">LLM API: <span class="{'success' if config.OPENROUTER_API_KEY else 'error'}">{'Configured' if config.OPENROUTER_API_KEY else 'Not Configured'}</span></p>
                
                <p><em>Use your frontend application to interact with the chatbot, or send POST requests to the API endpoints.</em></p>
            </div>
        </body>
        </html>
        """

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check database connectivity
        db_status = "healthy"
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
        except Exception:
            db_status = "unhealthy"
        
        # Check vector store
        vector_status = "available" if vector_store else "unavailable"
        
        # Check LLM API configuration
        llm_status = "configured" if config.OPENROUTER_API_KEY else "not_configured"
        
        # Get system stats
        stats = llm_client.get_stats()
        
        health_data = {
            'status': 'healthy' if db_status == 'healthy' else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'version': '2.1',
            'services': {
                'database': db_status,
                'vector_store': vector_status,
                'llm_api': llm_status
            },
            'configuration': {
                'model': config.LLM_MODEL,
                'fallback_model': config.FALLBACK_MODEL,
                'embeddings_mode': config.EMBEDDINGS_MODE,
                'debug_mode': config.DEBUG
            },
            'stats': stats
        }
        
        status_code = 200 if health_data['status'] == 'healthy' else 503
        return jsonify(health_data), status_code
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/suggest', methods=['POST'])
@limiter.limit("30 per minute")
def suggest_colleges():
    """
    Enhanced college suggestion endpoint with better filtering and ranking
    
    Expected JSON payload:
    {
        "rank": 12345,
        "category": "open|obc|sc|st", 
        "branch": "computer science",
        "city": "mumbai"
    }
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data or 'rank' not in data:
            return jsonify({'error': 'Rank is required'}), 400
        
        # Validate and extract parameters
        try:
            rank = int(data['rank'])
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid rank format'}), 400
        
        category = data.get('category', 'open').lower()
        branch = data.get('branch', '').lower().strip()
        city = data.get('city', '').lower().strip()
        
        # Validate inputs
        if rank < 1 or rank > 200000:
            return jsonify({'error': 'Rank must be between 1 and 200,000'}), 400
        
        valid_categories = ['open', 'obc', 'sc', 'st']
        if category not in valid_categories:
            return jsonify({'error': f'Category must be one of: {", ".join(valid_categories)}'}), 400
        
        session_id = get_session_id()
        
        # Build search query
        query_parts = []
        if branch:
            query_parts.append(branch)
        if city:
            query_parts.append(city)
        if category != 'open':
            query_parts.append(f"{category} category")
        query_parts.append(f"rank {rank}")
        
        search_query = " ".join(query_parts)
        
        # Retrieve suggestions from vector store
        safe_options = []
        ambitious_options = []
        backup_options = []
        
        if vector_store:
            try:
                results = vector_store.retrieve_relevant_colleges(
                    search_query, 
                    top_k=30,
                    score_threshold=0.1
                )
                
                for result in results:
                    # Skip if doesn't match category
                    if category != 'open' and result.category.lower() != category:
                        continue
                    
                    # Skip if doesn't match branch (if specified)
                    if branch and branch not in result.branch.lower():
                        continue
                    
                    # Skip if doesn't match city (if specified)
                    if city and city not in result.city.lower():
                        continue
                    
                    college_data = {
                        'college': result.college,
                        'branch': result.branch,
                        'city': result.city,
                        'category': result.category,
                        'closing_rank': result.closing_rank,
                        'fees': result.full_record.get('fees'),
                        'naac_rating': result.full_record.get('naac_rating'),
                        'nirf_ranking': result.full_record.get('nirf_ranking'),
                        'confidence': round(result.score, 3),
                        'recommendation': None
                    }
                    
                    # Categorize based on rank comparison
                    if result.closing_rank:
                        rank_diff = result.closing_rank - rank
                        
                        if rank_diff >= -1000:  # Safe options (your rank is better or close)
                            college_data['recommendation'] = 'safe'
                            safe_options.append(college_data)
                        elif rank_diff >= -5000:  # Ambitious but possible
                            college_data['recommendation'] = 'ambitious'
                            ambitious_options.append(college_data)
                        else:  # Backup options
                            college_data['recommendation'] = 'backup'
                            backup_options.append(college_data)
                    else:
                        # Unknown rank - add to backup options
                        college_data['recommendation'] = 'unknown_cutoff'
                        backup_options.append(college_data)
                
                # Sort by confidence and limit results
                safe_options = sorted(safe_options, key=lambda x: (-x['confidence'], x.get('closing_rank', 999999)))[:8]
                ambitious_options = sorted(ambitious_options, key=lambda x: (-x['confidence'], x.get('closing_rank', 999999)))[:8]
                backup_options = sorted(backup_options, key=lambda x: (-x['confidence'], x.get('closing_rank', 999999)))[:6]
                
            except Exception as e:
                logger.error(f"Vector search error: {e}")
                # Continue with empty results
        
        # Update statistics
        update_daily_stats('suggestions_generated')
        
        # Log suggestion request
        response_time_ms = int((time.time() - start_time) * 1000)
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO chat_logs 
                    (session_id, query, response_length, retrieval_ids, llm_decision, 
                     has_verified_context, response_time_ms, client_ip)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    f"Suggestion request: rank {rank}, category {category}, branch {branch}, city {city}",
                    len(safe_options) + len(ambitious_options) + len(backup_options),
                    json.dumps([]),
                    json.dumps({'type': 'suggestion', 'filters': {'rank': rank, 'category': category, 'branch': branch, 'city': city}}),
                    False,
                    response_time_ms,
                    get_remote_address()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging suggestion: {e}")
        
        return jsonify({
            'safe_options': safe_options,
            'ambitious_options': ambitious_options,
            'backup_options': backup_options,
            'query_info': {
                'rank': rank,
                'category': category,
                'branch': branch,
                'city': city,
                'total_results': len(safe_options) + len(ambitious_options) + len(backup_options)
            },
            'explanations': {
                'safe': 'Colleges where you have a good chance of admission based on historical cutoffs',
                'ambitious': 'Competitive colleges that may be challenging but possible to get',
                'backup': 'Additional options with higher cutoffs than your rank'
            },
            'disclaimer': 'Rankings are based on previous year data and may not reflect current year cutoffs. Always verify with official sources.',
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"Suggestion error: {e}")
        return jsonify({'error': 'Internal server error. Please try again.'}), 500

@app.route('/chat', methods=['POST'])
@limiter.limit("20 per minute")
def chat():
    """
    Enhanced RAG-powered chat endpoint with streaming response
    Implements improved double approval workflow: RAG → LLM validation → stream
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        user_message = data['message'].strip()
        session_id = data.get('session_id') or get_session_id()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        if len(user_message) > 1000:
            return jsonify({'error': 'Message too long (max 1000 characters)'}), 400
        
        # Get user session
        user_session = get_user_session(session_id)
        user_session['total_queries'] = user_session.get('total_queries', 0) + 1
        
        # Detect query intent
        intent = query_processor.detect_intent(user_message)
        logger.info(f"Query intent detected: {intent}")
        
        # Step 1: RAG Retrieval
        verified_context = ""
        retrieval_ids = []
        rag_confidence = 0.0
        
        if vector_store and intent['type'] in ['rank_suggestion', 'general_chat']:
            try:
                # Build optimized search query
                search_query = query_processor.build_search_query(intent, user_message)
                
                results = vector_store.retrieve_relevant_colleges(
                    search_query,
                    top_k=8,
                    score_threshold=config.RAG_SCORE_THRESHOLD
                )
                
                # Apply intelligent filtering based on intent
                if intent.get('rank') and intent['rank'] > 0:
                    filtered_results = []
                    query_rank = intent['rank']
                    
                    for result in results:
                        if result.closing_rank:
                            # Include if rank is within reasonable range (±15k)
                            rank_diff = abs(result.closing_rank - query_rank)
                            if rank_diff <= 15000:
                                filtered_results.append(result)
                        else:
                            # Include records without rank data
                            filtered_results.append(result)
                    
                    results = filtered_results[:5]  # Top 5 most relevant
                
                # Calculate confidence and create context
                rag_confidence = calculate_retrieval_confidence(results, intent)
                
                if rag_confidence >= config.RAG_SCORE_THRESHOLD and len(results) >= config.RAG_MIN_RESULTS:
                    verified_context = create_verified_context_block(results)
                    retrieval_ids = [result.record_id for result in results]
                    logger.info(f"Using RAG context with {len(results)} results, confidence: {rag_confidence:.3f}")
                else:
                    logger.info(f"RAG confidence too low ({rag_confidence:.3f}) or insufficient results ({len(results)})")
                    
            except Exception as e:
                logger.error(f"RAG retrieval error: {e}")
                rag_confidence = 0.0
        
        # Step 2: Prepare LLM messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add conversation history for context (last 3 exchanges)
        recent_history = user_session.get('conversation_history', [])[-3:]
        for exchange in recent_history:
            messages.append({"role": "user", "content": exchange['query']})
            messages.append({"role": "assistant", "content": exchange['response'][:500]})  # Truncate for context
        
        # Add current query with context if available
        user_content = user_message
        if verified_context:
            user_content = f"{verified_context}\n\nUser Question: {user_message}"
        
        messages.append({"role": "user", "content": user_content})
        
        # Log LLM decision
        llm_decision = {
            'has_context': bool(verified_context),
            'context_records': len(retrieval_ids),
            'intent_type': intent['type'],
            'rag_confidence': rag_confidence,
            'model_used': config.LLM_MODEL,
            'query_rank': intent.get('rank'),
            'conversation_turns': len(recent_history)
        }
        
        # Step 3: Stream Response
        def generate_response():
            nonlocal llm_decision
            
            try:
                full_response = ""
                chunk_count = 0
                
                # Send initial metadata
                yield f"data: {json.dumps({'type': 'metadata', 'session_id': session_id, 'has_context': bool(verified_context), 'confidence': rag_confidence})}\n\n"
                
                # Stream LLM response
                for content_chunk in llm_client.stream_response(messages):
                    full_response += content_chunk
                    chunk_count += 1
                    yield f"data: {json.dumps({'type': 'delta', 'content': content_chunk})}\n\n"
                
                # Update session with conversation
                conversation_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'query': user_message,
                    'response': full_response,
                    'intent': intent,
                    'used_rag': bool(verified_context),
                    'confidence': rag_confidence
                }
                
                user_session['conversation_history'].append(conversation_entry)
                
                # Keep only last 10 exchanges
                if len(user_session['conversation_history']) > 10:
                    user_session['conversation_history'] = user_session['conversation_history'][-10:]
                
                save_user_session(session_id, user_session)
                
                # Calculate final response time
                response_time_ms = int((time.time() - start_time) * 1000)
                llm_decision['response_time_ms'] = response_time_ms
                llm_decision['chunk_count'] = chunk_count
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'done', 'total_length': len(full_response), 'chunks': chunk_count})}\n\n"
                
                # Log complete interaction
                log_chat_interaction(
                    session_id=session_id,
                    query=user_message,
                    response=full_response,
                    retrieval_ids=retrieval_ids,
                    llm_decision=llm_decision,
                    response_time_ms=response_time_ms
                )
                
                # Update statistics
                update_daily_stats('chat_messages')
                if verified_context:
                    update_daily_stats('rag_responses')
                else:
                    update_daily_stats('llm_only_responses')
                
            except Exception as e:
                logger.error(f"Response generation error: {e}")
                error_msg = "I apologize, but I encountered an error generating the response. Please try again."
                yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
        
        return Response(
            generate_response(),
            content_type='text/plain; charset=utf-8',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',  # Disable nginx buffering
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Internal server error. Please try again.'}), 500

@app.route('/feedback', methods=['POST'])
@limiter.limit("10 per minute")
def submit_feedback():
    """Enhanced feedback submission with better validation"""
    try:
        data = request.get_json()
        
        required_fields = ['session_id', 'rating']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields: session_id, rating'}), 400
        
        session_id = data['session_id']
        rating = data['rating'].lower()
        comment = data.get('comment', '').strip()
        query_context = data.get('query_context', '').strip()
        
        # Validate rating
        if rating not in ['positive', 'negative']:
            return jsonify({'error': 'Rating must be either "positive" or "negative"'}), 400
        
        # Validate comment length
        if len(comment) > 500:
            return jsonify({'error': 'Comment too long (max 500 characters)'}), 400
        
        # Store feedback in database
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO feedback (session_id, rating, comment, query_context, client_ip)
                    VALUES (?, ?, ?, ?, ?)
                ''', (session_id, rating, comment, query_context, get_remote_address()))
                conn.commit()
                
                # Update statistics
                update_daily_stats('feedback_submitted')
                update_daily_stats(f'feedback_{rating}')
                
        except Exception as e:
            logger.error(f"Database error in feedback: {e}")
            return jsonify({'error': 'Failed to save feedback'}), 500
        
        logger.info(f"Feedback received: {rating} for session {session_id}")
        
        return jsonify({
            'status': 'success',
            'message': 'Thank you for your feedback! It helps us improve CET-Mentor.'
        })
        
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get comprehensive application statistics"""
    try:
        # Get database stats
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Total sessions
            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]
            
            # Active sessions (last 24 hours)
            cursor.execute(
                "SELECT COUNT(*) FROM sessions WHERE last_activity > ?",
                (datetime.now() - timedelta(hours=24),)
            )
            active_sessions = cursor.fetchone()[0]
            
            # Total chat messages
            cursor.execute("SELECT COUNT(*) FROM chat_logs")
            total_chats = cursor.fetchone()[0]
            
            # Feedback stats
            cursor.execute("SELECT rating, COUNT(*) FROM feedback GROUP BY rating")
            feedback_data = dict(cursor.fetchall())
            
            # Daily stats (last 7 days)
            cursor.execute('''
                SELECT date, metric_name, metric_value 
                FROM app_stats 
                WHERE date >= ? 
                ORDER BY date DESC
            ''', (datetime.now().date() - timedelta(days=7),))
            
            daily_stats = {}
            for row in cursor.fetchall():
                date_str = row[0]
                if date_str not in daily_stats:
                    daily_stats[date_str] = {}
                daily_stats[date_str][row[1]] = row[2]
        
        # LLM client stats
        llm_stats = llm_client.get_stats()
        
        # System stats
        system_stats = {
            'vector_store_status': 'available' if vector_store else 'unavailable',
            'vector_store_module': VECTOR_STORE_AVAILABLE,
            'configuration': {
                'model': config.LLM_MODEL,
                'fallback_model': config.FALLBACK_MODEL,
                'embeddings_mode': config.EMBEDDINGS_MODE,
                'rag_threshold': config.RAG_SCORE_THRESHOLD
            }
        }
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'sessions': {
                'total': total_sessions,
                'active_24h': active_sessions
            },
            'interactions': {
                'total_chats': total_chats,
                'feedback': {
                    'positive': feedback_data.get('positive', 0),
                    'negative': feedback_data.get('negative', 0),
                    'total': sum(feedback_data.values())
                }
            },
            'daily_stats': daily_stats,
            'llm_stats': llm_stats,
            'system': system_stats
        })
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': 'Could not retrieve statistics'}), 500

@app.route('/session', methods=['GET'])
def get_session_info():
    """Get current session information"""
    try:
        session_id = get_session_id()
        user_session = get_user_session(session_id)
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'total_queries': user_session.get('total_queries', 0),
            'conversation_count': len(user_session.get('conversation_history', [])),
            'created_at': user_session.get('created_at'),
            'recent_history': user_session.get('conversation_history', [])[-3:]  # Last 3 exchanges
        })
        
    except Exception as e:
        logger.error(f"Session info error: {e}")
        return jsonify({'error': 'Could not retrieve session information'}), 500

@app.route('/session', methods=['DELETE'])
def clear_session():
    """Clear current session data"""
    try:
        session_id = session.get('session_id')
        if session_id:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.commit()
            
            session.clear()
            logger.info(f"Session {session_id} cleared")
        
        return jsonify({
            'status': 'success',
            'message': 'Session cleared successfully'
        })
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return jsonify({'error': 'Could not clear session'}), 500

# Admin Routes
@app.route('/admin/reindex', methods=['POST'])
def admin_reindex():
    """Admin endpoint to rebuild vector index"""
    try:
        # Check admin authorization
        auth_header = request.headers.get('Authorization')
        if not auth_header or auth_header != f"Bearer {config.ADMIN_SECRET}":
            return jsonify({'error': 'Unauthorized'}), 401
        
        data = request.get_json() or {}
        data_file = data.get('data_file', 'data/mht_cet_data.json')
        
        if not os.path.exists(data_file):
            return jsonify({'error': f'Data file {data_file} not found'}), 400
        
        # Rebuild index
        global vector_store
        if vector_store:
            try:
                vector_store.build_index(data_file, save_index=True)
                logger.info(f"Index rebuilt from {data_file}")
                
                return jsonify({
                    'status': 'success',
                    'message': f'Index rebuilt from {data_file}',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Index rebuild error: {e}")
                return jsonify({'error': f'Index rebuild failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Vector store not available'}), 500
            
    except Exception as e:
        logger.error(f"Reindex error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/admin/logs', methods=['GET'])
def admin_logs():
    """Admin endpoint to view chat logs with pagination"""
    try:
        # Check admin authorization
        auth_header = request.headers.get('Authorization')
        if not auth_header or auth_header != f"Bearer {config.ADMIN_SECRET}":
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Get pagination parameters
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 50)), 200)
        offset = (page - 1) * per_page
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM chat_logs")
            total_logs = cursor.fetchone()[0]
            
            # Get paginated logs
            cursor.execute('''
                SELECT session_id, timestamp, query, response_length, 
                       retrieval_ids, llm_decision, has_verified_context, 
                       response_time_ms, client_ip
                FROM chat_logs 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            ''', (per_page, offset))
            
            logs = []
            for row in cursor.fetchall():
                log_entry = {
                    'session_id': row[0],
                    'timestamp': row[1],
                    'query': row[2][:200] + '...' if len(row[2]) > 200 else row[2],  # Truncate for display
                    'response_length': row[3],
                    'retrieval_ids': json.loads(row[4]) if row[4] else [],
                    'llm_decision': json.loads(row[5]) if row[5] else {},
                    'has_verified_context': bool(row[6]),
                    'response_time_ms': row[7],
                    'client_ip': row[8]
                }
                logs.append(log_entry)
            
            # Get feedback stats
            cursor.execute("SELECT rating, COUNT(*) FROM feedback GROUP BY rating")
            feedback_stats = dict(cursor.fetchall())
            
            # Get recent activity stats
            cursor.execute('''
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM chat_logs 
                WHERE timestamp >= ? 
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            ''', (datetime.now() - timedelta(days=7),))
            
            activity_stats = dict(cursor.fetchall())
        
        return jsonify({
            'status': 'success',
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total_logs,
                'pages': (total_logs + per_page - 1) // per_page
            },
            'logs': logs,
            'feedback_stats': {
                'positive': feedback_stats.get('positive', 0),
                'negative': feedback_stats.get('negative', 0),
                'total': sum(feedback_stats.values())
            },
            'activity_stats': activity_stats,
            'system_info': {
                'vector_store_status': 'available' if vector_store else 'unavailable',
                'llm_stats': llm_client.get_stats()
            }
        })
        
    except Exception as e:
        logger.error(f"Admin logs error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/admin/dashboard', methods=['GET'])
def admin_dashboard():
    """Admin dashboard with comprehensive analytics"""
    try:
        # Check admin authorization
        auth_header = request.headers.get('Authorization')
        if not auth_header or auth_header != f"Bearer {config.ADMIN_SECRET}":
            return jsonify({'error': 'Unauthorized'}), 401
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Overview stats
            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chat_logs")
            total_chats = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total_feedback = cursor.fetchone()[0]
            
            # Today's activity
            today = datetime.now().date()
            cursor.execute(
                "SELECT COUNT(*) FROM chat_logs WHERE DATE(timestamp) = ?", 
                (today,)
            )
            today_chats = cursor.fetchone()[0]
            
            # RAG usage stats
            cursor.execute(
                "SELECT has_verified_context, COUNT(*) FROM chat_logs GROUP BY has_verified_context"
            )
            rag_stats = dict(cursor.fetchall())
            
            # Response time stats
            cursor.execute('''
                SELECT 
                    AVG(response_time_ms) as avg_time,
                    MIN(response_time_ms) as min_time,
                    MAX(response_time_ms) as max_time
                FROM chat_logs 
                WHERE response_time_ms IS NOT NULL
            ''')
            time_stats = cursor.fetchone()
            
            # Top error patterns
            cursor.execute('''
                SELECT llm_decision, COUNT(*) as count
                FROM chat_logs 
                WHERE JSON_EXTRACT(llm_decision, '$.error') IS NOT NULL
                GROUP BY llm_decision
                ORDER BY count DESC
                LIMIT 10
            ''')
            error_patterns = cursor.fetchall()
            
            # User engagement metrics
            cursor.execute('''
                SELECT 
                    COUNT(DISTINCT session_id) as unique_users,
                    AVG(query_count) as avg_queries_per_user
                FROM (
                    SELECT session_id, COUNT(*) as query_count 
                    FROM chat_logs 
                    GROUP BY session_id
                ) user_stats
            ''')
            engagement = cursor.fetchone()
            
            # Weekly trends
            cursor.execute('''
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as chats,
                    COUNT(DISTINCT session_id) as unique_users,
                    AVG(response_time_ms) as avg_response_time
                FROM chat_logs 
                WHERE timestamp >= ? 
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', (datetime.now() - timedelta(days=14),))
            
            weekly_trends = []
            for row in cursor.fetchall():
                weekly_trends.append({
                    'date': row[0],
                    'chats': row[1],
                    'unique_users': row[2],
                    'avg_response_time': round(row[3], 2) if row[3] else 0
                })
        
        # System health metrics
        config_issues = config.validate()
        
        dashboard_data = {
            'overview': {
                'total_sessions': total_sessions,
                'total_chats': total_chats,
                'total_feedback': total_feedback,
                'today_chats': today_chats
            },
            'rag_usage': {
                'with_context': rag_stats.get(1, 0),
                'without_context': rag_stats.get(0, 0),
                'context_rate': rag_stats.get(1, 0) / max(total_chats, 1)
            },
            'performance': {
                'avg_response_time': round(time_stats[0], 2) if time_stats[0] else 0,
                'min_response_time': time_stats[1] or 0,
                'max_response_time': time_stats[2] or 0
            },
            'engagement': {
                'unique_users': engagement[0] or 0,
                'avg_queries_per_user': round(engagement[1], 2) if engagement[1] else 0
            },
            'trends': weekly_trends,
            'system_health': {
                'vector_store': 'healthy' if vector_store else 'unavailable',
                'database': 'healthy',  # If we got here, DB is working
                'llm_api': 'healthy' if config.OPENROUTER_API_KEY else 'not_configured',
                'config_issues': config_issues
            },
            'llm_stats': llm_client.get_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(dashboard_data)
        
    except Exception as e:
        logger.error(f"Admin dashboard error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(405)
def method_not_allowed_error(error):
    """Handle 405 errors"""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The HTTP method is not allowed for this endpoint'
    }), 405

@app.errorhandler(429)
def ratelimit_handler(error):
    """Enhanced rate limit error handler"""
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests. Please wait before trying again.',
        'retry_after': '60 seconds'
    }), 429

@app.errorhandler(500)
def internal_error_handler(error):
    """Enhanced internal error handler"""
    error_id = str(uuid.uuid4())[:8]
    logger.error(f"Internal error [{error_id}]: {error}")
    
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong on our end. Please try again later.',
        'error_id': error_id
    }), 500

@app.errorhandler(Exception)
def handle_exception(error):
    """Global exception handler"""
    error_id = str(uuid.uuid4())[:8]
    logger.error(f"Unhandled exception [{error_id}]: {error}", exc_info=True)
    
    return jsonify({
        'error': 'Unexpected error',
        'message': 'An unexpected error occurred. Please try again.',
        'error_id': error_id
    }), 500

# Utility Routes for Testing and Development
@app.route('/test-llm', methods=['POST'])
def test_llm():
    """Test endpoint for LLM connectivity (development only)"""
    if not config.DEBUG:
        return jsonify({'error': 'Test endpoints not available in production'}), 403
    
    try:
        data = request.get_json() or {}
        test_query = data.get('query', 'Hello, are you working properly?')
        
        messages = [
            {"role": "system", "content": "You are a test assistant. Respond briefly to confirm you're working."},
            {"role": "user", "content": test_query}
        ]
        
        response = llm_client.call_api(messages, stream=False)
        result = response.json()
        
        return jsonify({
            'status': 'success',
            'response': result['choices'][0]['message']['content'],
            'model_used': config.LLM_MODEL,
            'test_query': test_query
        })
        
    except Exception as e:
        logger.error(f"LLM test error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'model_tested': config.LLM_MODEL
        }), 500

@app.route('/test-rag', methods=['POST'])
def test_rag():
    """Test endpoint for RAG functionality (development only)"""
    if not config.DEBUG:
        return jsonify({'error': 'Test endpoints not available in production'}), 403
    
    try:
        data = request.get_json() or {}
        query = data.get('query', 'computer science mumbai rank 50000')
        
        if not vector_store:
            return jsonify({
                'status': 'error',
                'error': 'Vector store not available',
                'suggestions': ['Check if vector store is properly initialized', 'Verify data files exist']
            })
        
        results = vector_store.retrieve_relevant_colleges(query, top_k=5)
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                'college': result.college,
                'branch': result.branch,
                'city': result.city,
                'category': result.category,
                'closing_rank': result.closing_rank,
                'score': round(result.score, 3)
            })
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results_count': len(results),
            'results': formatted_results,
            'vector_store_config': {
                'embeddings_mode': config.EMBEDDINGS_MODE,
                'embeddings_model': config.EMBEDDINGS_MODEL,
                'threshold': config.RAG_SCORE_THRESHOLD
            }
        })
        
    except Exception as e:
        logger.error(f"RAG test error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

# Application Initialization and Startup
def initialize_application():
    """Initialize all application components"""
    logger.info("Initializing CET-Mentor v2.1...")
    
    # Check configuration
    config_issues = config.validate()
    if config_issues:
        logger.warning(f"Configuration issues detected: {config_issues}")
        if not config.DEBUG:
            for issue in config_issues:
                if 'API_KEY' in issue:
                    logger.error(f"Critical configuration issue: {issue}")
    
    # Initialize database
    try:
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        if not config.DEBUG:
            raise
    
    # Initialize vector store
    vector_initialized = initialize_vector_store()
    logger.info(f"Vector store: {'Available' if vector_initialized else 'Unavailable'}")
    
    # Cleanup old data
    try:
        cleanup_old_sessions()
        logger.info("Old session cleanup completed")
    except Exception as e:
        logger.warning(f"Session cleanup failed: {e}")
    
    # Log startup information
    logger.info("=" * 60)
    logger.info("CET-Mentor v2.1 Flask Backend Started")
    logger.info("=" * 60)
    logger.info(f"Environment: {'Development' if config.DEBUG else 'Production'}")
    logger.info(f"Port: {config.PORT}")
    logger.info(f"Vector Store: {'Available' if vector_store else 'Limited functionality'}")
    logger.info(f"LLM Model: {config.LLM_MODEL}")
    logger.info(f"Embeddings: {config.EMBEDDINGS_MODE} ({config.EMBEDDINGS_MODEL})")
    logger.info(f"Database: {config.DATABASE_URL}")
    logger.info("=" * 60)

# WSGI Application for Production
def create_app():
    """Factory function for creating Flask app (for WSGI servers)"""
    initialize_application()
    return app

# Development Server
def main():
    """Main function for development server"""
    initialize_application()
    
    # Run Flask development server
    try:
        app.run(
            host='0.0.0.0',
            port=config.PORT,
            debug=config.DEBUG,
            threaded=True,
            use_reloader=config.DEBUG
        )
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise

if __name__ == "__main__":
    main()
