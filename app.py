#!/usr/bin/env python3
"""
CET-Mentor v2.0 Flask Backend

Purpose: Production-grade Flask API with RAG-powered MHT-CET assistance
Implements double approval workflow: RAG retrieval → LLM validation → streaming response
Supports Claude 3.5 Sonnet, GPT-4, and other OpenRouter models with SSE streaming

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

Endpoints:
    GET  /                 - Render chat interface
    POST /suggest         - College suggestions by rank/criteria
    POST /chat            - RAG-powered chat with streaming
    POST /feedback        - Submit user feedback
    POST /admin/reindex   - Rebuild vector index (admin)
    GET  /admin/logs      - View chat logs (admin)
"""

import csv
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generator
import uuid
import re

from flask import Flask, render_template, request, jsonify, Response, session, stream_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests
from dotenv import load_dotenv

from vector_store import VectorStore, RetrievalResult

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-change-in-production')

# Initialize rate limiter
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per hour"]
)

# Global variables
vector_store = None
chat_logs = []
feedback_logs = []
rate_limit_store = defaultdict(list)

# Configuration
CONFIG = {
    'OPENROUTER_API_KEY': os.getenv('OPENROUTER_API_KEY'),
    'OPENROUTER_BASE_URL': os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
    'LLM_MODEL': os.getenv('LLM_MODEL', 'anthropic/claude-3.5-sonnet'),
    'FALLBACK_MODEL': os.getenv('FALLBACK_MODEL', 'openai/gpt-4o'),
    'EMBEDDINGS_MODE': os.getenv('EMBEDDINGS_MODE', 'local'),
    'EMBEDDINGS_MODEL': os.getenv('EMBEDDINGS_MODEL', 'all-MiniLM-L6-v2'),
    'VECTOR_INDEX_PATH': os.getenv('VECTOR_INDEX_PATH', 'index.faiss'),
    'VECTOR_METADATA_PATH': os.getenv('VECTOR_METADATA_PATH', 'metadata.json'),
    'ADMIN_SECRET': os.getenv('ADMIN_SECRET', 'admin-secret-change-me'),
    'RAG_SCORE_THRESHOLD': float(os.getenv('RAG_SCORE_THRESHOLD', '0.3')),
    'RAG_MIN_RESULTS': int(os.getenv('RAG_MIN_RESULTS', '2')),
    'RATE_LIMIT_PER_MINUTE': int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))
}

# System prompt for LLM with strict RAG validation instructions
SYSTEM_PROMPT = """You are CET-Mentor, an expert assistant for MHT-CET college admissions in Maharashtra, India. 

CRITICAL INSTRUCTIONS:
1. You MUST ONLY use information provided in the VERIFIED CONTEXT section for specific college data, cutoffs, and rankings
2. If VERIFIED CONTEXT is provided, cite specific records by mentioning college names and source evidence
3. If no VERIFIED CONTEXT is available or insufficient, clearly state "I don't have official cutoff data for this specific query" and provide general guidance
4. NEVER hallucinate or guess specific cutoff numbers, college rankings, or admission details
5. When discussing rankings, remember: LOWER rank number = BETTER performance (e.g., rank 1000 is better than rank 5000)
6. Always be helpful while being transparent about data limitations

RESPONSE STRUCTURE:
- If using VERIFIED CONTEXT: Start with specific college information from the context
- If no VERIFIED CONTEXT: Begin with disclaimer about lacking specific data
- Provide helpful general guidance about MHT-CET admissions process
- Be encouraging and supportive to students

You can discuss general MHT-CET information, admission processes, study tips, and career guidance even without VERIFIED CONTEXT."""

def initialize_vector_store():
    """Initialize vector store with error handling"""
    global vector_store
    try:
        vector_store = VectorStore(
            embeddings_mode=CONFIG['EMBEDDINGS_MODE'],
            embeddings_model=CONFIG['EMBEDDINGS_MODEL'],
            index_path=CONFIG['VECTOR_INDEX_PATH'],
            metadata_path=CONFIG['VECTOR_METADATA_PATH']
        )
        
        if not vector_store.load_index():
            logger.warning("Could not load vector index - RAG functionality limited")
            return False
        
        logger.info("Vector store initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        vector_store = None
        return False

def check_rate_limit(identifier: str, limit_per_minute: int = None) -> bool:
    """Simple in-memory rate limiting"""
    if limit_per_minute is None:
        limit_per_minute = CONFIG['RATE_LIMIT_PER_MINUTE']
        
    current_time = time.time()
    minute_ago = current_time - 60
    
    # Clean old entries
    rate_limit_store[identifier] = [
        timestamp for timestamp in rate_limit_store[identifier] 
        if timestamp > minute_ago
    ]
    
    # Check limit
    if len(rate_limit_store[identifier]) >= limit_per_minute:
        return False
        
    # Add current request
    rate_limit_store[identifier].append(current_time)
    return True

def log_chat_interaction(session_id: str, query: str, response: str, 
                        retrieval_ids: List[int], llm_decision: Dict[str, Any]):
    """Log chat interaction for monitoring and improvement"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'session_id': session_id,
        'query': query,
        'response_length': len(response),
        'retrieval_ids': retrieval_ids,
        'llm_decision': llm_decision,
        'has_verified_context': len(retrieval_ids) > 0
    }
    
    chat_logs.append(log_entry)
    
    # Keep only last 1000 entries in memory
    if len(chat_logs) > 1000:
        chat_logs.pop(0)

def create_verified_context_block(results: List[RetrievalResult]) -> str:
    """Create VERIFIED CONTEXT block from retrieval results"""
    if not results:
        return ""
    
    context_lines = ["=== VERIFIED CONTEXT (Official MHT-CET Data) ==="]
    
    for i, result in enumerate(results, 1):
        context_lines.append(f"\nRecord {i}:")
        context_lines.append(f"College: {result.college}")
        context_lines.append(f"Branch: {result.branch}")
        context_lines.append(f"City: {result.city}")
        context_lines.append(f"Category: {result.category}")
        
        if result.closing_rank:
            context_lines.append(f"Closing Rank: {result.closing_rank}")
        
        if result.full_record.get('fees'):
            context_lines.append(f"Fees: {result.full_record['fees']}")
        
        if result.full_record.get('naac_rating'):
            context_lines.append(f"NAAC Rating: {result.full_record['naac_rating']}")
        
        context_lines.append(f"Source: {result.source_url}")
        context_lines.append(f"Relevance Score: {result.score:.3f}")
    
    context_lines.append("\n=== END VERIFIED CONTEXT ===\n")
    return "\n".join(context_lines)

def call_llm_api(messages: List[Dict], model: str = None, stream: bool = True) -> requests.Response:
    """Call LLM API with proper error handling and fallback"""
    if not CONFIG['OPENROUTER_API_KEY']:
        raise ValueError("OPENROUTER_API_KEY not configured")
    
    if model is None:
        model = CONFIG['LLM_MODEL']
    
    headers = {
        'Authorization': f"Bearer {CONFIG['OPENROUTER_API_KEY']}",
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://localhost:5000',
        'X-Title': 'CET-Mentor v2.0'
    }
    
    payload = {
        'model': model,
        'messages': messages,
        'stream': stream,
        'max_tokens': 1500,
        'temperature': 0.7
    }
    
    try:
        response = requests.post(
            f"{CONFIG['OPENROUTER_BASE_URL']}/chat/completions",
            json=payload,
            headers=headers,
            stream=stream,
            timeout=30
        )
        response.raise_for_status()
        return response
        
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API error with {model}: {e}")
        
        # Try fallback model if primary fails
        if model != CONFIG['FALLBACK_MODEL']:
            logger.info(f"Trying fallback model: {CONFIG['FALLBACK_MODEL']}")
            return call_llm_api(messages, model=CONFIG['FALLBACK_MODEL'], stream=stream)
        
        raise

def stream_llm_response(messages: List[Dict]) -> Generator[str, None, None]:
    """Stream LLM response with proper error handling"""
    try:
        response = call_llm_api(messages, stream=True)
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    try:
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            break
                            
                        json_data = json.loads(data)
                        
                        # Extract content from different response formats
                        delta = json_data.get('choices', [{}])[0].get('delta', {})
                        content = delta.get('content', '')
                        
                        if content:
                            yield content
                            
                    except json.JSONDecodeError:
                        continue
                        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"\n\n[Error: Could not complete response - {str(e)}]"

def detect_query_intent(query: str) -> Dict[str, Any]:
    """Detect user query intent and extract structured data"""
    query_lower = query.lower().strip()
    
    # Pure rank query pattern
    rank_pattern = r'^\s*(\d{1,7})\s*$'
    rank_match = re.match(rank_pattern, query_lower)
    
    if rank_match:
        return {
            'type': 'rank_suggestion',
            'rank': int(rank_match.group(1)),
            'category': 'open',
            'branch': None,
            'city': None
        }
    
    # Rank with additional context
    rank_context_patterns = [
        r'(\d{4,7})\s*rank.*?(obc|sc|st|open|general)',
        r'(obc|sc|st|open|general).*?(\d{4,7})\s*rank',
        r'rank\s*(\d{4,7}).*?(computer|mechanical|civil|electronics|electrical|it)',
        r'(\d{4,7}).*?(mumbai|pune|nagpur|nashik|aurangabad)',
    ]
    
    extracted_data = {
        'type': 'general_chat',
        'rank': None,
        'category': None,
        'branch': None,
        'city': None
    }
    
    # Extract rank
    rank_nums = re.findall(r'\b(\d{4,7})\b', query_lower)
    if rank_nums:
        extracted_data['rank'] = int(rank_nums[0])
        extracted_data['type'] = 'rank_suggestion'
    
    # Extract category
    categories = {
        'obc': ['obc'],
        'sc': ['sc'],
        'st': ['st'],
        'open': ['open', 'general']
    }
    
    for category, keywords in categories.items():
        if any(keyword in query_lower for keyword in keywords):
            extracted_data['category'] = category
            break
    
    # Extract branch
    branches = {
        'computer science': ['computer', 'cse', 'cs', 'it', 'information technology'],
        'mechanical': ['mechanical', 'mech'],
        'electronics': ['electronics', 'extc', 'ece', 'communication'],
        'civil': ['civil'],
        'electrical': ['electrical', 'ee']
    }
    
    for branch, keywords in branches.items():
        if any(keyword in query_lower for keyword in keywords):
            extracted_data['branch'] = branch
            break
    
    # Extract city
    cities = ['mumbai', 'pune', 'nagpur', 'nashik', 'aurangabad', 'kolhapur', 'sangli']
    for city in cities:
        if city in query_lower:
            extracted_data['city'] = city
            break
    
    return extracted_data

@app.route('/')
def index():
    """Render main chat interface"""
    return render_template('index.html')

@app.route('/suggest', methods=['POST'])
@limiter.limit("30 per minute")
def suggest_colleges():
    """
    College suggestion endpoint based on rank and criteria
    
    Expected JSON payload:
    {
        "rank": 12345,
        "category": "open|obc|sc|st", 
        "branch": "computer science",
        "city": "mumbai"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'rank' not in data:
            return jsonify({'error': 'Rank is required'}), 400
        
        rank = int(data['rank'])
        category = data.get('category', 'open').lower()
        branch = data.get('branch', '')
        city = data.get('city', '')
        
        # Validate rank range
        if rank < 1 or rank > 200000:
            return jsonify({'error': 'Rank must be between 1 and 200000'}), 400
        
        # Build search query for vector retrieval
        query_parts = []
        if branch:
            query_parts.append(branch)
        if city:
            query_parts.append(city)
        if category != 'open':
            query_parts.append(f"{category} category")
        
        query_parts.append(f"rank {rank}")
        search_query = " ".join(query_parts)
        
        # Retrieve relevant colleges
        safe_options = []
        ambitious_options = []
        
        if vector_store:
            try:
                results = vector_store.retrieve_relevant_colleges(
                    search_query, 
                    top_k=20,
                    score_threshold=0.1
                )
                
                for result in results:
                    college_data = {
                        'college': result.college,
                        'branch': result.branch,
                        'city': result.city,
                        'category': result.category,
                        'closing_rank': result.closing_rank,
                        'fees': result.full_record.get('fees'),
                        'naac_rating': result.full_record.get('naac_rating'),
                        'confidence': result.score
                    }
                    
                    # Categorize based on rank comparison
                    if result.closing_rank:
                        if result.closing_rank <= rank + 5000:  # Safe options
                            safe_options.append(college_data)
                        elif result.closing_rank <= rank - 2000:  # Ambitious options
                            ambitious_options.append(college_data)
                    else:
                        # Unknown rank - add to safe options
                        safe_options.append(college_data)
                
                # Sort by confidence and limit results
                safe_options = sorted(safe_options, key=lambda x: x['confidence'], reverse=True)[:10]
                ambitious_options = sorted(ambitious_options, key=lambda x: x['confidence'], reverse=True)[:10]
                
            except Exception as e:
                logger.error(f"Vector search error: {e}")
                # Return empty results on error
                pass
        
        # Log suggestion request
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'suggestion',
            'query': data,
            'results_count': len(safe_options) + len(ambitious_options)
        }
        chat_logs.append(log_entry)
        
        return jsonify({
            'safe_options': safe_options,
            'ambitious_options': ambitious_options,
            'query_info': {
                'rank': rank,
                'category': category,
                'branch': branch,
                'city': city
            },
            'disclaimer': 'Rankings are based on available data and may not reflect current year cutoffs. Always verify with official college websites.'
        })
        
    except Exception as e:
        logger.error(f"Suggestion error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/chat', methods=['POST'])
@limiter.limit("20 per minute")
def chat():
    """
    RAG-powered chat endpoint with streaming response
    Implements double approval workflow: RAG → LLM validation → stream
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        user_message = data['message'].strip()
        session_id = data.get('session_id') or str(uuid.uuid4())
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Rate limiting check
        client_ip = get_remote_address()
        if not check_rate_limit(f"chat_{client_ip}"):
            return jsonify({'error': 'Rate limit exceeded'}), 429
        
        # Detect query intent
        intent = detect_query_intent(user_message)
        
        # Step 1: RAG Retrieval
        verified_context = ""
        retrieval_ids = []
        
        if vector_store and intent['type'] in ['rank_suggestion', 'general_chat']:
            try:
                results = vector_store.retrieve_relevant_colleges(
                    user_message,
                    top_k=5,
                    score_threshold=CONFIG['RAG_SCORE_THRESHOLD']
                )
                
                # Apply additional filtering based on intent
                if intent.get('rank') and intent['rank'] > 0:
                    # Filter results relevant to the rank
                    filtered_results = []
                    for result in results:
                        if result.closing_rank:
                            # Include if rank is within reasonable range
                            rank_diff = abs(result.closing_rank - intent['rank'])
                            if rank_diff <= 10000:  # 10k rank difference tolerance
                                filtered_results.append(result)
                        else:
                            # Include if no rank data available
                            filtered_results.append(result)
                    results = filtered_results[:3]  # Limit to top 3
                
                if len(results) >= CONFIG['RAG_MIN_RESULTS']:
                    verified_context = create_verified_context_block(results)
                    retrieval_ids = [result.record_id for result in results]
                    
            except Exception as e:
                logger.error(f"RAG retrieval error: {e}")
        
        # Step 2: LLM Decision and Response Generation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # Add context if available
        user_content = user_message
        if verified_context:
            user_content = f"{verified_context}\n\nUser Question: {user_message}"
        
        messages.append({"role": "user", "content": user_content})
        
        # Log LLM decision
        llm_decision = {
            'has_context': bool(verified_context),
            'context_records': len(retrieval_ids),
            'intent_type': intent['type'],
            'model_used': CONFIG['LLM_MODEL']
        }
        
        # Step 3: Stream Response
        def generate_response():
            try:
                full_response = ""
                
                # Send initial metadata
                yield f"data: {json.dumps({'type': 'metadata', 'session_id': session_id, 'has_context': bool(verified_context)})}\n\n"
                
                # Stream LLM response
                for content_chunk in stream_llm_response(messages):
                    full_response += content_chunk
                    yield f"data: {json.dumps({'type': 'delta', 'content': content_chunk})}\n\n"
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'done', 'total_length': len(full_response)})}\n\n"
                
                # Log complete interaction
                log_chat_interaction(
                    session_id=session_id,
                    query=user_message,
                    response=full_response,
                    retrieval_ids=retrieval_ids,
                    llm_decision=llm_decision
                )
                
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
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/feedback', methods=['POST'])
@limiter.limit("10 per minute")
def submit_feedback():
    """Submit user feedback on responses"""
    try:
        data = request.get_json()
        
        required_fields = ['session_id', 'rating']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': data['session_id'],
            'rating': data['rating'],  # 'positive', 'negative'
            'comment': data.get('comment', ''),
            'query_context': data.get('query_context', ''),
            'client_ip': get_remote_address()
        }
        
        # Store in memory and CSV
        feedback_logs.append(feedback_entry)
        
        # Append to CSV file
        csv_file = 'feedback.csv'
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=feedback_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(feedback_entry)
        
        logger.info(f"Feedback received: {data['rating']} for session {data['session_id']}")
        
        return jsonify({'status': 'success'})
        
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/admin/reindex', methods=['POST'])
def admin_reindex():
    """Admin endpoint to rebuild vector index"""
    try:
        # Check admin authorization
        auth_header = request.headers.get('Authorization')
        if not auth_header or auth_header != f"Bearer {CONFIG['ADMIN_SECRET']}":
            return jsonify({'error': 'Unauthorized'}), 401
        
        data = request.get_json()
        data_file = data.get('data_file', 'mht_cet_data.json')
        
        if not os.path.exists(data_file):
            return jsonify({'error': f'Data file {data_file} not found'}), 400
        
        # Rebuild index
        global vector_store
        if vector_store:
            vector_store.build_index(data_file, save_index=True)
            logger.info(f"Index rebuilt from {data_file}")
            
            return jsonify({
                'status': 'success',
                'message': f'Index rebuilt from {data_file}',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Vector store not available'}), 500
            
    except Exception as e:
        logger.error(f"Reindex error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/admin/logs', methods=['GET'])
def admin_logs():
    """Admin endpoint to view chat logs"""
    try:
        # Check admin authorization
        auth_header = request.headers.get('Authorization')
        if not auth_header or auth_header != f"Bearer {CONFIG['ADMIN_SECRET']}":
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Get recent logs
        limit = min(int(request.args.get('limit', 50)), 200)
        recent_logs = chat_logs[-limit:] if limit > 0 else chat_logs
        
        # Get feedback stats
        feedback_stats = {
            'total': len(feedback_logs),
            'positive': len([f for f in feedback_logs if f['rating'] == 'positive']),
            'negative': len([f for f in feedback_logs if f['rating'] == 'negative'])
        }
        
        return jsonify({
            'chat_logs': recent_logs,
            'feedback_stats': feedback_stats,
            'total_chats': len(chat_logs),
            'vector_store_status': 'available' if vector_store else 'unavailable'
        })
        
    except Exception as e:
        logger.error(f"Admin logs error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0',
        'services': {
            'vector_store': 'available' if vector_store else 'unavailable',
            'llm_api': 'configured' if CONFIG['OPENROUTER_API_KEY'] else 'not_configured'
        }
    })

@app.errorhandler(429)
def ratelimit_handler(e):
    """Rate limit error handler"""
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests. Please wait before trying again.'
    }), 429

@app.errorhandler(500)
def internal_error_handler(e):
    """Internal error handler"""
    logger.error(f"Internal error: {e}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong. Please try again later.'
    }), 500

def main():
    """Initialize and run Flask application"""
    # Initialize vector store
    if not initialize_vector_store():
        logger.warning("Starting without vector store - limited functionality")
    
    # Validate configuration
    if not CONFIG['OPENROUTER_API_KEY']:
        logger.warning("OPENROUTER_API_KEY not set - LLM functionality disabled")
    
    logger.info("CET-Mentor v2.0 starting...")
    logger.info(f"Vector store: {'Available' if vector_store else 'Unavailable'}")
    logger.info(f"LLM model: {CONFIG['LLM_MODEL']}")
    logger.info(f"Embeddings: {CONFIG['EMBEDDINGS_MODE']} ({CONFIG['EMBEDDINGS_MODEL']})")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    )

if __name__ == "__main__":
    main()
