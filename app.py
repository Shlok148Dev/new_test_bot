#!/usr/bin/env python3
"""
CET-Mentor v2.0 Flask Application
AI-powered MHT-CET college recommendation system with RAG and streaming responses
"""

import os
import json
import csv
import logging
from datetime import datetime
from typing import List, Dict, Optional, Generator, Any
import uuid

from flask import Flask, request, jsonify, render_template, session, Response, stream_template
from flask_session import Session
from flask_cors import CORS
import requests
from dotenv import load_dotenv

# Import our custom modules
from vector_store import CollegeVectorStore, retrieve_relevant_colleges
import pandas as pd

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cet_mentor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'cet-mentor-secret-key-2024')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True

# Initialize extensions
Session(app)
CORS(app, supports_credentials=True)

class CETMentorAI:
    """Main AI pipeline for CET-Mentor application"""
    
    def __init__(self):
        self.college_data = []
        self.vector_store = None
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        self.default_model = "anthropic/claude-3.5-sonnet"
        self.fallback_model = "openai/gpt-4-turbo"
        
        # System prompts
        self.rag_system_prompt = """You are CET-Mentor, an expert MHT-CET counseling assistant. 

IMPORTANT GUIDELINES:
- Use VERIFIED CONTEXT from the knowledge base when available
- If no relevant context is found, use your general knowledge about MHT-CET and Maharashtra engineering colleges
- RANK is the PRIMARY metric - LOWER rank is BETTER (Rank 1 is best, Rank 50000 is poor)
- Always provide clear, honest, and actionable advice
- Consider category-wise reservations (Open, OBC, SC, ST, EWS)
- Factor in college reputation, location, fees, and placement records
- Be encouraging but realistic about admission chances

CONTEXT HANDLING:
- If verified context is provided, prioritize it over general knowledge
- Cross-reference context data for accuracy
- Explain rank requirements clearly
- Mention both safe and ambitious options when relevant"""

        self.conversational_prompt = """You are CET-Mentor, a friendly and knowledgeable MHT-CET counseling expert.

You help students with:
- College selection based on MHT-CET ranks
- Branch/course guidance
- Category-wise admission criteria
- Fee structure and scholarship information
- Career guidance and placement insights
- General MHT-CET process questions

RANK UNDERSTANDING:
- Lower rank = Better performance (Rank 1 is the best possible)
- Closing ranks indicate the last student admitted to that branch/category
- Students need ranks EQUAL TO OR BETTER (lower) than closing rank for admission

Provide helpful, accurate, and encouraging advice while being realistic about admission possibilities."""
        
        self.load_college_data()
        self.initialize_vector_store()
        
    def load_college_data(self):
        """Load college data from JSON file"""
        try:
            data_file = 'mht_cet_data.json'
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    self.college_data = json.load(f)
                logger.info(f"Loaded {len(self.college_data)} college records")
            else:
                logger.warning(f"College data file {data_file} not found")
                self.college_data = []
        except Exception as e:
            logger.error(f"Error loading college data: {e}")
            self.college_data = []
    
    def initialize_vector_store(self):
        """Initialize vector store for RAG"""
        try:
            self.vector_store = CollegeVectorStore()
            if not self.vector_store.load_vector_store():
                if self.college_data:
                    logger.info("Building new vector store...")
                    self.vector_store.build_from_data('mht_cet_data.json')
                else:
                    logger.warning("No college data available for vector store")
                    self.vector_store = None
            else:
                logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            self.vector_store = None
    
    def get_rank_based_suggestions(self, 
                                 user_rank: int, 
                                 category: str = "Open",
                                 branch_filter: str = None,
                                 city_filter: str = None) -> Dict[str, List[Dict]]:
        """
        Get rank-based college suggestions
        
        Args:
            user_rank: User's MHT-CET rank
            category: Admission category
            branch_filter: Optional branch filter
            city_filter: Optional city filter
            
        Returns:
            Dictionary with safe_options and ambitious_options
        """
        safe_options = []
        ambitious_options = []
        
        try:
            # Filter data based on category
            filtered_data = [
                college for college in self.college_data 
                if college.get('category', 'Open') == category
            ]
            
            # Apply additional filters if specified
            if branch_filter:
                filtered_data = [
                    college for college in filtered_data
                    if branch_filter.lower() in college.get('branch', '').lower()
                ]
            
            if city_filter:
                filtered_data = [
                    college for college in filtered_data
                    if city_filter.lower() in college.get('city', '').lower()
                ]
            
            for college in filtered_data:
                closing_rank = college.get('closing_rank', 0)
                
                if not closing_rank:
                    continue
                
                # Safe options: User rank is better (lower) than closing rank
                # Add buffer of 10% for safety
                safe_threshold = closing_rank * 0.9
                if user_rank <= safe_threshold:
                    safe_options.append({
                        'college': college.get('college', ''),
                        'branch': college.get('branch', ''),
                        'category': college.get('category', ''),
                        'closing_rank': closing_rank,
                        'user_rank': user_rank,
                        'safety_margin': closing_rank - user_rank,
                        'fees': college.get('fees', ''),
                        'city': college.get('city', ''),
                        'naac_rating': college.get('naac_rating', ''),
                        'admission_probability': 'High'
                    })
                
                # Ambitious options: User rank is slightly worse than closing rank
                # Within 20% margin for ambitious attempts
                ambitious_threshold = closing_rank * 1.2
                if closing_rank < user_rank <= ambitious_threshold:
                    rank_gap = user_rank - closing_rank
                    probability = max(10, 60 - (rank_gap / closing_rank * 100))
                    
                    ambitious_options.append({
                        'college': college.get('college', ''),
                        'branch': college.get('branch', ''),
                        'category': college.get('category', ''),
                        'closing_rank': closing_rank,
                        'user_rank': user_rank,
                        'rank_gap': rank_gap,
                        'fees': college.get('fees', ''),
                        'city': college.get('city', ''),
                        'naac_rating': college.get('naac_rating', ''),
                        'admission_probability': f'{probability:.0f}%'
                    })
            
            # Sort by safety margin for safe options
            safe_options.sort(key=lambda x: x.get('safety_margin', 0), reverse=True)
            
            # Sort by rank gap for ambitious options
            ambitious_options.sort(key=lambda x: x.get('rank_gap', float('inf')))
            
            # Limit results
            safe_options = safe_options[:15]
            ambitious_options = ambitious_options[:10]
            
            logger.info(f"Generated {len(safe_options)} safe and {len(ambitious_options)} ambitious options for rank {user_rank}")
            
        except Exception as e:
            logger.error(f"Error generating rank-based suggestions: {e}")
        
        return {
            'safe_options': safe_options,
            'ambitious_options': ambitious_options
        }
    
    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant context using vector store"""
        if not self.vector_store:
            logger.warning("Vector store not available")
            return []
        
        try:
            results = self.vector_store.retrieve_relevant_colleges(
                query, 
                top_k=top_k,
                min_similarity=0.3
            )
            
            # Filter high-quality results
            quality_results = [
                result for result in results 
                if result.get('similarity_score', 0) > 0.4
            ]
            
            logger.info(f"Retrieved {len(quality_results)} quality context results for: {query}")
            return quality_results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def format_context_for_llm(self, context_results: List[Dict]) -> str:
        """Format retrieved context for LLM"""
        if not context_results:
            return ""
        
        context_text = "VERIFIED CONTEXT FROM KNOWLEDGE BASE:\n\n"
        
        for i, result in enumerate(context_results, 1):
            context_text += f"Option {i}:\n"
            context_text += f"College: {result.get('college', 'N/A')}\n"
            context_text += f"Branch: {result.get('branch', 'N/A')}\n"
            context_text += f"Category: {result.get('category', 'N/A')}\n"
            context_text += f"Closing Rank: {result.get('closing_rank', 'N/A')}\n"
            context_text += f"City: {result.get('city', 'N/A')}\n"
            context_text += f"Fees: {result.get('fees', 'N/A')}\n"
            context_text += f"NAAC Rating: {result.get('naac_rating', 'N/A')}\n"
            context_text += f"Relevance Score: {result.get('similarity_score', 0):.3f}\n\n"
        
        context_text += "Use this verified data to provide accurate recommendations.\n"
        context_text += "Remember: Lower rank numbers are better (Rank 1 is best).\n\n"
        
        return context_text
    
    def call_llm_api(self, 
                     messages: List[Dict], 
                     model: str = None, 
                     stream: bool = True) -> Generator[str, None, None]:
        """Call LLM API with streaming support"""
        if not self.openrouter_api_key:
            yield "Error: OpenRouter API key not configured"
            return
        
        if not model:
            model = self.default_model
        
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://cet-mentor.com",
            "X-Title": "CET-Mentor v2.0"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=stream,
                timeout=30
            )
            
            if not response.ok:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                yield f"Error: Failed to get response from AI model"
                return
            
            if stream:
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                json_data = json.loads(data)
                                if 'choices' in json_data and json_data['choices']:
                                    delta = json_data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
            else:
                json_response = response.json()
                if 'choices' in json_response and json_response['choices']:
                    yield json_response['choices'][0]['message']['content']
        
        except requests.exceptions.Timeout:
            logger.error("LLM API timeout")
            yield "Error: Request timed out. Please try again."
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API request error: {e}")
            yield "Error: Failed to connect to AI service."
        except Exception as e:
            logger.error(f"Unexpected error in LLM call: {e}")
            yield "Error: An unexpected error occurred."
    
    def chat_with_rag(self, user_query: str, chat_history: List[Dict] = None) -> Generator[str, None, None]:
        """
        Handle chat with RAG pipeline
        
        Args:
            user_query: User's question
            chat_history: Previous chat messages
            
        Yields:
            Streaming response chunks
        """
        if not chat_history:
            chat_history = []
        
        # Step 1: Try to retrieve relevant context
        logger.info(f"Processing query: {user_query}")
        context_results = self.retrieve_context(user_query)
        
        # Step 2: Prepare messages for LLM
        messages = []
        
        # Add system prompt
        if context_results:
            # RAG mode with context
            system_prompt = self.rag_system_prompt
            context_text = self.format_context_for_llm(context_results)
            messages.append({
                "role": "system", 
                "content": f"{system_prompt}\n\n{context_text}"
            })
            logger.info(f"Using RAG mode with {len(context_results)} context results")
        else:
            # Conversational mode without context
            messages.append({
                "role": "system", 
                "content": self.conversational_prompt
            })
            logger.info("Using conversational mode (no relevant context found)")
        
        # Add chat history
        for msg in chat_history[-6:]:  # Keep last 6 messages for context
            messages.append(msg)
        
        # Add current user query
        messages.append({"role": "user", "content": user_query})
        
        # Step 3: Call LLM with streaming
        try:
            for chunk in self.call_llm_api(messages, stream=True):
                yield chunk
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            yield "I apologize, but I encountered an error while processing your question. Please try again."

# Initialize AI system
ai_system = CETMentorAI()

@app.route('/')
def index():
    """Serve the main frontend"""
    return render_template('index.html')

@app.route('/suggest', methods=['POST'])
def suggest_colleges():
    """
    Rank-based college suggestion endpoint
    
    Expected JSON input:
    {
        "rank": 12000,
        "category": "Open",
        "branch": "Computer Science",
        "city": "Mumbai"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'rank' not in data:
            return jsonify({'error': 'Rank is required'}), 400
        
        user_rank = int(data['rank'])
        category = data.get('category', 'Open')
        branch_filter = data.get('branch')
        city_filter = data.get('city')
        
        # Validate inputs
        if user_rank <= 0:
            return jsonify({'error': 'Invalid rank provided'}), 400
        
        # Get suggestions
        suggestions = ai_system.get_rank_based_suggestions(
            user_rank=user_rank,
            category=category,
            branch_filter=branch_filter,
            city_filter=city_filter
        )
        
        # Prepare response
        response_data = {
            'user_rank': user_rank,
            'category': category,
            'filters': {
                'branch': branch_filter,
                'city': city_filter
            },
            'safe_options': suggestions['safe_options'],
            'ambitious_options': suggestions['ambitious_options'],
            'total_safe': len(suggestions['safe_options']),
            'total_ambitious': len(suggestions['ambitious_options']),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Served suggestions for rank {user_rank}, category {category}")
        return jsonify(response_data)
        
    except ValueError as e:
        return jsonify({'error': 'Invalid rank format'}), 400
    except Exception as e:
        logger.error(f"Error in suggest endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    RAG-enhanced chat endpoint with streaming
    
    Expected JSON input:
    {
        "message": "What are good colleges for computer science with rank 5000?",
        "session_id": "optional-session-id"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        user_message = data['message'].strip()
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get chat history from session
        if 'chat_history' not in session:
            session['chat_history'] = {}
        
        if session_id not in session['chat_history']:
            session['chat_history'][session_id] = []
        
        chat_history = session['chat_history'][session_id]
        
        # Stream response
        def generate_response():
            try:
                response_content = ""
                for chunk in ai_system.chat_with_rag(user_message, chat_history):
                    response_content += chunk
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                # Save to chat history
                chat_history.append({"role": "user", "content": user_message})
                chat_history.append({"role": "assistant", "content": response_content})
                
                # Keep only last 10 exchanges
                if len(chat_history) > 20:
                    session['chat_history'][session_id] = chat_history[-20:]
                
                yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming response: {e}")
                yield f"data: {json.dumps({'error': 'An error occurred while processing your request'})}\n\n"
        
        return Response(
            generate_response(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Log user feedback to CSV
    
    Expected JSON input:
    {
        "rating": 5,
        "comment": "Very helpful suggestions",
        "query": "original user query",
        "session_id": "session-id"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Feedback data is required'}), 400
        
        # Prepare feedback record
        feedback_record = {
            'timestamp': datetime.now().isoformat(),
            'session_id': data.get('session_id', ''),
            'rating': data.get('rating', ''),
            'comment': data.get('comment', ''),
            'query': data.get('query', ''),
            'user_agent': request.headers.get('User-Agent', ''),
            'ip_address': request.remote_addr
        }
        
        # Save to CSV
        feedback_file = 'feedback.csv'
        file_exists = os.path.exists(feedback_file)
        
        with open(feedback_file, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'session_id', 'rating', 'comment', 'query', 'user_agent', 'ip_address']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(feedback_record)
        
        logger.info(f"Feedback recorded: Rating {feedback_record['rating']}")
        return jsonify({'message': 'Feedback recorded successfully'})
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        return jsonify({'error': 'Failed to record feedback'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'data_loaded': len(ai_system.college_data) > 0,
            'vector_store': ai_system.vector_store is not None,
            'api_configured': ai_system.openrouter_api_key is not None
        }
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        stats = {
            'total_colleges': len(set(college.get('college', '') for college in ai_system.college_data)),
            'total_records': len(ai_system.college_data),
            'categories': list(set(college.get('category', '') for college in ai_system.college_data)),
            'cities': list(set(college.get('city', '') for college in ai_system.college_data)),
            'branches': list(set(college.get('branch', '') for college in ai_system.college_data)),
            'vector_store_stats': ai_system.vector_store.get_statistics() if ai_system.vector_store else {}
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': 'Failed to get statistics'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Create a simple index.html if it doesn't exist
    index_html_path = 'templates/index.html'
    if not os.path.exists(index_html_path):
        with open(index_html_path, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>CET-Mentor v2.0</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
    <h1>CET-Mentor v2.0</h1>
    <p>AI-powered MHT-CET college recommendation system</p>
    <p>API endpoints available:</p>
    <ul>
        <li><strong>POST /suggest</strong> - Get rank-based college suggestions</li>
        <li><strong>POST /chat</strong> - Chat with AI counselor</li>
        <li><strong>POST /feedback</strong> - Submit feedback</li>
        <li><strong>GET /health</strong> - Health check</li>
        <li><strong>GET /stats</strong> - System statistics</li>
    </ul>
</body>
</html>
            """)
    
    # Get configuration from environment
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('FLASK_PORT', 5000))
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    
    logger.info(f"Starting CET-Mentor v2.0 on {host}:{port}")
    app.run(debug=debug_mode, host=host, port=port, threaded=True)
