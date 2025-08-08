#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import logging
import time
from datetime import datetime
from typing import Any, Optional, List, Dict

try:
    import pandas as pd  # type: ignore
    HAS_PANDAS = True
except Exception:  # pragma: no cover
    pd = None  # type: ignore
    HAS_PANDAS = False

from flask import Flask, Response, request, jsonify, render_template
from flask_session import Session
from dotenv import load_dotenv
from openai import OpenAI

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cet-mentor")

# --- App Initialization ---
app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "a-strong-default-secret-key")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
Session(app)

# Security headers
@app.after_request
def apply_security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    resp.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    csp = "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com;"
    resp.headers["Content-Security-Policy"] = csp
    return resp

# --- Simple in-memory rate limiting ---
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "60"))        # requests per window
_rate_store = {}

def is_rate_limited(ip: str, key: str) -> bool:
    now = time.time()
    bucket = _rate_store.setdefault((ip, key), [])
    # prune
    cutoff = now - RATE_LIMIT_WINDOW
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)
    if len(bucket) >= RATE_LIMIT_MAX:
        return True
    bucket.append(now)
    return False

# --- OpenAI Client (OpenRouter) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client: Optional[OpenAI] = None
if OPENROUTER_API_KEY:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
else:
    logger.warning("OPENROUTER_API_KEY not provided. /chat will be unavailable.")

# --- Data Loading ---
DATA_PATH = os.getenv("MHT_CET_DATA_PATH", "mht_cet_data.json")

data_rows: List[Dict[str, Any]] = []

try:
    if HAS_PANDAS:
        df = pd.read_json(DATA_PATH)
        if not {'college', 'branch', 'closing_rank'}.issubset(df.columns):
            raise ValueError("Unexpected columns in data file")
        logger.info(f"Loaded {len(df)} records from {DATA_PATH} (pandas mode)")
    else:
        # Fallback: load JSON into list of dicts
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, 'r', encoding='utf-8') as f:
                data_rows = json.load(f)
                if not isinstance(data_rows, list):
                    raise ValueError("Data file not a list of records")
            logger.info(f"Loaded {len(data_rows)} records from {DATA_PATH} (fallback mode)")
        else:
            logger.error(f"Data file not found: {DATA_PATH}")
            df = None  # type: ignore
except Exception as e:
    logger.error(f"Failed to load data file {DATA_PATH}: {e}")
    df = None  # type: ignore
    data_rows = []

# --- Utilities ---

def sanitize_int(value: Any) -> Optional[int]:
    try:
        if isinstance(value, str):
            value = int(value.strip())
        elif isinstance(value, (int, float)):
            value = int(value)
        else:
            return None
        return value if value > 0 else None
    except Exception:
        return None


def data_is_empty() -> bool:
    if HAS_PANDAS:
        return df is None or df.empty  # type: ignore
    return len(data_rows) == 0


def get_system_prompt() -> str:
    return (
        "You are CET-Mentor, a professional AI assistant for MHT-CET engineering admissions. "
        "Work strictly with ranks (lower is better). Prefer verified context from scraped Shiksha data. "
        "Do not discuss IITs/NITs or JEE. Keep scope to MHT-CET colleges and counseling. "
        "Be concise, accurate, and supportive. Use markdown sparingly for clarity."
    )


def build_context(query: str, max_items: int = 10) -> str:
    if data_is_empty():
        return ""
    if HAS_PANDAS:
        try:
            mask = df['college'].str.contains(query, case=False, na=False)  # type: ignore
            subset = df[mask].head(max_items)  # type: ignore
        except Exception:
            return ""
        if subset.empty:
            return ""
        lines = ["### VERIFIED CONTEXT (Shiksha scrape)"]
        for _, row in subset.iterrows():
            closing = int(row['closing_rank']) if pd.notna(row['closing_rank']) else 'NA'  # type: ignore
            lines.append(f"- College: {row['college']} | Branch: {row['branch']} | Closing Rank: {closing}")
        return "\n".join(lines)
    else:
        q = (query or '').lower()
        subset = [r for r in data_rows if isinstance(r, dict) and q in str(r.get('college','')).lower()][:max_items]
        if not subset:
            return ""
        lines = ["### VERIFIED CONTEXT (Shiksha scrape)"]
        for r in subset:
            closing = r.get('closing_rank') if r.get('closing_rank') is not None else 'NA'
            lines.append(f"- College: {r.get('college','')} | Branch: {r.get('branch','')} | Closing Rank: {closing}")
        return "\n".join(lines)


# --- Double-Approval RAG Workflow ---
APPROVAL_INSTRUCTIONS = (
    "You are a strict verifier. Given a user question and an optional VERIFIED CONTEXT, "
    "respond ONLY with compact JSON: {\"approved\": boolean, \"reason\": string}. "
    "Approve only if the context explicitly supports the answer to the question."
)


def ask_approval(question: str, context: str) -> bool:
    if not client:
        return False
    messages = [
        {"role": "system", "content": APPROVAL_INSTRUCTIONS},
        {"role": "system", "content": context or "No context available."},
        {"role": "user", "content": question},
    ]
    try:
        comp = client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        txt = comp.choices[0].message.content
        data = json.loads(txt)
        return bool(data.get("approved", False))
    except Exception as e:
        logger.error(f"Approval call failed: {e}\n{txt if 'txt' in locals() else ''}")
        return False


def stream_answer(question: str, context: str, approved: bool):
    if not client:
        yield f"data: {json.dumps({'error': 'AI client not configured'})}\n\n"
        return

    system_content = get_system_prompt()
    if approved and context:
        system_content += "\nYou MUST ground answers in the VERIFIED CONTEXT provided. If context is insufficient, say so."
        system_context = context
    else:
        system_context = "No verified context found. Answer generally for MHT-CET scope."

    messages = [
        {"role": "system", "content": system_content},
        {"role": "system", "content": system_context},
        {"role": "user", "content": question},
    ]

    try:
        stream = client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=messages,
            stream=True,
            temperature=0.2,
        )
        for chunk in stream:
            try:
                content = chunk.choices[0].delta.content
            except Exception:
                content = None
            if content:
                yield f"data: {json.dumps({'content': content})}\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': 'The AI service is currently unavailable.'})}\n\n"


# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/suggest', methods=['POST'])
def suggest_colleges():
    ip = request.headers.get('X-Forwarded-For', request.remote_addr or 'unknown')
    if is_rate_limited(ip, 'suggest'):
        return jsonify({"error": "Rate limit exceeded. Please try again shortly."}), 429

    payload = request.get_json(silent=True) or {}
    user_rank = sanitize_int(payload.get('rank'))
    if not user_rank:
        return jsonify({"error": "Please provide a valid positive integer rank."}), 400
    if data_is_empty():
        return jsonify({"error": "College data not available. Please run the scraper."}), 503

    if HAS_PANDAS:
        safe_df = df[df['closing_rank'].notna() & (df['closing_rank'] >= user_rank)].sort_values(by='closing_rank').head(10)  # type: ignore
        ambitious_df = df[df['closing_rank'].notna() & (df['closing_rank'] < user_rank) & (df['closing_rank'] >= user_rank - 5000)].sort_values(by='closing_rank', ascending=False).head(10)  # type: ignore
        return jsonify({
            "safe_options": safe_df.to_dict('records'),
            "ambitious_options": ambitious_df.to_dict('records')
        })
    else:
        # Fallback pure-Python
        rows = [r for r in data_rows if isinstance(r, dict) and isinstance(r.get('closing_rank'), int)]
        safe = sorted([r for r in rows if r['closing_rank'] >= user_rank], key=lambda x: x['closing_rank'])[:10]
        ambitious = sorted([r for r in rows if (r['closing_rank'] < user_rank and r['closing_rank'] >= user_rank - 5000)], key=lambda x: x['closing_rank'], reverse=True)[:10]
        return jsonify({
            "safe_options": safe,
            "ambitious_options": ambitious
        })


@app.route('/chat', methods=['POST'])
def chat():
    ip = request.headers.get('X-Forwarded-For', request.remote_addr or 'unknown')
    if is_rate_limited(ip, 'chat'):
        return Response("Rate limit exceeded.", status=429)

    payload = request.get_json(silent=True) or {}
    user_message = payload.get('message', '').strip()
    if not user_message:
        return Response("Empty message.", status=400)

    context = build_context(user_message, max_items=8)
    approved = ask_approval(user_message, context)

    def generate():
        yield from stream_answer(user_message, context, approved)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/feedback', methods=['POST'])
def feedback():
    payload = request.get_json(silent=True) or {}
    try:
        with open('feedback_log.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                payload.get('type'),
                payload.get('message'),
                payload.get('response')
            ])
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"Feedback logging failed: {e}")
        return jsonify({"status": "error"}), 500


@app.route('/healthz')
def healthz():
    status = {
        "ok": True,
        "data_loaded": not data_is_empty(),
        "model": bool(client),
        "pandas": HAS_PANDAS,
    }
    return jsonify(status)


if __name__ == '__main__':
    port = int(os.getenv('PORT', '8000'))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', '0') == '1')