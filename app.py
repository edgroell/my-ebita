"""
My EBITA* - An AI Financial Sidekick
*EBITA: Earnings Beat Indicator & Text Analyzer
by Ed Groell
Latest: 10-NOV-2025
"""

import os
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import datetime, timezone, timedelta
from typing import Any, Dict
import logging
from logging.handlers import RotatingFileHandler
from functools import wraps
import concurrent.futures
import time
from threading import Thread
from flask_swagger_ui import get_swaggerui_blueprint

# Importing database and data manager
from data.data_models import db, EarningsCallTranscript, User
from data.data_manager import DataManager

# Import AI components
from services_ai.rag_manager import RAGManager
from services_ai.vector_store import ChromaVectorStore
from services_ai.agentic_bot import AgenticBot

# Import API services
from services_api.ninjas_service import NinjasService
from services_api.chatgpt_service import ChatGPTService
from services_api.gemini_service import GeminiService
from services_api.groq_service import GroqService

from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# Default model configurations
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_CHATGPT_MODEL = "gpt-4o-mini"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Flask Configuration ---
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
basedir = os.path.abspath(os.path.dirname(__file__))

# Session configuration for auto-logout
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)  # Auto-logout after 1 hour of inactivity
app.config['SESSION_COOKIE_SECURE'] = not app.debug  # HTTPS only in production
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection

# Ensure database directory exists
db_dir = os.path.join(basedir, 'data', 'transcripts_db')
os.makedirs(db_dir, exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(db_dir, 'transcripts.sqlite3')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Logging Configuration ---
if not app.debug:
    # Create logs directory
    log_dir = os.path.join(basedir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup file handler with rotation
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'ebita.log'),
        maxBytes=10240000,  # 10MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    
    app.logger.setLevel(logging.INFO)
    app.logger.info('EBITA startup')
else:
    # Console logging in debug mode
    app.logger.setLevel(logging.DEBUG)
    app.logger.info('EBITA startup (DEBUG mode)')

# Initialize extensions
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' #type: ignore
login_manager.login_message = "Please log in to access this page." #type: ignore

# --- Rate Limiting Configuration ---
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
    strategy="fixed-window"
)

# --- Initialize Core Services ---
data_manager = DataManager(db)

# Initialize API Services
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
API_NINJAS_KEY = os.environ.get("API_NINJAS_KEY")

# Create ChatGPT service first (used by both RAG and analysis)
chatgpt_service = ChatGPTService(api_key=OPENAI_API_KEY, model_name=DEFAULT_CHATGPT_MODEL) if OPENAI_API_KEY else None

# Initialize other services
ninjas_service = NinjasService(api_key=API_NINJAS_KEY) if API_NINJAS_KEY else None
gemini_service = GeminiService(api_key=GEMINI_API_KEY, model_name=DEFAULT_GEMINI_MODEL) if GEMINI_API_KEY else None
groq_service = GroqService(api_key=GROQ_API_KEY, model_name=DEFAULT_GROQ_MODEL) if GROQ_API_KEY else None

# Initialize Vector Store and RAG Manager (reusing chatgpt_service)
vector_store = None
rag_manager = None
try:
    vector_store = ChromaVectorStore()
    rag_manager = RAGManager(
        vector_store=vector_store,
        chatgpt_service=chatgpt_service
    )
    app.logger.info("RAG Manager initialized successfully")
except Exception as e:
    app.logger.error(f"Failed to initialize RAG Manager: {e}")

# Dictionary to store active bots per transcript
active_bots: Dict[int, AgenticBot] = {}

# --- Flask-Login User Loader ---
@login_manager.user_loader
def load_user(user_id):
    return data_manager.get_user_by_id(int(user_id))

# --- Context Processor ---
@app.context_processor
def inject_global_variables():
    return dict(
        datetime=datetime,
        current_user=current_user,
        data_manager=data_manager,
        rag_enabled=rag_manager is not None
    )

# --- Helper Functions ---
def _extract_transcript_text(transcript_raw: Any) -> str:
    """Extract text from transcript_raw"""
    if transcript_raw is None:
        return ""
    if isinstance(transcript_raw, str):
        return transcript_raw
    if isinstance(transcript_raw, dict):
        return str(transcript_raw.get("text") or transcript_raw.get("transcript") or "")
    return str(transcript_raw)

def _index_transcript_to_rag(transcript: EarningsCallTranscript) -> bool:
    """Index a transcript into the RAG system with proper metadata"""
    if not rag_manager:
        return False
    
    try:
        transcript_text = _extract_transcript_text(transcript.transcript_raw)
        if not transcript_text:
            app.logger.warning(f"Empty transcript text for ID {transcript.transcript_id}")
            return False
        
        metadata = {
            "ticker_symbol": transcript.ticker_symbol,
            "fiscal_year": transcript.fiscal_year,
            "fiscal_quarter": transcript.fiscal_quarter,
            "call_date": transcript.call_date.isoformat() if transcript.call_date else None
        }
        
        rag_manager.add_texts(
            texts=[transcript_text],
            transcript_id=transcript.transcript_id,
            base_metadata=metadata
        )
        
        app.logger.info(f"Indexed transcript {transcript.transcript_id} to RAG")
        return True
        
    except Exception as e:
        app.logger.error(f"Failed to index transcript {transcript.transcript_id}: {e}")
        return False

@app.route('/api/v1/status')
@limiter.limit("30 per minute")
def status():
    """API status endpoint"""
    app.logger.debug("Status check requested")
    return jsonify({
        "status": "ok",
        "message": "EBITA API is operational",
        "services": {
            "ninjas": ninjas_service is not None,
            "chatgpt": chatgpt_service is not None,
            "gemini": gemini_service is not None,
            "groq": groq_service is not None,
            "rag": rag_manager is not None
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200

# -----------------------------------------------------
# Authentication
# -----------------------------------------------------

@app.route('/')
@limiter.limit("100 per minute")
def index():
    return render_template('index.html')

def _read_body():
    # Prefer form data; fallback to JSON
    if request.form:
        return {k: v for k, v in request.form.items()}
    return (request.get_json(silent=True) or {})

@app.route('/register', methods=['GET', 'POST'])
@limiter.limit("5 per hour")
def register():
    if request.method == 'POST':
        data = _read_body()
        username = (data.get('username') or '').strip()
        email = (data.get('email') or '').strip()
        password = (data.get('password') or '').strip()

        app.logger.info(f"Registration attempt for username: {username}, email: {email}")

        if not username or not email or not password:
            app.logger.warning(f"Registration failed: Missing fields for {username}")
            flash('All fields are required.', 'danger')
            return render_template('register.html')

        if not User.validate_username_format(username):
            app.logger.warning(f"Registration failed: Invalid username format: {username}")
            flash('Username must be 2+ chars; only letters, numbers, underscores, and hyphens.', 'danger')
            return render_template('register.html')

        if data_manager.get_user_by_username(username):
            app.logger.warning(f"Registration failed: Username already exists: {username}")
            flash('Username already exists.', 'danger')
            return render_template('register.html')

        if data_manager.get_user_by_email(email):
            app.logger.warning(f"Registration failed: Email already exists: {email}")
            flash('Email already exists.', 'danger')
            return render_template('register.html')

        user = data_manager.create_user(username, email, password)
        if user:
            app.logger.info(f"User registered successfully: {username} (ID: {user.user_id})")
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            app.logger.error(f"Registration failed: Could not create user {username}")
            flash('Registration failed.', 'danger')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per hour")
def login():
    if request.method == 'POST':
        data = _read_body()
        username = (data.get('username') or data.get('username_or_email') or '').strip()
        password = (data.get('password') or '').strip()
        remember_me = data.get('remember_me', False)

        app.logger.info(f"Login attempt for username: {username}")

        if not username or not password:
            app.logger.warning(f"Login failed: Missing credentials for {username}")
            flash('Username and password are required.', 'danger')
            return render_template('login.html')

        user = data_manager.get_user_by_username(username)
        if user and user.check_password(password):
            # Set session as permanent if remember_me is checked
            login_user(user, remember=remember_me)
            
            # Mark session as permanent to enable timeout
            from flask import session
            if remember_me:
                session.permanent = True
                app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)  # 30 days if "remember me"
            else:
                session.permanent = True
                app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)  # 2 hours otherwise
            
            app.logger.info(f"User logged in successfully: {username} (ID: {user.user_id})")
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            app.logger.warning(f"Login failed: Invalid credentials for {username}")
            flash('Invalid credentials.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
@limiter.limit("20 per hour")
def logout():
    username = current_user.username
    user_id = current_user.user_id
    
    # Clean up user's active bot sessions
    bots_to_remove = []
    for transcript_id, bot in active_bots.items():
        if bot.user_id == str(user_id):
            bots_to_remove.append(transcript_id)
    
    for transcript_id in bots_to_remove:
        del active_bots[transcript_id]
        app.logger.info(f"Cleaned up bot session for transcript {transcript_id}")
    
    logout_user()
    app.logger.info(f"User logged out: {username}, cleaned up {len(bots_to_remove)} bot sessions")
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
@limiter.limit("10 per hour")
def profile():
    """User profile page for updating account information"""
    if request.method == 'POST':
        data = _read_body()
        username = (data.get('username') or '').strip()
        email = (data.get('email') or '').strip()
        current_password = (data.get('current_password') or '').strip()
        new_password = (data.get('new_password') or '').strip()
        confirm_password = (data.get('confirm_password') or '').strip()
        
        app.logger.info(f"Profile update attempt for user: {current_user.username} (ID: {current_user.user_id})")
        
        # Verify current password first
        if not current_user.check_password(current_password):
            app.logger.warning(f"Profile update failed: Incorrect password for {current_user.username}")
            flash('Current password is incorrect.', 'danger')
            return render_template('profile.html')
        
        # Update username if changed
        if username and username != current_user.username:
            if not User.validate_username_format(username):
                app.logger.warning(f"Profile update failed: Invalid username format: {username}")
                flash('Username must be 2+ chars; only letters, numbers, underscores, and hyphens.', 'danger')
                return render_template('profile.html')
            
            if data_manager.get_user_by_username(username):
                app.logger.warning(f"Profile update failed: Username already taken: {username}")
                flash('Username already taken.', 'danger')
                return render_template('profile.html')
            
            old_username = current_user.username
            current_user.username = username
            app.logger.info(f"Username updated: {old_username} -> {username}")
        
        # Update email if changed
        if email and email != current_user.email:
            if data_manager.get_user_by_email(email):
                app.logger.warning(f"Profile update failed: Email already in use: {email}")
                flash('Email already in use.', 'danger')
                return render_template('profile.html')
            
            old_email = current_user.email
            current_user.email = email
            app.logger.info(f"Email updated for {current_user.username}: {old_email} -> {email}")
        
        # Update password if provided
        if new_password:
            if new_password != confirm_password:
                app.logger.warning(f"Profile update failed: Password mismatch for {current_user.username}")
                flash('New passwords do not match.', 'danger')
                return render_template('profile.html')
            
            if len(new_password) < 6:
                app.logger.warning(f"Profile update failed: Password too short for {current_user.username}")
                flash('New password must be at least 6 characters.', 'danger')
                return render_template('profile.html')
            
            current_user.set_password(new_password)
            app.logger.info(f"Password updated for {current_user.username}")
        
        # Save changes
        try:
            db.session.commit()
            app.logger.info(f"Profile updated successfully for {current_user.username}")
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('profile'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Profile update failed for {current_user.username}: {e}")
            flash(f'Error updating profile: {e}', 'danger')
            return render_template('profile.html')
    
    return render_template('profile.html')

# --- Swagger UI Configuration ---
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.yaml'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "My EBITA API Documentation",
        'docExpansion': 'list',
        'defaultModelsExpandDepth': 3,
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# -----------------------------------------------------
# Dashboard
# -----------------------------------------------------

@app.route('/dashboard')
@login_required
@limiter.limit("60 per minute")
def dashboard():
    """User dashboard"""
    app.logger.debug(f"Dashboard accessed by user: {current_user.username} (ID: {current_user.user_id})")
    reports = data_manager.get_reports_for_user(current_user.user_id)
    transcripts = data_manager.get_transcripts_for_user(current_user.user_id)
    companies = data_manager.get_all_companies()
    
    return render_template('dashboard.html',
                         reports=reports[:10],
                         transcripts=transcripts[:20],
                         companies=companies)

# -----------------------------------------------------
# Transcript Acquisition (Step 2-3)
# -----------------------------------------------------

@app.route('/api/v1/acquire_transcript', methods=['POST'])
@login_required
@limiter.limit("10 per hour")
def acquire_transcript():
    """Acquire transcript and automatically index to RAG"""
    if not ninjas_service:
        app.logger.error("Acquire transcript failed: Ninjas service not available")
        return jsonify({"error": "Ninjas service not available"}), 503
    
    ticker = request.form.get('ticker', '').strip().upper()
    year = request.form.get('year', type=int)
    quarter = request.form.get('quarter', type=int)
    
    app.logger.info(f"Transcript acquisition requested by {current_user.username}: {ticker} Q{quarter} {year}")
    
    if not all([ticker, year, quarter]):
        app.logger.warning(f"Acquire transcript failed: Missing parameters")
        flash("Ticker, year, and quarter are required.", 'danger')
        return redirect(url_for('dashboard'))
    
    try:
        # Check/create company
        company = data_manager.get_company_by_ticker(ticker)
        if not company:
            app.logger.info(f"Fetching company profile for {ticker}")
            company_profile = ninjas_service.get_company_profile_basic(ticker)
            if company_profile:
                company = data_manager.add_company(
                    ticker_symbol=company_profile.get('symbol') or "",
                    company_name=company_profile.get('name') or "",
                    logo_url=company_profile.get('logo_url')
                )
                app.logger.info(f"Company created: {ticker}")
        
        # Check if transcript exists
        if year is None or quarter is None:
            flash("Year and quarter are required.", 'danger')
            return redirect(url_for('dashboard'))
        existing = data_manager.get_transcript_by_details(ticker, year, quarter)
        if existing:
            app.logger.info(f"Transcript already exists: {ticker} Q{quarter} {year} (ID: {existing.transcript_id})")
            flash(f"Transcript already exists (ID: {existing.transcript_id})", 'info')
            return redirect(url_for('dashboard'))
        
        # Fetch transcript from Ninjas API
        app.logger.info(f"Fetching transcript from Ninjas API: {ticker} Q{quarter} {year}")
        transcript_data = ninjas_service.get_earnings_transcript(ticker, year, quarter)
        if not transcript_data:
            app.logger.warning(f"Transcript not found: {ticker} Q{quarter} {year}")
            flash(f"Transcript not found for {ticker} Q{quarter} {year}", 'danger')
            return redirect(url_for('dashboard'))
        
        call_date = datetime.strptime(transcript_data['date'], '%Y-%m-%d')
        
        # Save transcript to database
        new_transcript = data_manager.add_transcript(
            user_id=current_user.user_id,
            ticker_symbol=ticker,
            fiscal_year=year,
            fiscal_quarter=quarter,
            call_date=call_date,
            transcript_raw=transcript_data['transcript'],
            transcript_split=transcript_data.get('transcript_split'),
            source_url=f"API-Ninjas:{ticker}-{year}-Q{quarter}"
        )
        
        if not new_transcript:
            app.logger.error(f"Failed to save transcript: {ticker} Q{quarter} {year}")
            flash("Failed to save transcript", 'danger')
            return redirect(url_for('dashboard'))
        
        app.logger.info(f"Transcript saved: {ticker} Q{quarter} {year} (ID: {new_transcript.transcript_id})")
        
        # Step 3: Auto-index to RAG
        if rag_manager:
            indexed = _index_transcript_to_rag(new_transcript)
            if indexed:
                app.logger.info(f"Transcript indexed to RAG: {new_transcript.transcript_id}")
                flash(f"✓ Transcript acquired and indexed! (ID: {new_transcript.transcript_id})", 'success')
            else:
                app.logger.warning(f"Transcript saved but indexing failed: {new_transcript.transcript_id}")
                flash(f"Transcript acquired (ID: {new_transcript.transcript_id}) but indexing failed", 'warning')
        else:
            flash(f"Transcript acquired (ID: {new_transcript.transcript_id})", 'success')
        
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        app.logger.error(f"Error acquiring transcript {ticker} Q{quarter} {year}: {e}", exc_info=True)
        flash(f"Error: {e}", 'danger')
        return redirect(url_for('dashboard'))

@app.route('/transcripts/<int:transcript_id>')
@login_required
@limiter.limit("30 per minute")
def view_transcript(transcript_id):
    """View transcript details"""
    app.logger.debug(f"Transcript view requested: {transcript_id} by {current_user.username}")
    transcript = data_manager.get_transcript_by_id(transcript_id)
    if not transcript:
        app.logger.warning(f"Transcript not found: {transcript_id}")
        flash("Transcript not found", 'danger')
        return redirect(url_for('dashboard'))
    
    return render_template('transcript_view.html', transcript=transcript)

@app.route('/api/v1/transcripts/<int:transcript_id>', methods=['GET'])
@login_required
@limiter.limit("30 per minute")
def get_transcript(transcript_id):
    """Get transcript data as JSON (API endpoint)"""
    app.logger.debug(f"Transcript API request: {transcript_id} by {current_user.username}")
    transcript = data_manager.get_transcript_by_id(transcript_id)
    if not transcript:
        app.logger.warning(f"Transcript not found: {transcript_id}")
        return jsonify({"error": "Transcript not found"}), 404
    
    return jsonify({
        "transcript_id": transcript.transcript_id,
        "ticker_symbol": transcript.ticker_symbol,
        "fiscal_year": transcript.fiscal_year,
        "fiscal_quarter": transcript.fiscal_quarter,
        "call_date": transcript.call_date.isoformat() if transcript.call_date else None,
        "transcript_raw": _extract_transcript_text(transcript.transcript_raw),
        "source_url": transcript.source_url
    }), 200

@app.route('/api/v1/transcripts/<int:transcript_id>/delete', methods=['POST'])
@login_required
@limiter.limit("10 per hour")
def delete_transcript(transcript_id):
    """Delete a transcript and its reports"""
    app.logger.info(f"Transcript deletion requested: {transcript_id} by {current_user.username}")
    transcript = data_manager.get_transcript_by_id(transcript_id)
    if not transcript:
        app.logger.warning(f"Transcript not found for deletion: {transcript_id}")
        flash("Transcript not found", 'danger')
        return redirect(url_for('dashboard'))
    
    if transcript.user_id != current_user.user_id:
        app.logger.warning(f"Unauthorized transcript deletion attempt: {transcript_id} by {current_user.username}")
        flash("Unauthorized", 'danger')
        return redirect(url_for('dashboard'))
    
    try:
        data_manager.delete_transcript(transcript_id)
        app.logger.info(f"Transcript deleted: {transcript_id}")
        flash(f"Transcript {transcript_id} deleted successfully", 'success')
    except Exception as e:
        app.logger.error(f"Error deleting transcript {transcript_id}: {e}")
        flash(f"Error deleting transcript: {e}", 'danger')
    
    return redirect(url_for('dashboard'))

@app.route('/api/v1/reports/<int:report_id>/delete', methods=['POST'])
@login_required
@limiter.limit("10 per hour")
def delete_analysis_report(report_id):
    """Delete an analysis report"""
    app.logger.info(f"Report deletion requested: {report_id} by {current_user.username}")
    report = data_manager.get_report_by_id(report_id)
    if not report:
        app.logger.warning(f"Report not found for deletion: {report_id}")
        flash("Report not found", 'danger')
        return redirect(url_for('dashboard'))
    
    if report.user_id != current_user.user_id:
        app.logger.warning(f"Unauthorized report deletion attempt: {report_id} by {current_user.username}")
        flash("Unauthorized", 'danger')
        return redirect(url_for('dashboard'))
    
    try:
        data_manager.delete_analysis_report(report_id)
        app.logger.info(f"Report deleted: {report_id}")
        flash(f"Report {report_id} deleted successfully", 'success')
    except Exception as e:
        app.logger.error(f"Error deleting report {report_id}: {e}")
        flash(f"Error deleting report: {e}", 'danger')
    
    return redirect(url_for('dashboard'))

# -----------------------------------------------------
# AI Analysis (Step 4)
# -----------------------------------------------------

@app.route('/api/v1/analysis', methods=['POST'])
@login_required
@limiter.limit("5 per hour")
def request_analysis():
    """Run AI analysis on transcript using all three services (in parallel)"""
    transcript_id = request.form.get('transcript_id', type=int)
    
    app.logger.info(f"AI analysis requested for transcript {transcript_id} by {current_user.username}")
    
    if not transcript_id:
        app.logger.warning("Analysis request missing transcript_id")
        flash("Transcript ID is required.", 'danger')
        return redirect(url_for('dashboard'))
    
    transcript = data_manager.get_transcript_by_id(transcript_id)
    if not transcript:
        app.logger.warning(f"Transcript not found for analysis: {transcript_id}")
        flash("Transcript not found", 'danger')
        return redirect(url_for('dashboard'))
    
    transcript_text = _extract_transcript_text(transcript.transcript_raw)
    
    # Define analysis functions for each service
    def analyze_with_chatgpt():
        if not chatgpt_service:
            return None
        try:
            app.logger.info(f"Running ChatGPT analysis for transcript {transcript_id}")
            result = chatgpt_service.analyze_transcript(transcript_text)
            app.logger.info(f"ChatGPT analysis complete for transcript {transcript_id}")
            return {
                'success': True,
                'data': result.structured_output.model_dump() if hasattr(result, 'structured_output') else {},
                'metrics': {
                    'request_time_ms': result.request_time_ms,
                    'model': result.model
                }
            }
        except Exception as e:
            app.logger.error(f"ChatGPT analysis failed for transcript {transcript_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    def analyze_with_gemini():
        if not gemini_service:
            return None
        try:
            app.logger.info(f"Running Gemini analysis for transcript {transcript_id}")
            result = gemini_service.analyze_transcript(transcript_text)
            app.logger.info(f"Gemini analysis complete for transcript {transcript_id}")
            return {
                'success': True,
                'data': result.structured_output.model_dump() if hasattr(result, 'structured_output') else {},
                'metrics': {
                    'request_time_ms': result.request_time_ms,
                    'model': result.model
                }
            }
        except Exception as e:
            app.logger.error(f"Gemini analysis failed for transcript {transcript_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    def analyze_with_groq():
        if not groq_service:
            return None
        try:
            app.logger.info(f"Running Groq analysis for transcript {transcript_id}")
            result = groq_service.analyze_transcript(transcript_text)
            app.logger.info(f"Groq analysis complete for transcript {transcript_id}")
            return {
                'success': True,
                'data': result.structured_output.model_dump() if hasattr(result, 'structured_output') else {},
                'metrics': {
                    'request_time_ms': result.request_time_ms,
                    'model': result.model
                }
            }
        except Exception as e:
            app.logger.error(f"Groq analysis failed for transcript {transcript_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    # Run all three analyses in parallel using ThreadPoolExecutor
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_chatgpt = executor.submit(analyze_with_chatgpt)
        future_gemini = executor.submit(analyze_with_gemini)
        future_groq = executor.submit(analyze_with_groq)
        
        # Wait for all to complete and collect results
        chatgpt_result = future_chatgpt.result()
        gemini_result = future_gemini.result()
        groq_result = future_groq.result()
        
        if chatgpt_result:
            results['chatgpt'] = chatgpt_result
        if gemini_result:
            results['gemini'] = gemini_result
        if groq_result:
            results['groq'] = groq_result
    
    app.logger.info(f"All parallel analyses complete for transcript {transcript_id}")
    
    # Helper function to extract nested fields
    def get_field(service_results, field, default: Any = ""):
        """Extract field from service results, checking both flat and nested structures"""
        if not service_results or not service_results.get('success') or 'data' not in service_results:
            return default
        
        data = service_results['data']
        
        # Try direct field access first
        if field in data:
            return data[field]
        
        # Try nested in key_metrics
        if 'key_metrics' in data and field in data['key_metrics']:
            return data['key_metrics'][field]
        
        # Special handling for score fields
        if field in ['management_confidence_score', 'evasiveness_score_q_a']:
            # Check key_metrics first
            if 'key_metrics' in data:
                return data['key_metrics'].get(field, default)
        
        return default
    
    report_data = {        
        # ChatGPT fields
        'chatgpt_summary': get_field(results.get('chatgpt'), 'summary'),
        'chatgpt_concise_rationale': get_field(results.get('chatgpt'), 'concise_rationale'),
        'chatgpt_overall_sentiment': get_field(results.get('chatgpt'), 'overall_sentiment', 'Neutral'),
        'chatgpt_management_confidence_score': float(get_field(results.get('chatgpt'), 'management_confidence_score', 0.0)),
        'chatgpt_evasiveness_score_q_a': float(get_field(results.get('chatgpt'), 'evasiveness_score_q_a', 0.0)),
        'chatgpt_key_topics_discussed': get_field(results.get('chatgpt'), 'key_topics', []),
        'chatgpt_red_flags_identified': get_field(results.get('chatgpt'), 'red_flags', []),
        'chatgpt_raw_response_json': results.get('chatgpt', {}),
        'chatgpt_request_ms': results.get('chatgpt', {}).get('metrics', {}).get('request_time_ms') if results.get('chatgpt') else None,
        
        # Gemini fields
        'gemini_summary': get_field(results.get('gemini'), 'summary'),
        'gemini_concise_rationale': get_field(results.get('gemini'), 'concise_rationale'),
        'gemini_overall_sentiment': get_field(results.get('gemini'), 'overall_sentiment', 'Neutral'),
        'gemini_management_confidence_score': float(get_field(results.get('gemini'), 'management_confidence_score', 0.0)),
        'gemini_evasiveness_score_q_a': float(get_field(results.get('gemini'), 'evasiveness_score_q_a', 0.0)),
        'gemini_key_topics_discussed': get_field(results.get('gemini'), 'key_topics', []),
        'gemini_red_flags_identified': get_field(results.get('gemini'), 'red_flags', []),
        'gemini_raw_response_json': results.get('gemini', {}),
        'gemini_request_ms': results.get('gemini', {}).get('metrics', {}).get('request_time_ms') if results.get('gemini') else None,

        # Groq fields
        'groq_summary': get_field(results.get('groq'), 'summary'),
        'groq_concise_rationale': get_field(results.get('groq'), 'concise_rationale'),
        'groq_overall_sentiment': get_field(results.get('groq'), 'overall_sentiment', 'Neutral'),
        'groq_management_confidence_score': float(get_field(results.get('groq'), 'management_confidence_score', 0.0)),
        'groq_evasiveness_score_q_a': float(get_field(results.get('groq'), 'evasiveness_score_q_a', 0.0)),
        'groq_key_topics_discussed': get_field(results.get('groq'), 'key_topics', []),
        'groq_red_flags_identified': get_field(results.get('groq'), 'red_flags', []),
        'groq_raw_response_json': results.get('groq', {}),
        'groq_request_ms': results.get('groq', {}).get('metrics', {}).get('request_time_ms') if results.get('groq') else None,
    }
    
    # Debug logging
    app.logger.info(f"[DEBUG] ChatGPT confidence: {report_data['chatgpt_management_confidence_score']}")
    app.logger.info(f"[DEBUG] Gemini confidence: {report_data['gemini_management_confidence_score']}")
    app.logger.info(f"[DEBUG] Groq confidence: {report_data['groq_management_confidence_score']}")
    
    new_report = data_manager.create_analysis_report(
        user_id=current_user.user_id,
        transcript_id=transcript_id,
        **report_data
    )
    
    if new_report:
        app.logger.info(f"Analysis report created: {new_report.report_id} for transcript {transcript_id}")
        flash(f"✓ Analysis complete! Report #{new_report.report_id} created.", 'success')
        return redirect(url_for('dashboard'))
    else:
        app.logger.error(f"Failed to save analysis report for transcript {transcript_id}")
        flash("Failed to save analysis report", 'danger')
        return redirect(url_for('dashboard'))

@app.route('/analysis/<int:report_id>')
@login_required
@limiter.limit("30 per minute")
def view_analysis_report(report_id):
    """View analysis report (Step 4 - display results)"""
    app.logger.debug(f"Analysis report view requested: {report_id} by {current_user.username}")
    report = data_manager.get_report_by_id(report_id)
    
    if not report:
        app.logger.warning(f"Report not found: {report_id}")
        flash("Report not found", 'danger')
        return redirect(url_for('dashboard'))
    
    if report.user_id != current_user.user_id:
        app.logger.warning(f"Unauthorized report access attempt: {report_id} by {current_user.username}")
        flash("Unauthorized", 'danger')
        return redirect(url_for('dashboard'))
    
    return render_template('analysis_report.html', report=report)

@app.route('/api/v1/reports/<int:report_id>/raw/<ai_name>')
@login_required
@limiter.limit("30 per minute")
def view_raw_analysis_json(report_id, ai_name):
    """View raw JSON data for a specific AI service's analysis"""
    app.logger.debug(f"Raw JSON requested: report {report_id}, AI {ai_name} by {current_user.username}")
    report = data_manager.get_report_by_id(report_id)
    
    if not report:
        app.logger.warning(f"Report not found for raw JSON: {report_id}")
        return jsonify({"error": "Report not found"}), 404
    
    if report.user_id != current_user.user_id:
        app.logger.warning(f"Unauthorized raw JSON access: report {report_id} by {current_user.username}")
        return jsonify({"error": "Unauthorized"}), 403
    
    # Get the appropriate raw response based on ai_name
    if ai_name == 'chatgpt':
        raw_data = report.chatgpt_raw_response_json
    elif ai_name == 'gemini':
        raw_data = report.gemini_raw_response_json
    elif ai_name == 'groq':
        raw_data = report.groq_raw_response_json
    else:
        app.logger.warning(f"Invalid AI service name requested: {ai_name}")
        return jsonify({"error": "Invalid AI service name"}), 400
    
    return jsonify(raw_data), 200, {'Content-Type': 'application/json'}

@app.route('/api/v1/reports/<int:report_id>/download')
@login_required
@limiter.limit("20 per hour")
def download_analysis_report(report_id):
    """Download analysis report as JSON"""
    app.logger.info(f"JSON download requested: report {report_id} by {current_user.username}")
    report = data_manager.get_report_by_id(report_id)
    
    if not report:
        app.logger.warning(f"Report not found for download: {report_id}")
        flash('Report not found.', 'danger')
        return redirect(url_for('dashboard'))
    
    if report.user_id != current_user.user_id:
        app.logger.warning(f"Unauthorized download attempt: report {report_id} by {current_user.username}")
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('dashboard'))
    
    # Prepare report data for download
    report_data = {
        'report_id': report.report_id,
        'company': {
            'name': report.transcript.company.company_name if report.transcript.company else report.transcript.ticker_symbol,
            'ticker': report.transcript.ticker_symbol,
            'logo_url': report.transcript.company.logo_url if report.transcript.company else None
        },
        'transcript': {
            'fiscal_year': report.transcript.fiscal_year,
            'fiscal_quarter': report.transcript.fiscal_quarter,
            'call_date': report.transcript.call_date.isoformat() if report.transcript.call_date else None,
        },
        'analysis_date': report.analysis_date.isoformat() if report.analysis_date else None,
        'gemini': {
            'summary': report.gemini_summary,
            'concise_rationale': report.gemini_concise_rationale,
            'overall_sentiment': report.gemini_overall_sentiment,
            'management_confidence_score': report.gemini_management_confidence_score,
            'evasiveness_score_q_a': report.gemini_evasiveness_score_q_a,
            'key_topics_discussed': report.gemini_key_topics_discussed,
            'red_flags_identified': report.gemini_red_flags_identified,
            'request_ms': report.gemini_request_ms,
        },
        'chatgpt': {
            'summary': report.chatgpt_summary,
            'concise_rationale': report.chatgpt_concise_rationale,
            'overall_sentiment': report.chatgpt_overall_sentiment,
            'management_confidence_score': report.chatgpt_management_confidence_score,
            'evasiveness_score_q_a': report.chatgpt_evasiveness_score_q_a,
            'key_topics_discussed': report.chatgpt_key_topics_discussed,
            'red_flags_identified': report.chatgpt_red_flags_identified,
            'request_ms': report.chatgpt_request_ms,
        },
        'groq': {
            'summary': report.groq_summary,
            'concise_rationale': report.groq_concise_rationale,
            'overall_sentiment': report.groq_overall_sentiment,
            'management_confidence_score': report.groq_management_confidence_score,
            'evasiveness_score_q_a': report.groq_evasiveness_score_q_a,
            'key_topics_discussed': report.groq_key_topics_discussed,
            'red_flags_identified': report.groq_red_flags_identified,
            'request_ms': report.groq_request_ms,
        }
    }
    
    # Create filename
    company_name = report.transcript.company.company_name if report.transcript.company else report.transcript.ticker_symbol
    filename = f"{company_name.replace(' ', '_')}_Q{report.transcript.fiscal_quarter}_{report.transcript.fiscal_year}_Analysis.json"
    
    app.logger.info(f"JSON downloaded: report {report_id} by {current_user.username}")
    
    # Return as downloadable JSON
    response = jsonify(report_data)
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    response.headers['Content-Type'] = 'application/json'
    
    return response

# -----------------------------------------------------
# Agentic Bot Chat (Step 5)
# -----------------------------------------------------

def _get_or_create_bot(transcript_id: int) -> AgenticBot:
    """Get or create a bot instance for the given transcript ID."""
    if transcript_id in active_bots:
        app.logger.debug(f"Reusing existing bot for transcript {transcript_id}")
        return active_bots[transcript_id]
    
    try:
        if not rag_manager:
            app.logger.error("RAG Manager not initialized")
            raise RuntimeError("RAG Manager not initialized")
        
        # Get transcript to verify it exists
        transcript = data_manager.get_transcript_by_id(transcript_id)
        if not transcript:
            app.logger.error(f"Transcript {transcript_id} not found in database")
            raise ValueError(f"Transcript {transcript_id} not found")
        
        app.logger.info(f"Creating new bot for transcript {transcript_id}")
        
        # Create new bot instance with RAG manager
        bot = AgenticBot(
            rag_manager=rag_manager,
            transcript_id=transcript_id,
            openai_model=DEFAULT_CHATGPT_MODEL,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            google_cse_id=os.environ.get("GOOGLE_CSE_ID"),
            user_id=str(current_user.user_id) if current_user.is_authenticated else None
        )
        
        # Cache the bot instance
        active_bots[transcript_id] = bot
        app.logger.info(f"Bot created and cached for transcript {transcript_id}")
        
        return bot
        
    except Exception as e:
        app.logger.error(f"Failed to create bot for transcript {transcript_id}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create chatbot: {str(e)}") from e

@app.route('/api/v1/chat', methods=['POST'])
@login_required
@limiter.limit("20 per minute")
def chat_with_bot():
    """Chat with an earnings call transcript using the agentic bot"""
    try:
        data = request.get_json()
        transcript_id = data.get('transcript_id')
        message = data.get('message', '').strip()
        
        app.logger.info(f"Chat request received - transcript_id: {transcript_id}, message length: {len(message) if message else 0}")
        
        if not transcript_id or not message:
            app.logger.warning(f"Invalid request - transcript_id: {transcript_id}, message: {bool(message)}")
            return jsonify({'error': 'Missing transcript_id or message'}), 400
        
        # Verify transcript exists
        transcript = data_manager.get_transcript_by_id(transcript_id)
        if not transcript:
            app.logger.warning(f"Transcript not found: {transcript_id}")
            return jsonify({'error': 'Transcript not found'}), 404
        
        # Check if RAG manager is available
        if not rag_manager:
            app.logger.error("RAG Manager not initialized")
            return jsonify({'error': 'RAG system not available. Please ensure vector database is initialized.'}), 503
        
        app.logger.info(f"Processing chat for transcript {transcript_id}: {message[:50]}...")
        
        # Get or create bot for this transcript
        try:
            bot = _get_or_create_bot(transcript_id)
            app.logger.info(f"Bot acquired for transcript {transcript_id}")
        except Exception as bot_error:
            app.logger.error(f"Failed to create bot for transcript {transcript_id}: {bot_error}", exc_info=True)
            return jsonify({'error': f'Failed to initialize chatbot: {str(bot_error)}'}), 500
        
        # Get response from bot
        try:
            response = bot.invoke(message)
            app.logger.info(f"Chat response generated for transcript {transcript_id}: {len(response)} chars")
        except Exception as invoke_error:
            app.logger.error(f"Bot invoke failed for transcript {transcript_id}: {invoke_error}", exc_info=True)
            return jsonify({'error': f'Failed to generate response: {str(invoke_error)}'}), 500
        
        return jsonify({
            'response': response,
            'transcript_id': transcript_id
        })
        
    except Exception as e:
        app.logger.error(f"Unexpected chat error: {e}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/v1/transcripts/<int:transcript_id>/chat/stream', methods=['POST'])
@login_required
@limiter.limit("30 per hour")
def stream_chat_with_transcript(transcript_id):
    """Stream chat responses (for real-time UI)"""
    app.logger.debug(f"Stream chat requested for transcript {transcript_id} by {current_user.username}")
    transcript = data_manager.get_transcript_by_id(transcript_id)
    if not transcript:
        app.logger.warning(f"Transcript not found for stream chat: {transcript_id}")
        return jsonify({"error": "Transcript not found"}), 404
    
    data = request.get_json()
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    
    def generate():
        try:
            bot = _get_or_create_bot(transcript_id)
            for chunk in bot.stream(user_message):
                yield f"data: {chunk}\n\n"
            app.logger.info(f"Stream chat completed for transcript {transcript_id}")
        except Exception as e:
            app.logger.error(f"Stream chat error for transcript {transcript_id}: {e}")
            yield f"data: Error: {str(e)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/v1/chat/history/<int:transcript_id>', methods=['GET'])
@login_required
@limiter.limit("30 per minute")
def get_chat_history(transcript_id):
    """Download conversation history for a transcript as JSON"""
    try:
        # Verify transcript exists and user has access
        transcript = data_manager.get_transcript_by_id(transcript_id)
        if not transcript:
            app.logger.warning(f"Transcript not found for history download: {transcript_id}")
            return jsonify({'error': 'Transcript not found'}), 404
        
        # Get conversation history
        if transcript_id not in active_bots:
            app.logger.info(f"No conversation history found for transcript {transcript_id}")
            return jsonify({'history': [], 'message': 'No conversation history found'}), 200
        
        bot = active_bots[transcript_id]
        history = bot.get_conversation_history()
        
        # Prepare download data
        download_data = {
            'transcript_id': transcript_id,
            'company': {
                'name': transcript.company.company_name if transcript.company else transcript.ticker_symbol,
                'ticker': transcript.ticker_symbol,
            },
            'fiscal_info': {
                'year': transcript.fiscal_year,
                'quarter': transcript.fiscal_quarter,
                'call_date': transcript.call_date.isoformat() if transcript.call_date else None,
            },
            'conversation': {
                'total_turns': len(history),
                'export_date': datetime.now(timezone.utc).isoformat(),
                'user_id': current_user.user_id,
                'username': current_user.username,
                'turns': history
            }
        }
        
        # Create filename
        company_name = transcript.company.company_name if transcript.company else transcript.ticker_symbol
        filename = f"{company_name.replace(' ', '_')}_Q{transcript.fiscal_quarter}_{transcript.fiscal_year}_Conversation.json"
        
        app.logger.info(f"Chat history downloaded for transcript {transcript_id}: {len(history)} turns by {current_user.username}")
        
        # Return as downloadable JSON
        response = jsonify(download_data)
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        response.headers['Content-Type'] = 'application/json'
        
        return response
        
    except Exception as e:
        app.logger.error(f"Failed to download chat history for transcript {transcript_id}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# -----------------------------------------------------
# Error Handlers
# -----------------------------------------------------

@app.errorhandler(404)
def page_not_found(e):
    app.logger.warning(f"404 error: {request.path}")
    if request.path.startswith('/api/'):
        return jsonify({"error": "Not Found"}), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error(f"500 error: {request.path}", exc_info=True)
    if request.path.startswith('/api/'):
        return jsonify({"error": "Internal Server Error"}), 500
    return render_template('500.html'), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    app.logger.warning(f"Rate limit exceeded: {request.path} by {get_remote_address()}")
    if request.path.startswith('/api/'):
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
    flash("Too many requests. Please slow down.", 'warning')
    return redirect(url_for('dashboard'))

# -----------------------------------------------------
# Automatic cleanup of inactive bot sessions
# -----------------------------------------------------

def cleanup_inactive_bots():
    """Background task to clean up inactive bot sessions"""
    CLEANUP_INTERVAL = 3600  # 1 hour
    MAX_IDLE_TIME = 7200  # 2 hours
    
    while True:
        time.sleep(CLEANUP_INTERVAL)
        try:
            with app.app_context():
                current_time = time.time()
                bots_to_remove = []
                
                for transcript_id, bot in active_bots.items():
                    pass
                
                app.logger.info(f"Bot cleanup task: {len(active_bots)} active sessions")
                
        except Exception as e:
            app.logger.error(f"Bot cleanup task error: {e}")

# Start cleanup thread
if not app.debug:
    cleanup_thread = Thread(target=cleanup_inactive_bots, daemon=True)
    cleanup_thread.start()
    app.logger.info("Bot cleanup thread started")

# -----------------------------------------------------
# Application Startup
# -----------------------------------------------------

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            app.logger.info("Database initialized")
            
            # Create test user
            if not data_manager.get_user_by_username("testuser"):
                data_manager.create_user("testuser", "test@example.com", "password123")
                app.logger.info("Test user created")
            
            app.logger.info("Application ready")
            
        except Exception as e:
            app.logger.error(f"Initialization error: {e}", exc_info=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
