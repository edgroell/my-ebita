# TODO List of Transcripts (in app.py), Comparison Notes, remove TESTING blocks
"""
My EBITA - An AI Financial Sidekick
*EBITA: Earnings Beat Indicator & Text Analyzer
by Ed Groell
Latest: 22-JUL-2025
"""

import os
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, flash, make_response
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from datetime import datetime, date

# Importing database and data manager
from data.data_models import db, User, Company, EarningsCallTranscript, AnalysisReport
from data.data_manager import DataManager

# Importing AI services
from services.gemini_service import GeminiService
from services.chatgpt_service import ChatGPTService
# Importing API Ninjas Service for data acquisition
from services.ninjas_service import NinjasService
from dotenv import load_dotenv

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Load environment variables ---
load_dotenv()

# --- Flask Configuration ---
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'data', 'my_ebita.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Initialize Extensions ---
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
setattr(login_manager, 'login_view', 'login')  # The name of the view function for the login page
setattr(login_manager, 'login_message', "Please log in to access this page.")

# --- Initialize Data Manager and External Services ---
data_manager = DataManager(db) # Initialize DataManager here

# Get API keys from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") # Assuming GOOGLE_API_KEY for Gemini
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
API_NINJAS_KEY = os.environ.get("API_NINJAS_KEY")

# Initialize AI services (handle cases where API keys might be missing)
gemini_service = None
if GEMINI_API_KEY:
    gemini_service = GeminiService(api_key=GEMINI_API_KEY)
else:
    print("Warning: GOOGLE_API_KEY not set. GeminiService will not be available.")

chatgpt_service = None
if OPENAI_API_KEY:
    chatgpt_service = ChatGPTService(api_key=OPENAI_API_KEY)
else:
    print("Warning: OPENAI_API_KEY not set. ChatGPTService will not be available.")

ninjas_service = None
if API_NINJAS_KEY:
    ninjas_service = NinjasService(api_key=API_NINJAS_KEY)
else:
    print("Warning: API_NINJAS_KEY not set. NinjasService will not be available for data fetching.")


# --- Flask-Login User Loader ---
@login_manager.user_loader
def load_user(user_id):
    """
    Required by Flask-Login: Loads a user from the database given their ID.
    """
    return data_manager.get_user_by_id(int(user_id))

# --- Context Processor for Templates ---
@app.context_processor
def inject_global_variables():
    """
    Injects global variables into all templates.
    This makes 'datetime', 'current_user', and 'data_manager' available in templates.
    """
    return dict(datetime=datetime, current_user=current_user, data_manager=data_manager)


# -----------------------------------------------------
# Status
# -----------------------------------------------------

@app.route('/api/v1/status')
def status():
    """API endpoint for checking the status of the EBITA API."""
    return jsonify({"status": "ok", "message": "EBITA API is operational, barely."}), 200

# -----------------------------------------------------
# Homepage
# -----------------------------------------------------

@app.route('/')
def index():
    """Renders the main homepage of the application."""
    return render_template('index.html')

# -----------------------------------------------------
# Authentication & User Management
# -----------------------------------------------------

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Web route for user registration (GET for form, POST for submission)."""
    if current_user.is_authenticated:
        flash('You are already logged in.', 'info')
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not password:
            flash("Username and password are required.", 'danger')
            return render_template('register.html')

        new_user = data_manager.create_user(username, email, password)
        if new_user:
            login_user(new_user)
            flash('Registration successful and you are now logged in!', 'success')
            return redirect(url_for('index'))
        else:
            flash("Registration failed. Username or email might already exist.", 'danger')
    return render_template('register.html')


@app.route('/api/v1/auth/register', methods=['POST'])
def api_register():
    """API endpoint for user registration (JSON request/response)."""
    data = request.get_json() # Expects JSON
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not password:
        return jsonify({"message": "Username and password are required."}), 400

    new_user = data_manager.create_user(username, email, password)
    if new_user:
        login_user(new_user)
        return jsonify({"message": "User registered and logged in successfully!", "user_id": new_user.user_id}), 201
    else:
        return jsonify({"message": "Registration failed. Username or email might already exist."}), 409

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Web route for user login (GET for form, POST for submission)."""
    if current_user.is_authenticated:
        flash('You are already logged in.', 'info')
        return redirect(url_for('index'))

    if request.method == 'POST':
        username_or_email = request.form.get('username_or_email')
        password = request.form.get('password')

        user = data_manager.get_user_by_username(username_or_email)
        if not user:
            user = data_manager.get_user_by_email(username_or_email)

        if user and user.check_password(password) and user.is_active:
            login_user(user)
            data_manager.update_user_profile(user.user_id, last_login_at=datetime.now())
            flash('Logged in successfully!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Login failed. Check your username/email and password, or your account may be inactive.', 'danger')
    return render_template('login.html')

@app.route('/api/v1/auth/login', methods=['POST'])
def api_login():
    """API endpoint for user login (JSON request/response)."""
    data = request.get_json() # Expects JSON
    username_or_email = data.get('username_or_email')
    password = data.get('password')

    user = data_manager.get_user_by_username(username_or_email)
    if not user:
        user = data_manager.get_user_by_email(username_or_email)

    if user and user.check_password(password) and user.is_active:
        login_user(user)
        data_manager.update_user_profile(user.user_id, last_login_at=datetime.now())
        return jsonify({"message": "Login successful!", "user_id": user.user_id}), 200
    else:
        return jsonify({"message": "Invalid credentials or inactive account."}), 401

@app.route('/logout')
@login_required
def logout():
    """Web route for user logout."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/api/v1/auth/logout', methods=['POST'])
@login_required
def api_logout():
    """API endpoint for user logout (JSON response)."""
    logout_user()
    return jsonify({"message": "Logged out successfully."}), 200

@app.route('/api/v1/users/me', methods=['GET'])
@login_required
def get_current_user_profile():
    """API endpoint to retrieve the authenticated user's profile."""
    user_data = {
        "user_id": current_user.user_id,
        "username": current_user.username,
        "email": current_user.email,
        "created_at": current_user.created_at.isoformat(),
        "updated_at": current_user.updated_at.isoformat(),
        "last_login_at": current_user.last_login_at.isoformat() if current_user.last_login_at else None,
        "is_active": current_user.is_active
    }
    return jsonify(user_data), 200

@app.route('/dashboard')
@login_required
def dashboard():
    """Renders the user's main dashboard."""
    return render_template('dashboard.html')

# -----------------------------------------------------
# Companies
# -----------------------------------------------------

@app.route('/api/v1/companies', methods=['GET'])
def list_companies():
    """API endpoint to list all companies with optional search/filter/pagination."""
    search_query = request.args.get('search')
    industry = request.args.get('industry')
    sector = request.args.get('sector')
    limit = request.args.get('limit', type=int)
    offset = request.args.get('offset', type=int)

    companies = data_manager.get_all_companies(
        search_query=search_query,
        industry=industry,
        sector=sector,
        limit=limit,
        offset=offset
    )

    companies_data = [{
        "ticker_symbol": c.ticker_symbol,
        "company_name": c.company_name,
        "industry": c.industry,
        "sector": c.sector,
        "exchange": c.exchange,
        "logo_url": c.logo_url,
        "last_updated": c.last_updated.isoformat() if c.last_updated else None
    } for c in companies]
    return jsonify(companies_data), 200

@app.route('/api/v1/companies/<string:ticker_symbol>', methods=['GET'])
def get_company(ticker_symbol):
    """API endpoint to get details for a specific company."""
    company = data_manager.get_company_by_ticker(ticker_symbol.upper())
    if not company:
        return jsonify({"message": "Company not found."}), 404

    company_data = {
        "ticker_symbol": company.ticker_symbol,
        "company_name": company.company_name,
        "industry": company.industry,
        "sector": company.sector,
        "exchange": company.exchange,
        "logo_url": company.logo_url,
        "last_updated": company.last_updated.isoformat() if company.last_updated else None
    }
    return jsonify(company_data), 200

# -----------------------------------------------------
# Earnings Call Transcripts
# -----------------------------------------------------

@app.route('/api/v1/companies/<string:ticker_symbol>/transcripts', methods=['GET'])
def list_company_transcripts(ticker_symbol):
    """API endpoint to list all transcripts for a given company."""
    transcripts = data_manager.get_transcripts_for_company(ticker_symbol.upper())
    if not transcripts:
        return jsonify({"message": f"No transcripts found for {ticker_symbol}."}), 404

    transcripts_data = [{
        "transcript_id": t.transcript_id,
        "ticker_symbol": t.ticker_symbol,
        "fiscal_year": t.fiscal_year,
        "fiscal_quarter": t.fiscal_quarter,
        "call_date": t.call_date.isoformat(),
        "source_url": t.source_url,
        "fetched_at": t.fetched_at.isoformat()
    } for t in transcripts]
    return jsonify(transcripts_data), 200

@app.route('/api/v1/transcripts/<int:transcript_id>', methods=['GET'])
def get_transcript(transcript_id):
    """API endpoint to retrieve a full earnings call transcript by ID."""
    transcript = data_manager.get_transcript_by_id(transcript_id)
    if not transcript:
        return jsonify({"message": "Transcript not found."}), 404

    transcript_data = {
        "transcript_id": transcript.transcript_id,
        "ticker_symbol": transcript.ticker_symbol,
        "fiscal_year": transcript.fiscal_year,
        "fiscal_quarter": transcript.fiscal_quarter,
        "call_date": transcript.call_date.isoformat(),
        "raw_text": transcript.raw_text,
        "speaker_segments": transcript.speaker_segments,
        "source_url": transcript.source_url,
        "fetched_at": transcript.fetched_at.isoformat()
    }
    return jsonify(transcript_data), 200

@app.route('/transcripts/<int:transcript_id>', methods=['POST'])
@login_required
def delete_transcript(transcript_id):
    """Deletes a transcript and all its associated reports."""
    # First, get the transcript to ensure it exists
    transcript = data_manager.get_transcript_by_id(transcript_id)
    if not transcript:
        flash(f"Transcript ID {transcript_id} not found.", 'danger')
        return redirect(url_for('dashboard'))

    # You could add a check here to ensure the user owns the reports associated
    # with this transcript, but since all reports are tied to the user,
    # deleting the transcript will only affect that user's reports.
    # The cascading delete in the data model will handle this.

    if data_manager.delete_transcript(transcript_id):
        flash(f"Transcript ID {transcript_id} and all associated reports have been deleted.", 'success')
    else:
        flash(f"Failed to delete transcript ID {transcript_id}.", 'danger')

    return redirect(url_for('dashboard'))

# -----------------------------------------------------
# Data Acquisition (using NinjasService)
# -----------------------------------------------------
@app.route('/api/v1/acquire_transcript', methods=['POST'])
@login_required
def acquire_transcript():
    """
    API endpoint to acquire an earnings transcript and company profile using NinjasService.
    """
    if not ninjas_service:
        flash("API Ninjas Service not available (API key missing).", 'danger')
        return redirect(url_for('dashboard'))

    # Changed from request.get_json() to request.form for HTML form submissions
    ticker = request.form.get('ticker')
    year = request.form.get('year', type=int)
    quarter = request.form.get('quarter', type=int)

    # Normalize ticker to a safe uppercase string (avoid calling .upper() on None)
    ticker_norm = (ticker or '').strip().upper() if isinstance(ticker, str) else ''

    if not all([ticker_norm, year, quarter]):
        flash("Ticker, year, and quarter are required.", 'danger')
        return redirect(url_for('dashboard'))

    try:
        # First, trying to get/add the company profile
        company = data_manager.get_company_by_ticker(ticker_norm)
        if not company:
            company_profile = ninjas_service.get_company_profile_basic(ticker_norm)
            if company_profile:
                company = data_manager.add_company(
                    ticker_symbol=company_profile.get('symbol'),
                    company_name=company_profile.get('name'),
                    # API Ninjas logo API doesn't provide industry/sector, so these remain null for now
                    logo_url=company_profile.get('logo_url')
                )
                if not company:
                    # If company could not be added after fetching, flash and redirect
                    flash(f"Warning: Company {ticker_norm} could not be added to database after fetching profile.", 'warning')
                    return redirect(url_for('dashboard'))
            else:
                flash(f"Company profile not found for {ticker_norm} from API Ninjas.", 'danger')
                return redirect(url_for('dashboard'))

        # Second, trying to get the transcript
        # At this point ticker_norm, year, and quarter have been validated above;
        # add an assertion to help static type checkers (and ensure runtime safety).
        assert isinstance(year, int) and isinstance(quarter, int)

        transcript_exists = data_manager.get_transcript_by_details(ticker_norm, year, quarter)
        if transcript_exists:
            flash(f"Transcript for {ticker_norm} Q{quarter} {year} already exists (ID: {transcript_exists.transcript_id}).", 'info')
            return redirect(url_for('dashboard'))

        ninjas_transcript_data = ninjas_service.get_earnings_transcript(ticker_norm, year, quarter)
        if not ninjas_transcript_data:
            flash(f"Transcript not found from API Ninjas for {ticker_norm} Q{quarter} {year}.", 'danger')
            return redirect(url_for('dashboard'))

        # Converting date string to datetime.date object
        call_date = datetime.strptime(ninjas_transcript_data['date'], '%Y-%m-%d').date()

        new_transcript = data_manager.add_transcript(
            ticker_symbol=ticker_norm,
            fiscal_year=year,
            fiscal_quarter=quarter,
            call_date=call_date,
            raw_text=ninjas_transcript_data['transcript'],
            speaker_segments=ninjas_transcript_data.get('transcript_split'),
            source_url=f"API-Ninjas:{ticker_norm}-{year}-Q{quarter}"
        )

        if new_transcript:
            flash(f"Transcript acquired and saved successfully! (ID: {new_transcript.transcript_id})", 'success')
            return redirect(url_for('dashboard'))
        else:
            flash("Failed to save acquired transcript (e.g., integrity error). Consider checking logs.", 'danger')
            return redirect(url_for('dashboard'))

    except ValueError as e:
        flash(f"Input error: {e}", 'danger')
        return redirect(url_for('dashboard'))
    except Exception as e:
        print(f"Error acquiring transcript: {e}")
        flash(f"An error occurred during data acquisition: {e}", 'danger')
        return redirect(url_for('dashboard'))

# -----------------------------------------------------
# Analysis Reports
# -----------------------------------------------------

@app.route('/api/v1/analysis', methods=['POST'])
@login_required
def request_analysis():
    """
    API endpoint to request a new dual AI analysis for an earnings call transcript.
    This performs analysis using both Gemini and ChatGPT.
    """
    transcript_id = request.form.get('transcript_id', type=int)
    # The default prompt has been moved to be a variable for easier management
    default_prompt = """Analyze the transcript. Provide the response as a JSON object with the following keys:
- "summary": A concise summary of the call (string).
- "overall_sentiment": "Positive", "Neutral", or "Negative" (string).
- "management_confidence_score": A score from 0 to 100 for management's confidence (integer).
- "evasiveness_score_q_a": A score from 0 to 100 for evasiveness in Q&A (integer).
- "key_topics": A list of 3-5 main topics discussed (array of strings).
- "red_flags": A list of any specific red flags or evasive phrases identified (array of strings).
"""
    # Use user prompt if provided, otherwise use the default
    analysis_prompt = request.form.get('analysis_prompt', default_prompt).strip()
    if not analysis_prompt:
        analysis_prompt = default_prompt

    comparison_notes = request.form.get('comparison_notes')

    if not transcript_id:
        flash("Transcript ID is required to perform analysis.", 'danger')
        return redirect(url_for('dashboard'))

    transcript = data_manager.get_transcript_by_id(transcript_id)
    if not transcript:
        flash("Transcript not found for analysis.", 'danger')
        return redirect(url_for('dashboard'))

    gemini_analysis_result = {"success": False, "error": "Gemini Service not available."}
    chatgpt_analysis_result = {"success": False, "error": "ChatGPT Service not available."}

    # --- Call Gemini Service ---
    if gemini_service:
        try:
            gemini_analysis_result = gemini_service.analyze_transcript(transcript.raw_text, analysis_prompt)
            print(f"Gemini Analysis Success: {gemini_analysis_result.get('success')}")
        except Exception as e:
            print(f"Error during Gemini analysis: {e}")
            gemini_analysis_result = {"success": False, "error": f"Gemini Service Error: {e}"}
    else:
        print("GeminiService not initialized.")

    # --- Call ChatGPT Service ---
    if chatgpt_service:
        try:
            chatgpt_analysis_result = chatgpt_service.analyze_transcript(transcript.raw_text, analysis_prompt)
            print(f"ChatGPT Analysis Success: {chatgpt_analysis_result.get('success')}")
        except Exception as e:
            print(f"Error during ChatGPT analysis: {e}")
            chatgpt_analysis_result = {"success": False, "error": f"ChatGPT Service Error: {e}"}
    else:
        print("ChatGPTService not initialized.")

    # Preparing data for DataManager based on AI results
    gemini_data_for_db = {}
    if gemini_analysis_result and gemini_analysis_result.get("success"):
        parsed_gemini_output = gemini_analysis_result["analysis"]
        gemini_data_for_db = {
            "gemini_summary": parsed_gemini_output.get("summary", ""),
            "gemini_overall_sentiment": parsed_gemini_output.get("overall_sentiment", "Neutral"),
            "gemini_sentiment_scores_by_segment": parsed_gemini_output.get("sentiment_scores_by_segment"),
            "gemini_management_confidence_score": parsed_gemini_output.get("management_confidence_score"),
            "gemini_evasiveness_score_q_a": parsed_gemini_output.get("evasiveness_score_q_a"),
            "gemini_key_topics_discussed": parsed_gemini_output.get("key_topics"),
            "gemini_red_flags_identified": parsed_gemini_output.get("red_flags"),
            "gemini_raw_response_json": parsed_gemini_output
        }
    else:
        # Ensure that values are JSON serializable if AI failed, e.g., None instead of complex objects
        gemini_data_for_db = {
            "gemini_summary": f"Gemini analysis failed: {gemini_analysis_result.get('error', 'N/A')}",
            "gemini_overall_sentiment": "Error",
            "gemini_sentiment_scores_by_segment": None, # Set to None if not available
            "gemini_management_confidence_score": None,
            "gemini_evasiveness_score_q_a": None,
            "gemini_key_topics_discussed": None,
            "gemini_red_flags_identified": None,
            "gemini_raw_response_json": {"error": gemini_analysis_result.get('error', 'N/A')}
        }

    chatgpt_data_for_db = {}
    if chatgpt_analysis_result and chatgpt_analysis_result.get("success"):
        parsed_chatgpt_output = chatgpt_analysis_result["analysis"]
        chatgpt_data_for_db = {
            "chatgpt_summary": parsed_chatgpt_output.get("summary", ""),
            "chatgpt_overall_sentiment": parsed_chatgpt_output.get("overall_sentiment", "Neutral"),
            "chatgpt_sentiment_scores_by_segment": parsed_chatgpt_output.get("sentiment_scores_by_segment"),
            "chatgpt_management_confidence_score": parsed_chatgpt_output.get("management_confidence_score"),
            "chatgpt_evasiveness_score_q_a": parsed_chatgpt_output.get("evasiveness_score_q_a"),
            "chatgpt_key_topics_discussed": parsed_chatgpt_output.get("key_topics"),
            "chatgpt_red_flags_identified": parsed_chatgpt_output.get("red_flags"),
            "chatgpt_raw_response_json": parsed_chatgpt_output
        }
    else:
        # Ensure that values are JSON serializable if AI failed, e.g., None instead of complex objects
        chatgpt_data_for_db = {
            "chatgpt_summary": f"ChatGPT analysis failed: {chatgpt_analysis_result.get('error', 'N/A')}",
            "chatgpt_overall_sentiment": "Error",
            "chatgpt_sentiment_scores_by_segment": None,
            "chatgpt_management_confidence_score": None,
            "chatgpt_evasiveness_score_q_a": None,
            "chatgpt_key_topics_discussed": None,
            "chatgpt_red_flags_identified": None,
            "chatgpt_raw_response_json": {"error": chatgpt_analysis_result.get('error', 'N/A')}
        }

    # Check for existing report to prevent UniqueConstraint violation
    # The UniqueConstraint includes 'analysis_date', so multiple reports for the same
    # user/transcript are allowed if they occur at different times.
    # However, if you want only ONE report per user/transcript, you need to
    # query for existence without the analysis_date, or update the existing one.
    # For now, we'll allow multiple but ensure the `create_analysis_report` handles it.

    new_report = data_manager.create_analysis_report(
        user_id=current_user.user_id,
        transcript_id=transcript_id,
        # Pass all relevant data from AI analysis results
        **gemini_data_for_db,
        **chatgpt_data_for_db,
        comparison_notes=comparison_notes
    )

    if new_report:
        flash("Dual AI analysis requested and saved successfully!", 'success')
        # Debugging: Print the report ID to confirm creation
        print(f"DEBUG: New analysis report created with ID: {new_report.report_id}")
        return redirect(url_for('dashboard')) # Redirect back to dashboard to see new report
    else:
        # This branch indicates a problem with db.session.add() or db.session.commit()
        flash("Failed to save dual analysis report. Check server logs for details (e.g., IntegrityError).", 'danger')
        print("DEBUG: Failed to create analysis report.") # More specific print
        return redirect(url_for('dashboard'))


@app.route('/api/v1/analysis', methods=['GET'])
@login_required
def list_user_analysis_reports():
    """API endpoint to list all analysis reports for the current user."""
    # This route is usually for API consumption. Your dashboard directly calls data_manager.
    # Leaving it as is, but focusing on the DataManager methods for the dashboard.
    reports = data_manager.get_reports_for_user(current_user.user_id)
    reports_data = []
    for r in reports:
        # Ensure these relationships are eagerly loaded or accessed within app context
        # In data_models.py, 'transcript' and 'company' are lazy-loaded by default,
        # which means accessing them here is fine as long as the session is open.
        transcript = r.transcript # Access relationship directly
        company = r.transcript.company

        # Ensure that `company` and `transcript` objects exist before trying to access their attributes
        ticker_symbol = company.ticker_symbol if company else None
        company_name = company.company_name if company else None
        fiscal_year = transcript.fiscal_year if transcript else None
        fiscal_quarter = transcript.fiscal_quarter if transcript else None
        call_date = transcript.call_date.isoformat() if transcript and transcript.call_date else None


        reports_data.append({
            "report_id": r.report_id,
            "transcript_id": r.transcript_id,
            "ticker_symbol": ticker_symbol,
            "company_name": company_name,
            "fiscal_year": fiscal_year,
            "fiscal_quarter": fiscal_quarter,
            "call_date": call_date,
            "analysis_date": r.analysis_date.isoformat(),

            "gemini_summary": r.gemini_summary,
            "gemini_overall_sentiment": r.gemini_overall_sentiment,
            "gemini_management_confidence_score": r.gemini_management_confidence_score,
            "gemini_evasiveness_score_q_a": r.gemini_evasiveness_score_q_a,
            "gemini_key_topics_discussed": r.gemini_key_topics_discussed,
            "gemini_red_flags_identified": r.gemini_red_flags_identified,
            "gemini_raw_response_json": r.gemini_raw_response_json,

            "chatgpt_summary": r.chatgpt_summary,
            "chatgpt_overall_sentiment": r.chatgpt_overall_sentiment,
            "chatgpt_sentiment_scores_by_segment": r.chatgpt_sentiment_scores_by_segment,
            "chatgpt_management_confidence_score": r.chatgpt_management_confidence_score,
            "chatgpt_evasiveness_score_q_a": r.chatgpt_evasiveness_score_q_a,
            "chatgpt_key_topics_discussed": r.chatgpt_key_topics_discussed,
            "chatgpt_red_flags_identified": r.chatgpt_red_flags_identified,
            "chatgpt_raw_response_json": r.chatgpt_raw_response_json,

            "comparison_notes": r.comparison_notes,
        })
    return jsonify(reports_data), 200

@app.route('/analysis/<int:report_id>')
@login_required
def view_analysis_report(report_id):
    """Renders a single, detailed analysis report page."""
    report = data_manager.get_report_by_id(report_id)

    if not report:
        flash("Analysis report not found.", 'danger')
        return redirect(url_for('dashboard'))

    if report.user_id != current_user.user_id:
        flash("You do not have permission to view this report.", 'danger')
        return redirect(url_for('dashboard'))

    return render_template('analysis_report.html', report=report)

@app.route('/analysis/<int:report_id>/delete', methods=['POST'])
@login_required
def delete_analysis_report(report_id):
    """Deletes a specific analysis report by ID."""
    report = data_manager.get_report_by_id(report_id)

    if not report:
        flash("Analysis report not found.", 'danger')
        return redirect(url_for('dashboard'))

    if report.user_id != current_user.user_id:
        flash("You do not have permission to delete this report.", 'danger')
        return redirect(url_for('dashboard'))

    if data_manager.delete_analysis_report(report_id):
        flash(f"Analysis report {report_id} has been deleted.", 'success')
    else:
        flash(f"Failed to delete analysis report {report_id}.", 'danger')

    return redirect(url_for('dashboard'))

@app.route('/analysis/raw/<int:report_id>/<string:ai_name>')
@login_required
def view_raw_analysis_json(report_id, ai_name):
    """
    API endpoint to retrieve the raw JSON response for a specific report and AI.
    """
    report = data_manager.get_report_by_id(report_id)
    if not report:
        return jsonify({"message": "Analysis report not found."}), 404

    if report.user_id != current_user.user_id:
        return jsonify({"message": "Unauthorized: You do not have access to this report."}), 403

    if ai_name.lower() == 'gemini':
        raw_json = report.gemini_raw_response_json
    elif ai_name.lower() == 'chatgpt':
        raw_json = report.chatgpt_raw_response_json
    else:
        return jsonify({"message": "Invalid AI name. Must be 'gemini' or 'chatgpt'."}), 400

    if raw_json:
        return jsonify(raw_json)
    else:
        return jsonify({"message": f"Raw JSON response not available for {ai_name} on this report."}), 404


@app.errorhandler(404)
def page_not_found(e) -> Response:
    """
    Handles 404 errors (Page Not Found).
    Renders a custom 404 page and returns the 404 status code.
    The 'e' parameter is the error object, which is required by Flask.
    """
    print(f"An error occurred: {e}")
    if request.path.startswith('/api/'):
        response = jsonify({"error": "Not Found", "message": "The requested API endpoint does not exist."})
        response.status_code = 404
        return response
    # Return an HTML response for non-API requests
    return make_response(render_template('404.html'), 404)

@app.errorhandler(500)
def internal_server_error(e) -> Response:
    """
    Handles 500 errors (Internal Server Error).
    This is triggered by unhandled exceptions in your code.
    Renders a custom 500 page and returns the 500 status code.
    """
    print(f"An internal server error occurred: {e}")
    if request.path.startswith('/api/'):
        response = jsonify({"error": "Internal Server Error", "message": "Something went wrong on the server."})
        response.status_code = 500
        return response
    # Return an HTML response for non-API requests
    return make_response(render_template('500.html'), 500)


if __name__ == '__main__':
    # Creating database tables if they don't exist yet
    with app.app_context():
        db.create_all()
        print("Database tables created/checked.")

        test_user = data_manager.get_user_by_username("testuser")
        if not test_user:
            print("Seeding initial user data...")
            test_user = data_manager.create_user("testuser", "test@example.com", "password123")
            if test_user:
                print(f"Test user '{test_user.username}' created.")
            else:
                print("Failed to create test user.")

        if not data_manager.get_company_by_ticker("AAPL"):
            print("Seeding initial company data...")
            data_manager.add_company("AAPL", "Apple Inc.", "Technology", "Consumer Electronics", "NASDAQ", "https://placehold.co/50x50/000/fff?text=AAPL")
            data_manager.add_company("MSFT", "Microsoft Corp.", "Technology", "Software", "NASDAQ", "https://placehold.co/50x50/000/fff?text=MSFT")
            data_manager.add_company("GOOGL", "Alphabet Inc. (Google)", "Technology", "Internet Services", "NASDAQ", "https://placehold.co/50x50/000/fff?text=GOOGL")
            print("Initial companies added.")

        # Adding a dummy transcript for AAPL
        aapl_transcript_2025_Q1 = data_manager.get_transcript_by_details("AAPL", 2025, 1)
        if not aapl_transcript_2025_Q1:
            print("Seeding initial transcript data for AAPL...")
            dummy_transcript_text = """
            Welcome to Apple's Q1 2025 Earnings Call.
            CEO Tim Cook: We are pleased to announce a record quarter, driven by strong iPhone sales and continued growth in Services. Our innovation pipeline remains robust. We navigated a challenging macroeconomic environment with resilience.
            CFO Luca Maestri: Revenue was $120 billion, up 5% year-over-year. Services revenue reached an all-time high of $25 billion. Gross margin was 45%. We returned $20 billion to shareholders through dividends and buybacks.
            Analyst 1: Can you elaborate on the supply chain improvements and their impact on iPhone availability?
            CEO Tim Cook: Our supply chain teams have done an incredible job. We've seen significant improvements, and availability is now much better across our product lines, especially for iPhone 16 Pro. We expect this trend to continue.
            CFO Luca Maestri: China remains a very important market for us. We are seeing some localized challenges, but our long-term view remains positive. We are investing in local talent and partnerships to strengthen our position. We believe our product innovation will resonate with customers.
            """
            aapl_transcript_2025_Q1 = data_manager.add_transcript(
                ticker_symbol="AAPL",
                fiscal_year=2025,
                fiscal_quarter=1,
                call_date=date(2025, 2, 1), # Use datetime.date
                raw_text=dummy_transcript_text,
                speaker_segments=[
                    {"speaker": "CEO Tim Cook", "text": "We are pleased to announce..."},
                    {"speaker": "CFO Luca Maestri", "text": "Revenue was $120 billion..."},
                    {"speaker": "Analyst 1", "text": "Can you elaborate..."},
                    {"speaker": "CEO Tim Cook", "text": "Our supply chain teams..."},
                    {"speaker": "Analyst 2", "text": "Regarding China..."},
                    {"speaker": "CFO Luca Maestri", "text": "China remains a very important market..."}
                ],
                source_url=f"API-Ninjas:AAPL-2025-Q1"
            )
            if aapl_transcript_2025_Q1:
                print("Initial AAPL transcript added.")
            else:
                print("Failed to add initial AAPL transcript.")

        # Seeding a dummy analysis report for the seeded transcript
        if aapl_transcript_2025_Q1 and test_user:
            existing_report = data_manager.get_latest_report_for_user_and_transcript(test_user.user_id, aapl_transcript_2025_Q1.transcript_id)
            if existing_report:
                print(f"Dummy analysis report for Transcript ID {aapl_transcript_2025_Q1.transcript_id} already exists (Report ID: {existing_report.report_id}). Not re-seeding.")
            else:
                print("Seeding a dummy analysis report...")
                data_manager.create_analysis_report(
                    user_id=test_user.user_id,
                    transcript_id=aapl_transcript_2025_Q1.transcript_id,
                    gemini_summary="Gemini: Apple's quarter was strong due to iPhone and Services growth, with supply chain improvements. China faces challenges.",
                    gemini_overall_sentiment="Positive",
                    gemini_management_confidence_score=85,
                    gemini_evasiveness_score_q_a=20,
                    gemini_key_topics_discussed=["iPhone Sales", "Services Growth", "Supply Chain", "China Market"],
                    gemini_red_flags_identified=["localized challenges (China)"],
                    gemini_raw_response_json={"summary": "...", "overall_sentiment": "Positive"}, # Placeholder
                    chatgpt_summary="ChatGPT: Apple had a transformative quarter with resilient performance. Management was cautiously optimistic but avoided specific future guidance.",
                    chatgpt_overall_sentiment="Neutral",
                    chatgpt_management_confidence_score=70,
                    chatgpt_evasiveness_score_q_a=60,
                    chatgpt_key_topics_discussed=["Macroeconomic Headwinds", "Operational Efficiencies", "Future Guidance Evasiveness"],
                    chatgpt_red_flags_identified=["not providing granular forward-looking revenue guidance"],
                    chatgpt_raw_response_json={"summary": "...", "overall_sentiment": "Neutral"}, # Placeholder
                    comparison_notes="Both AIs noted strong performance but also management's general guidance."
                )
                print("Dummy analysis report seeded.")

    app.run(debug=True)
