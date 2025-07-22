"""
Module containing all data models of the app.
Designed for SQLite using Flask-SQLAlchemy.
"""

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import UniqueConstraint, event
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash


db = SQLAlchemy()

def get_current_datetime():
    return datetime.now()

class User(db.Model, UserMixin):
    """Contains all instances of users"""
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=get_current_datetime)
    updated_at = db.Column(db.DateTime, default=get_current_datetime, onupdate=get_current_datetime)
    last_login_at = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, nullable=False, default=True)

    def get_id(self):
        return str(self.user_id)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    # Relationships
    analysis_reports = db.relationship('AnalysisReport', backref='user', lazy=True, cascade="all, delete-orphan")
    watchlists = db.relationship('UserWatchlist', backref='user', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.username} (Active: {self.is_active})>"

class Company(db.Model):
    """Stores static metadata about companies whose earnings calls will be analyzed."""
    __tablename__ = 'companies'
    ticker_symbol = db.Column(db.String(10), primary_key=True)
    company_name = db.Column(db.String(255), nullable=False)
    industry = db.Column(db.String(100))
    sector = db.Column(db.String(100))
    exchange = db.Column(db.String(50))
    logo_url = db.Column(db.String(255))
    last_updated = db.Column(db.DateTime, default=get_current_datetime, onupdate=get_current_datetime)

    # Relationships
    earnings_call_transcripts = db.relationship('EarningsCallTranscript', backref='company', lazy=True)
    watchlist_items = db.relationship('UserWatchlist', backref='company', lazy=True)

    def __repr__(self):
        return f"<Company {self.ticker_symbol} ({self.company_name})>"

class EarningsCallTranscript(db.Model):
    """Stores the raw, processed text of earnings call transcripts."""
    __tablename__ = 'earnings_call_transcripts'
    transcript_id = db.Column(db.Integer, primary_key=True)
    ticker_symbol = db.Column(db.String(10), db.ForeignKey('companies.ticker_symbol'), nullable=False)
    fiscal_year = db.Column(db.Integer, nullable=False)
    fiscal_quarter = db.Column(db.Integer, nullable=False)
    call_date = db.Column(db.Date, nullable=False)
    raw_text = db.Column(db.Text, nullable=False)
    speaker_segments = db.Column(db.JSON)
    source_url = db.Column(db.String(255))
    fetched_at = db.Column(db.DateTime, nullable=False, default=get_current_datetime)

    # Unique constraint for preventing duplicate transcripts
    __table_args__ = (UniqueConstraint('ticker_symbol', 'fiscal_year', 'fiscal_quarter',
                                       name='_ticker_fiscal_year_quarter_uc'),)

    # Relationship to Analysis Reports
    analysis_reports = db.relationship('AnalysisReport', backref='transcript', lazy=True)

    def __repr__(self):
        return f"<Transcript {self.ticker_symbol} FY{self.fiscal_year} Q{self.fiscal_quarter}>"

class AnalysisReport(db.Model):
    """
    Stores the AI's generated "between-the-lines" reports for each transcript,
    including output from both Gemini and ChatGPT for direct comparison.
    """
    __tablename__ = 'analysis_reports'
    report_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    transcript_id = db.Column(db.Integer, db.ForeignKey('earnings_call_transcripts.transcript_id'), nullable=False)
    analysis_date = db.Column(db.DateTime, nullable=False, default=get_current_datetime)

    # --- Gemini Analysis Fields ---
    gemini_summary = db.Column(db.Text) # Nullable, as analysis might fail for one model
    gemini_overall_sentiment = db.Column(db.String(50))
    gemini_sentiment_scores_by_segment = db.Column(db.JSON)
    gemini_management_confidence_score = db.Column(db.Float)
    gemini_evasiveness_score_q_a = db.Column(db.Float)
    gemini_key_topics_discussed = db.Column(db.JSON)
    gemini_red_flags_identified = db.Column(db.JSON)
    gemini_raw_response_json = db.Column(db.JSON)

    # --- ChatGPT Analysis Fields ---
    chatgpt_summary = db.Column(db.Text) # Nullable, as analysis might fail for one model
    chatgpt_overall_sentiment = db.Column(db.String(50))
    chatgpt_sentiment_scores_by_segment = db.Column(db.JSON)
    chatgpt_management_confidence_score = db.Column(db.Float)
    chatgpt_evasiveness_score_q_a = db.Column(db.Float)
    chatgpt_key_topics_discussed = db.Column(db.JSON)
    chatgpt_red_flags_identified = db.Column(db.JSON)
    chatgpt_raw_response_json = db.Column(db.JSON)

    comparison_notes = db.Column(db.Text)

    # Ensure a user can generate multiple comparison reports for the same transcript over time
    # The 'analysis_date' should ensure uniqueness if multiple analyses are run.
    __table_args__ = (UniqueConstraint('user_id', 'transcript_id', 'analysis_date',
                                       name='_user_transcript_date_uc'),)

    def __repr__(self):
        return f"<AnalysisReport {self.report_id} on Transcript {self.transcript_id} for User {self.user_id} ({self.analysis_date.isoformat()})>"

class UserWatchlist(db.Model):
    """Allows users to create and manage lists of stocks they want to track."""
    __tablename__ = 'user_watchlists'
    watchlist_item_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    ticker_symbol = db.Column(db.String(10), db.ForeignKey('companies.ticker_symbol'), nullable=False)
    added_at = db.Column(db.DateTime, nullable=False, default=get_current_datetime)

    # Unique constraint to ensure a user cannot add the same stock to their watchlist multiple times
    __table_args__ = (UniqueConstraint('user_id', 'ticker_symbol', name='_user_ticker_watchlist_uc'),)

    def __repr__(self):
        return f"<WatchlistItem User {self.user_id} - {self.ticker_symbol}>"
