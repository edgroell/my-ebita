# Data models for the application

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import UniqueConstraint
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

def get_current_datetime():
    return datetime.now()

class User(db.Model, UserMixin):
    """Contains all instances of users"""
    __tablename__ = "users"
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_active = db.Column(db.Boolean, default=True) # type: ignore
    created_at = db.Column(db.DateTime, default=get_current_datetime)
    updated_at = db.Column(db.DateTime, default=get_current_datetime, onupdate=get_current_datetime)
    last_login_at = db.Column(db.DateTime, nullable=True)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def get_id(self) -> str:
        """
        Flask-Login expects get_id() to return a unicode ID uniquely identifying
        this user. Return the primary key as a string.
        """
        return str(self.user_id)

    def __repr__(self):
        return f"<User {self.username} ({self.user_id})>"

class Company(db.Model):
    """Stores static metadata about companies whose earnings calls will be analyzed."""
    __tablename__ = "companies"
    ticker_symbol = db.Column(db.String(16), primary_key=True)
    company_name = db.Column(db.String(255), nullable=False)
    industry = db.Column(db.String(128))
    sector = db.Column(db.String(128))
    exchange = db.Column(db.String(64))
    logo_url = db.Column(db.String(512))

    transcripts = db.relationship("EarningsCallTranscript", back_populates="company", lazy=True)

    def __repr__(self):
        return f"<Company {self.ticker_symbol} - {self.company_name}>"

class EarningsCallTranscript(db.Model):
    """Stores the raw, processed text of earnings call transcripts."""
    __tablename__ = "earnings_call_transcripts"
    transcript_id = db.Column(db.Integer, primary_key=True)
    ticker_symbol = db.Column(db.String(16), db.ForeignKey("companies.ticker_symbol"), nullable=False)
    fiscal_year = db.Column(db.Integer, nullable=False)
    fiscal_quarter = db.Column(db.Integer, nullable=False)
    call_date = db.Column(db.DateTime)
    raw_text = db.Column(db.Text, nullable=False)
    speaker_segments = db.Column(db.JSON)
    source_url = db.Column(db.String(1024))

    company = db.relationship("Company", back_populates="transcripts", lazy=True)
    analysis_reports = db.relationship("AnalysisReport", back_populates="transcript", lazy=True, cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint('ticker_symbol', 'fiscal_year', 'fiscal_quarter', name='_company_period_uc'),)

    def __repr__(self):
        return f"<Transcript {self.transcript_id} {self.ticker_symbol} Q{self.fiscal_quarter} {self.fiscal_year}>"

class AnalysisReport(db.Model):
    """
    Stores AI-generated analysis for a transcript from multiple services.
    Includes concise rationale and timing metrics for each service.
    """
    __tablename__ = 'analysis_reports'
    report_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    transcript_id = db.Column(db.Integer, db.ForeignKey('earnings_call_transcripts.transcript_id'), nullable=False)
    analysis_date = db.Column(db.DateTime, nullable=False, default=get_current_datetime)

    # Relationship
    transcript = db.relationship(
        'EarningsCallTranscript',
        foreign_keys=[transcript_id],
        back_populates='analysis_reports',
        lazy=True
    )

    # --- Gemini Analysis Fields ---
    gemini_summary = db.Column(db.Text)
    gemini_concise_rationale = db.Column(db.Text)
    gemini_overall_sentiment = db.Column(db.String(50))
    gemini_sentiment_scores_by_segment = db.Column(db.JSON)
    gemini_management_confidence_score = db.Column(db.Float)
    gemini_evasiveness_score_q_a = db.Column(db.Float)
    gemini_key_topics_discussed = db.Column(db.JSON)
    gemini_red_flags_identified = db.Column(db.JSON)
    gemini_raw_response_json = db.Column(db.JSON)

    # Gemini timing
    gemini_request_ms = db.Column(db.Float)
    gemini_parse_ms = db.Column(db.Float)
    gemini_total_ms = db.Column(db.Float)

    # --- ChatGPT Analysis Fields ---
    chatgpt_summary = db.Column(db.Text)
    chatgpt_concise_rationale = db.Column(db.Text)
    chatgpt_overall_sentiment = db.Column(db.String(50))
    chatgpt_sentiment_scores_by_segment = db.Column(db.JSON)
    chatgpt_management_confidence_score = db.Column(db.Float)
    chatgpt_evasiveness_score_q_a = db.Column(db.Float)
    chatgpt_key_topics_discussed = db.Column(db.JSON)
    chatgpt_red_flags_identified = db.Column(db.JSON)
    chatgpt_raw_response_json = db.Column(db.JSON)

    # ChatGPT timing
    chatgpt_request_ms = db.Column(db.Float)
    chatgpt_parse_ms = db.Column(db.Float)
    chatgpt_total_ms = db.Column(db.Float)

    # --- Groq Analysis Fields ---
    groq_summary = db.Column(db.Text)
    groq_concise_rationale = db.Column(db.Text)
    groq_overall_sentiment = db.Column(db.String(50))
    groq_sentiment_scores_by_segment = db.Column(db.JSON)
    groq_management_confidence_score = db.Column(db.Float)
    groq_evasiveness_score_q_a = db.Column(db.Float)
    groq_key_topics_discussed = db.Column(db.JSON)
    groq_red_flags_identified = db.Column(db.JSON)
    groq_raw_response_json = db.Column(db.JSON)

    # Groq timing
    groq_request_ms = db.Column(db.Float)
    groq_parse_ms = db.Column(db.Float)
    groq_total_ms = db.Column(db.Float)

    __table_args__ = (UniqueConstraint('user_id', 'transcript_id', 'analysis_date',
                                       name='_user_transcript_date_uc'),)

    def __repr__(self):
        return f"<AnalysisReport {self.report_id} on Transcript {self.transcript_id} for User {self.user_id} ({self.analysis_date.isoformat()})>"
