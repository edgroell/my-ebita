# Data models for the application

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import UniqueConstraint
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.orm import validates
import re

db = SQLAlchemy()

def get_current_datetime():
    return datetime.now()

class User(UserMixin, db.Model):
    """Contains all instances of users"""
    __tablename__ = "users"
    
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    active = db.Column('is_active', db.Boolean, default=True, nullable=False)  # persist as 'is_active' but avoid overriding UserMixin property
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    analysis_reports = db.relationship('AnalysisReport', back_populates='user', lazy='dynamic')
    transcripts = db.relationship('EarningsCallTranscript', back_populates='user', lazy='dynamic')

    def __repr__(self):
        return f"<User {self.username}"

    def get_id(self):
        return str(self.user_id)

    def set_password(self, password):
        """Hash and set the user's password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Check if the provided password matches the hashed password."""
        return check_password_hash(self.password_hash, password)

    @property
    def is_active(self) -> bool:
        """Flask-Login compatibility property backed by the 'active' column."""
        return bool(self.active)

    @is_active.setter
    def is_active(self, value: bool) -> None:
        self.active = bool(value)

    @staticmethod
    def validate_username_format(username: str) -> bool:
        """Validate username: 2+ chars, letters/numbers/_/- only."""
        if not isinstance(username, str):
            return False
        return bool(re.fullmatch(r'[A-Za-z0-9_-]{2,}', username))

class Company(db.Model):
    """Stores static metadata about companies whose earnings calls will be analyzed."""
    __tablename__ = "companies"
    ticker_symbol = db.Column(db.String(16), primary_key=True)
    company_name = db.Column(db.String(255), nullable=False)
    logo_url = db.Column(db.String(512))

    transcripts = db.relationship("EarningsCallTranscript", back_populates="company", lazy=True)

    def __repr__(self):
        return f"<Company {self.ticker_symbol} - {self.company_name}>"

class EarningsCallTranscript(db.Model):
    """Stores the earnings call transcripts. Each transcript is owned by a user."""
    __tablename__ = "earnings_call_transcripts"
    transcript_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.user_id"), nullable=False)
    ticker_symbol = db.Column(db.String(16), db.ForeignKey("companies.ticker_symbol"), nullable=False)
    fiscal_year = db.Column(db.Integer, nullable=False)
    fiscal_quarter = db.Column(db.Integer, nullable=False)
    call_date = db.Column(db.DateTime)
    transcript_raw = db.Column(db.JSON, nullable=False)
    transcript_split = db.Column(db.JSON, nullable=True)
    source_url = db.Column(db.String(1024))

    company = db.relationship("Company", back_populates="transcripts", lazy=True)
    user = db.relationship("User", back_populates="transcripts", lazy=True)
    analysis_reports = db.relationship("AnalysisReport", back_populates="transcript", lazy=True, cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint('user_id', 'ticker_symbol', 'fiscal_year', 'fiscal_quarter', name='_user_company_period_uc'),)

    def __repr__(self):
        return f"<Transcript {self.transcript_id} {self.ticker_symbol} Q{self.fiscal_quarter} {self.fiscal_year} (user={self.user_id})>"

class AnalysisReport(db.Model):
    """
    Stores AI-generated analysis for a transcript from multiple services.
    Each analysis report is owned by a user and linked to a specific transcript.
    """
    __tablename__ = 'analysis_reports'
    report_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    transcript_id = db.Column(db.Integer, db.ForeignKey('earnings_call_transcripts.transcript_id'), nullable=False)
    analysis_date = db.Column(db.DateTime, nullable=False, default=get_current_datetime)

    # Relationships
    user = db.relationship('User', foreign_keys=[user_id], back_populates='analysis_reports', lazy=True)
    transcript = db.relationship(
        'EarningsCallTranscript',
        foreign_keys=[transcript_id],
        back_populates='analysis_reports',
        lazy=True
    )

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

    # ChatGPT timing + token/metadata
    chatgpt_request_ms = db.Column(db.Float)
    chatgpt_model = db.Column(db.String(128))
    chatgpt_temperature = db.Column(db.Float)
    chatgpt_max_tokens = db.Column(db.Integer)
    chatgpt_total_tokens = db.Column(db.Integer)
    chatgpt_prompt_tokens = db.Column(db.Integer)
    chatgpt_completion_tokens = db.Column(db.Integer)

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

    # Gemini timing + token/metadata
    gemini_request_ms = db.Column(db.Float)
    gemini_model = db.Column(db.String(128))
    gemini_temperature = db.Column(db.Float)
    gemini_max_tokens = db.Column(db.Integer)
    gemini_prompt_tokens = db.Column(db.Integer)
    gemini_thoughts_tokens = db.Column(db.Integer)
    gemini_candidates_tokens = db.Column(db.Integer)

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

    # Groq timing + token/metadata
    groq_request_ms = db.Column(db.Float)
    groq_model = db.Column(db.String(128))
    groq_temperature = db.Column(db.Float)
    groq_max_tokens = db.Column(db.Integer)
    groq_total_tokens = db.Column(db.Integer)
    groq_prompt_tokens = db.Column(db.Integer)
    groq_completion_tokens = db.Column(db.Integer)

    # enforce uniqueness: one analysis report per user/transcript/date
    __table_args__ = (UniqueConstraint('user_id', 'transcript_id', 'analysis_date',
                                       name='_user_transcript_date_uc'),)

    def __repr__(self):
        return f"<AnalysisReport {self.report_id} on Transcript {self.transcript_id} for User {self.user_id} ({self.analysis_date.isoformat()})>"
