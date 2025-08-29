# DataManager: Data Access Layer for the application

from typing import Optional, List, Any
from datetime import datetime

from .data_models import User, Company, EarningsCallTranscript, AnalysisReport
from sqlalchemy.exc import IntegrityError


class DataManager:
    """
    Manages all database interactions for the application.
    Encapsulates CRUD operations for users, companies, transcripts,
    and analysis reports.
    """
    def __init__(self, app_db):
        """
        Initializes the DataManager with the SQLAlchemy db object.
        """
        self.db = app_db

    # --- User Management ---
    def create_user(self, username: str, email: str, password: str) -> Optional[User]:
        """
        Creates a new user in the database.
        Returns the User object on success, None on error (e.g., username/email already exists).
        """
        try:
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            self.db.session.add(new_user)
            self.db.session.commit()
            return new_user
        except IntegrityError:
            self.db.session.rollback()
            return None
        except Exception as e:
            self.db.session.rollback()
            print(f"Error creating user: {e}")
            return None

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Retrieves a user by their ID."""
        return User.query.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Retrieves a user by their username."""
        return User.query.filter_by(username=username).first()

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Retrieves a user by their email address."""
        return User.query.filter_by(email=email).first()

    def update_user_profile(self, user_id: int, **kwargs) -> Optional[User]:
        """
        Updates a user's profile information.
        Kwargs can include: email, is_active, last_login_at.
        Returns the updated User object on success, None if user not found or on error.
        """
        user = self.get_user_by_id(user_id)
        if not user:
            return None
        try:
            for key, value in kwargs.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            self.db.session.commit()
            return user
        except IntegrityError:
            self.db.session.rollback()
            return None
        except Exception as e:
            self.db.session.rollback()
            print(f"Error updating user profile: {e}")
            return None

    def activate_user(self, user_id: int) -> Optional[User]:
        """Activates a user account."""
        return self.update_user_profile(user_id, is_active=True)

    def deactivate_user(self, user_id: int) -> Optional[User]:
        """Deactivates a user account."""
        return self.update_user_profile(user_id, is_active=False)

    def delete_user(self, user_id: int) -> bool:
        """
        Deletes a user and their associated data (transcripts & reports) due to cascading deletes.
        Returns True on success, False otherwise.
        """
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        try:
            self.db.session.delete(user)
            self.db.session.commit()
            return True
        except Exception as e:
            self.db.session.rollback()
            print(f"Error deleting user: {e}")
            return False

    # --- Company Management ---
    def add_company(self, ticker_symbol: str, company_name: str, industry: Optional[str] = None,
                    sector: Optional[str] = None, exchange: Optional[str] = None,
                    logo_url: Optional[str] = None) -> Optional[Company]:
        """
        Adds a new company to the database.
        Returns the Company object on success, None if ticker_symbol already exists or on error.
        """
        try:
            new_company = Company(
                ticker_symbol=ticker_symbol,
                company_name=company_name,
                logo_url=logo_url
            )
            self.db.session.add(new_company)
            self.db.session.commit()
            return new_company
        except IntegrityError:
            self.db.session.rollback()
            return None
        except Exception as e:
            self.db.session.rollback()
            print(f"Error adding company: {e}")
            return None

    def get_company_by_ticker(self, ticker_symbol: str) -> Optional[Company]:
        """Retrieves a company by its ticker symbol."""
        return Company.query.get(ticker_symbol)

    def get_all_companies(self, search_query: Optional[str] = None, industry: Optional[str] = None,
                          sector: Optional[str] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Company]:
        """
        Retrieves all companies, with optional search and filtering.
        """
        query = Company.query
        if search_query:
            query = query.filter(Company.company_name.ilike(f'%{search_query}%') |
                                 Company.ticker_symbol.ilike(f'%{search_query}%'))
        if industry and hasattr(Company, "industry"):
            query = query.filter_by(industry=industry)
        if sector and hasattr(Company, "sector"):
            query = query.filter_by(sector=sector)

        if limit is not None:
            query = query.limit(limit)
        if offset is not None:
            query = query.offset(offset)

        return query.all()

    # --- Earnings Call Transcript Management ---
    def add_transcript(self, user_id: int, ticker_symbol: str, fiscal_year: int, fiscal_quarter: int,
                       call_date: Optional[datetime] , transcript_raw: Any,
                       source_url: Optional[str] = None) -> Optional[EarningsCallTranscript]:
        """
        Adds a new earnings call transcript associated with a specific user.
        Returns the Transcript object on success, None on error (e.g., duplicate entry).
        """
        try:
            new_transcript = EarningsCallTranscript(
                user_id=user_id,
                ticker_symbol=ticker_symbol,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
                call_date=call_date,
                transcript_raw=transcript_raw,
                source_url=source_url
            )
            self.db.session.add(new_transcript)
            self.db.session.commit()
            return new_transcript
        except IntegrityError:
            self.db.session.rollback()
            return None
        except Exception as e:
            self.db.session.rollback()
            print(f"Error adding transcript: {e}")
            return None

    def get_transcript_by_id(self, transcript_id: int) -> Optional[EarningsCallTranscript]:
        """Retrieves an earnings call transcript by its ID."""
        return EarningsCallTranscript.query.get(transcript_id)

    def get_transcripts_for_user(self, user_id: int) -> List[EarningsCallTranscript]:
        """Retrieves all earnings call transcripts owned by a specific user."""
        return EarningsCallTranscript.query.filter_by(user_id=user_id).order_by(
            EarningsCallTranscript.call_date.desc()
        ).all()

    def get_transcripts_for_company(self, ticker_symbol: str) -> List[EarningsCallTranscript]:
        """Retrieves all transcripts for a given company (across users)."""
        return EarningsCallTranscript.query.filter_by(ticker_symbol=ticker_symbol).order_by(
            EarningsCallTranscript.fiscal_year.desc(),
            EarningsCallTranscript.fiscal_quarter.desc()
        ).all()

    def get_transcript_for_user_by_details(self, user_id: int, ticker_symbol: str, fiscal_year: int, fiscal_quarter: int) -> Optional[EarningsCallTranscript]:
        """Retrieves a specific transcript by company and fiscal period for a given user."""
        return EarningsCallTranscript.query.filter_by(
            user_id=user_id,
            ticker_symbol=ticker_symbol,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter
        ).first()

    def get_transcript_by_details(self, ticker_symbol: str, fiscal_year: int, fiscal_quarter: int) -> Optional[EarningsCallTranscript]:
        """Retrieves a specific transcript by company and fiscal period (any user)."""
        return EarningsCallTranscript.query.filter_by(
            ticker_symbol=ticker_symbol,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter
        ).first()

    def delete_transcript(self, transcript_id: int, requesting_user_id: Optional[int] = None) -> bool:
        """
        Deletes a transcript by ID.
        If requesting_user_id is provided, enforce ownership (only owner can delete).
        Returns True on success, False otherwise.
        """
        transcript = self.get_transcript_by_id(transcript_id)
        if not transcript:
            return False
        if requesting_user_id is not None and transcript.user_id != requesting_user_id:
            print(f"Delete denied: user {requesting_user_id} does not own transcript {transcript_id}")
            return False
        try:
            self.db.session.delete(transcript)
            self.db.session.commit()
            return True
        except Exception as e:
            self.db.session.rollback()
            print(f"Error deleting transcript {transcript_id}: {e}")
            return False

    # --- Analysis Report Management ---
    def create_analysis_report(self, user_id: int, transcript_id: int,
                               gemini_summary: Optional[str] = None, gemini_concise_rationale: Optional[str] = None,
                               gemini_overall_sentiment: Optional[str] = None,
                               gemini_sentiment_scores_by_segment: Optional[Any] = None,
                               gemini_management_confidence_score: Optional[float] = None,
                               gemini_evasiveness_score_q_a: Optional[float] = None,
                               gemini_key_topics_discussed: Optional[Any] = None,
                               gemini_red_flags_identified: Optional[Any] = None,
                               gemini_raw_response_json: Optional[Any] = None,
                               gemini_request_ms: Optional[float] = None,
                               chatgpt_summary: Optional[str] = None, chatgpt_concise_rationale: Optional[str] = None,
                               chatgpt_overall_sentiment: Optional[str] = None,
                               chatgpt_sentiment_scores_by_segment: Optional[Any] = None,
                               chatgpt_management_confidence_score: Optional[float] = None,
                               chatgpt_evasiveness_score_q_a: Optional[float] = None,
                               chatgpt_key_topics_discussed: Optional[Any] = None,
                               chatgpt_red_flags_identified: Optional[Any] = None,
                               chatgpt_raw_response_json: Optional[Any] = None,
                               chatgpt_request_ms: Optional[float] = None,
                               groq_summary: Optional[str] = None, groq_concise_rationale: Optional[str] = None,
                               groq_overall_sentiment: Optional[str] = None,
                               groq_sentiment_scores_by_segment: Optional[Any] = None,
                               groq_management_confidence_score: Optional[float] = None,
                               groq_evasiveness_score_q_a: Optional[float] = None,
                               groq_key_topics_discussed: Optional[Any] = None,
                               groq_red_flags_identified: Optional[Any] = None,
                               groq_raw_response_json: Optional[Any] = None,
                               groq_request_ms: Optional[float] = None) -> Optional[AnalysisReport]:
        """
        Creates a new AI analysis report, storing results from ChatGPT, Gemini and Groq.
        Ensures the transcript belongs to the provided user_id.
        Returns the AnalysisReport object on success, None on error.
        """
        try:
            transcript = self.get_transcript_by_id(transcript_id)
            if transcript is None:
                print(f"Transcript {transcript_id} not found.")
                return None
            if transcript.user_id != user_id:
                print(f"User {user_id} does not own transcript {transcript_id}; aborting report creation.")
                return None

            new_report = AnalysisReport(
                user_id=user_id,
                transcript_id=transcript_id,

                gemini_summary=gemini_summary,
                gemini_concise_rationale=gemini_concise_rationale,
                gemini_overall_sentiment=gemini_overall_sentiment,
                gemini_sentiment_scores_by_segment=gemini_sentiment_scores_by_segment,
                gemini_management_confidence_score=gemini_management_confidence_score,
                gemini_evasiveness_score_q_a=gemini_evasiveness_score_q_a,
                gemini_key_topics_discussed=gemini_key_topics_discussed,
                gemini_red_flags_identified=gemini_red_flags_identified,
                gemini_raw_response_json=gemini_raw_response_json,

                gemini_request_ms=gemini_request_ms,

                chatgpt_summary=chatgpt_summary,
                chatgpt_concise_rationale=chatgpt_concise_rationale,
                chatgpt_overall_sentiment=chatgpt_overall_sentiment,
                chatgpt_sentiment_scores_by_segment=chatgpt_sentiment_scores_by_segment,
                chatgpt_management_confidence_score=chatgpt_management_confidence_score,
                chatgpt_evasiveness_score_q_a=chatgpt_evasiveness_score_q_a,
                chatgpt_key_topics_discussed=chatgpt_key_topics_discussed,
                chatgpt_red_flags_identified=chatgpt_red_flags_identified,
                chatgpt_raw_response_json=chatgpt_raw_response_json,

                chatgpt_request_ms=chatgpt_request_ms,

                groq_summary=groq_summary,
                groq_concise_rationale=groq_concise_rationale,
                groq_overall_sentiment=groq_overall_sentiment,
                groq_sentiment_scores_by_segment=groq_sentiment_scores_by_segment,
                groq_management_confidence_score=groq_management_confidence_score,
                groq_evasiveness_score_q_a=groq_evasiveness_score_q_a,
                groq_key_topics_discussed=groq_key_topics_discussed,
                groq_red_flags_identified=groq_red_flags_identified,
                groq_raw_response_json=groq_raw_response_json,

                groq_request_ms=groq_request_ms,
            )
            self.db.session.add(new_report)
            self.db.session.commit()
            return new_report
        except IntegrityError:
            self.db.session.rollback()
            print(f"Integrity Error creating analysis report for user {user_id} transcript {transcript_id}.")
            return None
        except Exception as e:
            self.db.session.rollback()
            print(f"Error creating analysis report: {e}")
            return None

    def get_report_by_id(self, report_id: int) -> Optional[AnalysisReport]:
        """Retrieves an analysis report by its ID."""
        return AnalysisReport.query.get(report_id)

    def get_reports_for_user(self, user_id: int) -> List[AnalysisReport]:
        """Retrieves all analysis reports for a specific user."""
        return AnalysisReport.query.filter_by(user_id=user_id).order_by(AnalysisReport.analysis_date.desc()).all()

    def get_reports_for_transcript(self, transcript_id: int) -> List[AnalysisReport]:
        """Retrieves all analysis reports for a specific transcript."""
        return AnalysisReport.query.filter_by(transcript_id=transcript_id).order_by(AnalysisReport.analysis_date.desc()).all()

    def get_latest_report_for_user_and_transcript(self, user_id: int, transcript_id: int) -> Optional[AnalysisReport]:
        """Retrieves the latest analysis report for a given user and transcript."""
        return AnalysisReport.query.filter_by(
            user_id=user_id,
            transcript_id=transcript_id
        ).order_by(AnalysisReport.analysis_date.desc()).first()

    def delete_analysis_report(self, report_id: int, requesting_user_id: Optional[int] = None) -> bool:
        """
        Deletes an analysis report by ID.
        If requesting_user_id is provided, enforce ownership (only owner can delete).
        Returns True on success, False otherwise.
        """
        report = self.get_report_by_id(report_id)
        if not report:
            return False
        if requesting_user_id is not None and report.user_id != requesting_user_id:
            print(f"Delete denied: user {requesting_user_id} does not own report {report_id}")
            return False
        try:
            self.db.session.delete(report)
            self.db.session.commit()
            return True
        except Exception as e:
            self.db.session.rollback()
            print(f"Error deleting analysis report {report_id}: {e}")
            return False
