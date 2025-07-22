"""
DataManager: Data Access Layer for the application.
This module provides methods for interacting with the database
using Flask-SQLAlchemy models.
"""

from .data_models import db, User, Company, EarningsCallTranscript, AnalysisReport, UserWatchlist
from sqlalchemy.exc import IntegrityError

class DataManager:
    """
    Manages all database interactions for the application.
    Encapsulates CRUD operations for users, companies, transcripts,
    analysis reports, and watchlists.
    """
    def __init__(self, app_db):
        """
        Initializes the DataManager with the SQLAlchemy db object.
        """
        self.db = app_db

    # --- User Management ---
    def create_user(self, username, email, password):
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
            return None # User/email already exists
        except Exception as e:
            self.db.session.rollback()
            print(f"Error creating user: {e}")
            return None

    def get_user_by_id(self, user_id):
        """Retrieves a user by their ID."""
        return User.query.get(user_id)

    def get_user_by_username(self, username):
        """Retrieves a user by their username."""
        return User.query.filter_by(username=username).first()

    def get_user_by_email(self, email):
        """Retrieves a user by their email address."""
        return User.query.filter_by(email=email).first()

    def update_user_profile(self, user_id, **kwargs):
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
            return None # e.g., new email already exists
        except Exception as e:
            self.db.session.rollback()
            print(f"Error updating user profile: {e}")
            return None

    def deactivate_user(self, user_id):
        """Deactivates a user account."""
        return self.update_user_profile(user_id, is_active=False)

    def activate_user(self, user_id):
        """Activates a user account."""
        return self.update_user_profile(user_id, is_active=True)

    def delete_user(self, user_id):
        """
        Deletes a user and their associated data (reports, watchlists) due to cascading deletes.
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
    def add_company(self, ticker_symbol, company_name, industry=None, sector=None, exchange=None, logo_url=None):
        """
        Adds a new company to the database.
        Returns the Company object on success, None if ticker_symbol already exists or on error.
        """
        try:
            new_company = Company(
                ticker_symbol=ticker_symbol,
                company_name=company_name,
                industry=industry,
                sector=sector,
                exchange=exchange,
                logo_url=logo_url
            )
            self.db.session.add(new_company)
            self.db.session.commit()
            return new_company
        except IntegrityError:
            self.db.session.rollback()
            return None # Ticker already exists
        except Exception as e:
            self.db.session.rollback()
            print(f"Error adding company: {e}")
            return None

    def get_company_by_ticker(self, ticker_symbol):
        """Retrieves a company by its ticker symbol."""
        return Company.query.get(ticker_symbol)

    def get_all_companies(self, search_query=None, industry=None, sector=None, limit=None, offset=None):
        """
        Retrieves all companies, with optional search and filtering.
        """
        query = Company.query
        if search_query:
            query = query.filter(Company.company_name.ilike(f'%{search_query}%') |
                                 Company.ticker_symbol.ilike(f'%{search_query}%'))
        if industry:
            query = query.filter_by(industry=industry)
        if sector:
            query = query.filter_by(sector=sector)

        if limit is not None:
            query = query.limit(limit)
        if offset is not None:
            query = query.offset(offset)

        return query.all()

    # --- Earnings Call Transcript Management ---
    def add_transcript(self, ticker_symbol, fiscal_year, fiscal_quarter, call_date, raw_text, speaker_segments=None, source_url=None):
        """
        Adds a new earnings call transcript.
        Returns the Transcript object on success, None on error (e.g., duplicate entry).
        """
        try:
            new_transcript = EarningsCallTranscript(
                ticker_symbol=ticker_symbol,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
                call_date=call_date,
                raw_text=raw_text,
                speaker_segments=speaker_segments,
                source_url=source_url
            )
            self.db.session.add(new_transcript)
            self.db.session.commit()
            return new_transcript
        except IntegrityError:
            self.db.session.rollback()
            return None # Duplicate transcript for company/year/quarter
        except Exception as e:
            self.db.session.rollback()
            print(f"Error adding transcript: {e}")
            return None

    def get_transcript_by_id(self, transcript_id):
        """Retrieves an earnings call transcript by its ID."""
        return EarningsCallTranscript.query.get(transcript_id)

    def get_transcripts_for_company(self, ticker_symbol):
        """Retrieves all transcripts for a given company."""
        return EarningsCallTranscript.query.filter_by(ticker_symbol=ticker_symbol).order_by(
            EarningsCallTranscript.fiscal_year.desc(),
            EarningsCallTranscript.fiscal_quarter.desc()
        ).all()

    def get_transcript_by_details(self, ticker_symbol, fiscal_year, fiscal_quarter):
        """Retrieves a specific transcript by company and fiscal period."""
        return EarningsCallTranscript.query.filter_by(
            ticker_symbol=ticker_symbol,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter
        ).first()

    # --- Analysis Report Management ---
    def create_analysis_report(self, user_id, transcript_id,
                               gemini_summary=None, gemini_overall_sentiment=None,
                               gemini_sentiment_scores_by_segment=None, gemini_management_confidence_score=None,
                               gemini_evasiveness_score_q_a=None, gemini_key_topics_discussed=None,
                               gemini_red_flags_identified=None, gemini_raw_response_json=None,
                               chatgpt_summary=None, chatgpt_overall_sentiment=None,
                               chatgpt_sentiment_scores_by_segment=None, chatgpt_management_confidence_score=None,
                               chatgpt_evasiveness_score_q_a=None, chatgpt_key_topics_discussed=None,
                               chatgpt_red_flags_identified=None, chatgpt_raw_response_json=None,
                               comparison_notes=None):
        """
        Creates a new AI analysis report, storing results from both Gemini and ChatGPT.
        Returns the AnalysisReport object on success, None on error.
        """
        try:
            new_report = AnalysisReport(
                user_id=user_id,
                transcript_id=transcript_id,

                gemini_summary=gemini_summary,
                gemini_overall_sentiment=gemini_overall_sentiment,
                gemini_sentiment_scores_by_segment=gemini_sentiment_scores_by_segment,
                gemini_management_confidence_score=gemini_management_confidence_score,
                gemini_evasiveness_score_q_a=gemini_evasiveness_score_q_a,
                gemini_key_topics_discussed=gemini_key_topics_discussed,
                gemini_red_flags_identified=gemini_red_flags_identified,
                gemini_raw_response_json=gemini_raw_response_json,

                chatgpt_summary=chatgpt_summary,
                chatgpt_overall_sentiment=chatgpt_overall_sentiment,
                chatgpt_sentiment_scores_by_segment=chatgpt_sentiment_scores_by_segment,
                chatgpt_management_confidence_score=chatgpt_management_confidence_score,
                chatgpt_evasiveness_score_q_a=chatgpt_evasiveness_score_q_a,
                chatgpt_key_topics_discussed=chatgpt_key_topics_discussed,
                chatgpt_red_flags_identified=chatgpt_red_flags_identified,
                chatgpt_raw_response_json=chatgpt_raw_response_json,

                comparison_notes=comparison_notes
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

    def get_report_by_id(self, report_id):
        """Retrieves an analysis report by its ID."""
        return AnalysisReport.query.get(report_id)

    def get_reports_for_user(self, user_id):
        """Retrieves all analysis reports for a specific user."""
        return AnalysisReport.query.filter_by(user_id=user_id).order_by(AnalysisReport.analysis_date.desc()).all()

    def get_reports_for_transcript(self, transcript_id):
        """Retrieves all analysis reports for a specific transcript."""
        return AnalysisReport.query.filter_by(transcript_id=transcript_id).order_by(AnalysisReport.analysis_date.desc()).all()

    def get_latest_report_for_user_and_transcript(self, user_id, transcript_id):
        """Retrieves the latest analysis report for a given user and transcript."""
        return AnalysisReport.query.filter_by(
            user_id=user_id,
            transcript_id=transcript_id
        ).order_by(AnalysisReport.analysis_date.desc()).first()

    # --- User Watchlist Management ---
    def add_stock_to_watchlist(self, user_id, ticker_symbol):
        """
        Adds a stock to a user's watchlist.
        Returns the WatchlistItem object on success, None if already in watchlist or on error.
        """
        try:
            new_watchlist_item = UserWatchlist(user_id=user_id, ticker_symbol=ticker_symbol)
            self.db.session.add(new_watchlist_item)
            self.db.session.commit()
            return new_watchlist_item
        except IntegrityError:
            self.db.session.rollback()
            return None # Stock already in watchlist for this user
        except Exception as e:
            self.db.session.rollback()
            print(f"Error adding stock to watchlist: {e}")
            return None

    def remove_stock_from_watchlist(self, user_id, ticker_symbol):
        """
        Removes a stock from a user's watchlist.
        Returns True on success, False if not found or on error.
        """
        item = UserWatchlist.query.filter_by(user_id=user_id, ticker_symbol=ticker_symbol).first()
        if not item:
            return False
        try:
            self.db.session.delete(item)
            self.db.session.commit()
            return True
        except Exception as e:
            self.db.session.rollback()
            print(f"Error removing stock from watchlist: {e}")
            return False

    def get_user_watchlist(self, user_id):
        """Retrieves all stocks in a user's watchlist."""
        return UserWatchlist.query.filter_by(user_id=user_id).join(Company).all()

    def is_stock_in_watchlist(self, user_id, ticker_symbol):
        """Checks if a specific stock is in a user's watchlist."""
        return UserWatchlist.query.filter_by(user_id=user_id, ticker_symbol=ticker_symbol).first() is not None
