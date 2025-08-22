# DataManager: Data Access Layer for the application

from .data_models import User, Company, EarningsCallTranscript, AnalysisReport
from sqlalchemy.exc import IntegrityError

class DataManager:
    """
    Manages all database interactions for the application.
    Encapsulates CRUD operations for users, companies, transcripts,
    and analysis reports.
    """
    def __init__(self, app_db): # type: ignore
        """
        Initializes the DataManager with the SQLAlchemy db object.
        """
        self.db = app_db

    # --- User Management ---
    def create_user(self, username, email, password): # type: ignore
        """
        Creates a new user in the database.
        Returns the User object on success, None on error (e.g., username/email already exists).
        """
        try:
            new_user = User(username=username, email=email) # type: ignore
            new_user.set_password(password) # type: ignore
            self.db.session.add(new_user) # type: ignore
            self.db.session.commit() # type: ignore
            return new_user
        except IntegrityError:
            self.db.session.rollback() # type: ignore
            return None # User/email already exists
        except Exception as e:
            self.db.session.rollback() # type: ignore
            print(f"Error creating user: {e}")
            return None

    def get_user_by_id(self, user_id): # type: ignore
        """Retrieves a user by their ID."""
        return User.query.get(user_id) # type: ignore

    def get_user_by_username(self, username): # type: ignore
        """Retrieves a user by their username."""
        return User.query.filter_by(username=username).first() # type: ignore

    def get_user_by_email(self, email): # type: ignore
        """Retrieves a user by their email address."""
        return User.query.filter_by(email=email).first() # type: ignore

    def update_user_profile(self, user_id, **kwargs): # type: ignore
        """
        Updates a user's profile information.
        Kwargs can include: email, is_active, last_login_at.
        Returns the updated User object on success, None if user not found or on error.
        """
        user = self.get_user_by_id(user_id) # type: ignore
        if not user:
            return None
        try:
            for key, value in kwargs.items(): # type: ignore
                if hasattr(user, key):
                    setattr(user, key, value)
            self.db.session.commit() # type: ignore
            return user
        except IntegrityError:
            self.db.session.rollback() # type: ignore
            return None # e.g., new email already exists
        except Exception as e:
            self.db.session.rollback() # type: ignore
            print(f"Error updating user profile: {e}")
            return None

    def deactivate_user(self, user_id): # type: ignore
        """Deactivates a user account."""
        return self.update_user_profile(user_id, is_active=False) # type: ignore

    def activate_user(self, user_id): # type: ignore
        """Activates a user account."""
        return self.update_user_profile(user_id, is_active=True) # type: ignore

    def delete_user(self, user_id): # type: ignore
        """
        Deletes a user and their associated data (reports) due to cascading deletes.
        Returns True on success, False otherwise.
        """
        user = self.get_user_by_id(user_id) # type: ignore
        if not user:
            return False
        try:
            self.db.session.delete(user) # type: ignore
            self.db.session.commit() # type: ignore
            return True
        except Exception as e:
            self.db.session.rollback() # type: ignore
            print(f"Error deleting user: {e}")
            return False

    # --- Company Management ---
    def add_company(self, ticker_symbol, company_name, industry=None, sector=None, exchange=None, logo_url=None): # type: ignore
        """
        Adds a new company to the database.
        Returns the Company object on success, None if ticker_symbol already exists or on error.
        """
        try:
            new_company = Company(
                ticker_symbol=ticker_symbol, # type: ignore
                company_name=company_name, # type: ignore
                industry=industry, # type: ignore
                sector=sector, # type: ignore
                exchange=exchange, # type: ignore
                logo_url=logo_url  # type: ignore
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
                ticker_symbol=ticker_symbol, # type: ignore
                fiscal_year=fiscal_year, # type: ignore
                fiscal_quarter=fiscal_quarter, # type: ignore
                call_date=call_date, # type: ignore
                raw_text=raw_text, # type: ignore
                speaker_segments=speaker_segments, # type: ignore
                source_url=source_url # type: ignore
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

    def get_all_transcripts(self):
        """Retrieves all earnings call transcripts."""
        return EarningsCallTranscript.query.order_by(
            EarningsCallTranscript.call_date.desc()
        ).all()

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

    def delete_transcript(self, transcript_id):
        """
        Deletes a transcript by ID.
        Returns True on success, False otherwise.
        """
        transcript = self.get_transcript_by_id(transcript_id)
        if not transcript:
            return False
        try:
            # Thanks to cascading deletes, this will also delete all
            # AnalysisReports associated with this transcript.
            self.db.session.delete(transcript)
            self.db.session.commit()
            return True
        except Exception as e:
            self.db.session.rollback()
            print(f"Error deleting transcript {transcript_id}: {e}")
            return False

    # --- Analysis Report Management ---
    def create_analysis_report(self, user_id, transcript_id,
                               gemini_summary=None, gemini_concise_rationale=None, gemini_overall_sentiment=None,
                               gemini_sentiment_scores_by_segment=None, gemini_management_confidence_score=None,
                               gemini_evasiveness_score_q_a=None, gemini_key_topics_discussed=None,
                               gemini_red_flags_identified=None, gemini_raw_response_json=None,
                               gemini_request_ms=None, gemini_parse_ms=None, gemini_total_ms=None,
                               chatgpt_summary=None, chatgpt_concise_rationale=None, chatgpt_overall_sentiment=None,
                               chatgpt_sentiment_scores_by_segment=None, chatgpt_management_confidence_score=None,
                               chatgpt_evasiveness_score_q_a=None, chatgpt_key_topics_discussed=None,
                               chatgpt_red_flags_identified=None, chatgpt_raw_response_json=None,
                               chatgpt_request_ms=None, chatgpt_parse_ms=None, chatgpt_total_ms=None,
                               groq_summary=None, groq_concise_rationale=None, groq_overall_sentiment=None,
                               groq_sentiment_scores_by_segment=None, groq_management_confidence_score=None,
                               groq_evasiveness_score_q_a=None, groq_key_topics_discussed=None,
                               groq_red_flags_identified=None, groq_raw_response_json=None,
                               groq_request_ms=None, groq_parse_ms=None, groq_total_ms=None):
        """
        Creates a new AI analysis report, storing results from Gemini, ChatGPT and Groq.
        Returns the AnalysisReport object on success, None on error.
        """
        try:
            new_report = AnalysisReport(
                user_id=user_id, # type: ignore
                transcript_id=transcript_id, # type: ignore

                gemini_summary=gemini_summary, # type: ignore
                gemini_concise_rationale=gemini_concise_rationale, # type: ignore
                gemini_overall_sentiment=gemini_overall_sentiment, # type: ignore
                gemini_sentiment_scores_by_segment=gemini_sentiment_scores_by_segment, # type: ignore
                gemini_management_confidence_score=gemini_management_confidence_score, # type: ignore
                gemini_evasiveness_score_q_a=gemini_evasiveness_score_q_a, # type: ignore
                gemini_key_topics_discussed=gemini_key_topics_discussed, # type: ignore
                gemini_red_flags_identified=gemini_red_flags_identified, # type: ignore
                gemini_raw_response_json=gemini_raw_response_json, # type: ignore

                gemini_request_ms=gemini_request_ms, # type: ignore
                gemini_parse_ms=gemini_parse_ms, # type: ignore
                gemini_total_ms=gemini_total_ms, # type: ignore

                chatgpt_summary=chatgpt_summary, # type: ignore
                chatgpt_concise_rationale=chatgpt_concise_rationale, # type: ignore
                chatgpt_overall_sentiment=chatgpt_overall_sentiment, # type: ignore
                chatgpt_sentiment_scores_by_segment=chatgpt_sentiment_scores_by_segment, # type: ignore
                chatgpt_management_confidence_score=chatgpt_management_confidence_score, # type: ignore
                chatgpt_evasiveness_score_q_a=chatgpt_evasiveness_score_q_a, # type: ignore
                chatgpt_key_topics_discussed=chatgpt_key_topics_discussed, # type: ignore
                chatgpt_red_flags_identified=chatgpt_red_flags_identified, # type: ignore
                chatgpt_raw_response_json=chatgpt_raw_response_json, # type: ignore

                chatgpt_request_ms=chatgpt_request_ms, # type: ignore
                chatgpt_parse_ms=chatgpt_parse_ms, # type: ignore
                chatgpt_total_ms=chatgpt_total_ms, # type: ignore

                groq_summary=groq_summary, # type: ignore
                groq_concise_rationale=groq_concise_rationale, # type: ignore
                groq_overall_sentiment=groq_overall_sentiment, # type: ignore
                groq_sentiment_scores_by_segment=groq_sentiment_scores_by_segment, # type: ignore
                groq_management_confidence_score=groq_management_confidence_score, # type: ignore
                groq_evasiveness_score_q_a=groq_evasiveness_score_q_a, # type: ignore
                groq_key_topics_discussed=groq_key_topics_discussed, # type: ignore
                groq_red_flags_identified=groq_red_flags_identified, # type: ignore
                groq_raw_response_json=groq_raw_response_json, # type: ignore

                groq_request_ms=groq_request_ms, # type: ignore
                groq_parse_ms=groq_parse_ms, # type: ignore
                groq_total_ms=groq_total_ms, # type: ignore
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

    def delete_analysis_report(self, report_id):
        """
        Deletes an analysis report by ID.
        Returns True on success, False otherwise.
        """
        report = self.get_report_by_id(report_id)
        if not report:
            return False
        try:
            self.db.session.delete(report)
            self.db.session.commit()
            return True
        except Exception as e:
            self.db.session.rollback()
            print(f"Error deleting analysis report {report_id}: {e}")
            return False
