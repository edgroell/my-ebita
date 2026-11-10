# DataManager: Data Access Layer for the application

from typing import Optional, List, Any
from datetime import datetime

from .data_models import User, Company, EarningsCallTranscript, AnalysisReport
from sqlalchemy import select, or_
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
    def create_user(self, username: str, email: str, password: str) -> User | None:
        """
        Create a new user with hashed password.
        Returns the created User object or None if creation fails.
        """
        try:
            # Validate username format
            if not User.validate_username_format(username):
                print(f"Invalid username format: {username}")
                return None
            
            # Check if user already exists
            if self.get_user_by_username(username):
                print(f"Username already exists: {username}")
                return None
            
            if self.get_user_by_email(email):
                print(f"Email already exists: {email}")
                return None
            
            # Create new user
            new_user = User()
            new_user.username = username
            new_user.email = email
            new_user.set_password(password)
            
            self.db.session.add(new_user)
            self.db.session.commit()
            
            print(f"User '{username}' created successfully with ID {new_user.user_id}")
            return new_user
            
        except Exception as e:
            self.db.session.rollback()
            print(f"Error creating user: {e}")
            return None

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Retrieves a user by their ID."""
        return self.db.session.get(User, user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Retrieves a user by their username."""
        stmt = select(User).filter_by(username=username)
        return self.db.session.execute(stmt).scalar_one_or_none()
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Retrieves a user by their email address."""
        stmt = select(User).filter_by(email=email)
        return self.db.session.execute(stmt).scalar_one_or_none()


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
            new_company = Company()
            new_company.ticker_symbol = ticker_symbol
            new_company.company_name = company_name
            new_company.logo_url = logo_url
            # Set optional attributes only if the model defines them
            if industry is not None and hasattr(Company, "industry"):
                setattr(new_company, "industry", industry)
            if sector is not None and hasattr(Company, "sector"):
                setattr(new_company, "sector", sector)
            if exchange is not None and hasattr(Company, "exchange"):
                setattr(new_company, "exchange", exchange)

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
    
    def get_all_companies(self, search_query: Optional[str] = None, industry: Optional[str] = None,
                          sector: Optional[str] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Company]:
        """
        Retrieves all companies, with optional search and filtering.
        """
        stmt = select(Company)
        if search_query:
            like = f'%{search_query}%'
            stmt = stmt.where(or_(Company.company_name.ilike(like),
                                  Company.ticker_symbol.ilike(like)))
        if industry:
            industry_col = getattr(Company, "industry", None)
            if industry_col is not None:
                stmt = stmt.where(industry_col == industry)
        if sector:
            sector_col = getattr(Company, "sector", None)
            if sector_col is not None:
                stmt = stmt.where(sector_col == sector)

        if limit is not None:
            stmt = stmt.limit(limit)
        if offset is not None:
            stmt = stmt.offset(offset)

        result = self.db.session.execute(stmt)
        return result.scalars().all()
    
    def get_company_by_ticker(self, ticker_symbol: str):
        """Return company object by ticker symbol, or None if not found."""
        return Company.query.filter_by(ticker_symbol=ticker_symbol).first()

    # --- Earnings Call Transcript Management ---
    def add_transcript(self, user_id: int, ticker_symbol: str, fiscal_year: int, fiscal_quarter: int,
                       call_date: Optional[datetime] , transcript_raw: Any,
                       source_url: Optional[str] = None,
                       transcript_split: Optional[List[str]] = None,
                       ) -> Optional[EarningsCallTranscript]:
        """
        Adds a new earnings call transcript associated with a specific user.
        Returns the Transcript object on success, None on error (e.g., duplicate entry).
        """
        try:
            new_transcript = EarningsCallTranscript()
            new_transcript.user_id = user_id
            new_transcript.fiscal_year = fiscal_year
            new_transcript.fiscal_quarter = fiscal_quarter
            new_transcript.call_date = call_date
            new_transcript.transcript_raw = transcript_raw
            if transcript_split is not None:
                new_transcript.transcript_split = transcript_split
            new_transcript.source_url = source_url
            new_transcript.ticker_symbol = ticker_symbol
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
        return self.db.session.get(EarningsCallTranscript, transcript_id)

    def get_transcripts_for_user(self, user_id: int) -> List[EarningsCallTranscript]:
        """Retrieves all earnings call transcripts owned by a specific user."""
        stmt = select(EarningsCallTranscript).filter_by(user_id=user_id).order_by(
            EarningsCallTranscript.call_date.desc()
        )
        result = self.db.session.execute(stmt)
        return result.scalars().all()
    
    def get_transcripts_for_company(self, ticker_symbol: str) -> List[EarningsCallTranscript]:
        """Retrieves all transcripts for a given company (across users)."""
        stmt = select(EarningsCallTranscript).filter_by(ticker_symbol=ticker_symbol).order_by(
            EarningsCallTranscript.fiscal_year.desc(),
            EarningsCallTranscript.fiscal_quarter.desc()
        )
        result = self.db.session.execute(stmt)
        return result.scalars().all()
    
    def get_transcript_by_details(self, ticker_symbol: str, fiscal_year: int, fiscal_quarter: int) -> Optional[EarningsCallTranscript]:
        """Retrieves a specific transcript by company and fiscal period (any user)."""
        stmt = select(EarningsCallTranscript).filter_by(
            ticker_symbol=ticker_symbol,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter
        )
        return self.db.session.execute(stmt).scalar_one_or_none()

    def delete_transcript(self, transcript_id: int) -> bool:
        """Delete a transcript and all its associated reports"""
        try:
            from .data_models import EarningsCallTranscript, AnalysisReport
            
            # Delete associated reports first
            AnalysisReport.query.filter_by(transcript_id=transcript_id).delete()
            
            # Delete the transcript
            transcript = EarningsCallTranscript.query.get(transcript_id)
            if transcript:
                self.db.session.delete(transcript)
                self.db.session.commit()
                print(f"✓ Deleted transcript {transcript_id}")
                return True
            return False
        except Exception as e:
            self.db.session.rollback()
            print(f"✗ Error deleting transcript: {e}")
            return False
    
    def create_analysis_report(
        self,
        user_id: int,
        transcript_id: int,
        mistral_analysis_json: Optional[Any] = None,
    ) -> Optional[AnalysisReport]:
        """
        Creates a new AI analysis report.
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

            new_report = AnalysisReport()
            new_report.user_id = user_id
            new_report.transcript_id = transcript_id
            new_report.mistral_analysis_json = mistral_analysis_json

            self.db.session.add(new_report)
            self.db.session.commit()
            return new_report
        
        except Exception as e:
            self.db.session.rollback()
            print(f"Error creating analysis report: {e}")
            return None

    def get_report_by_id(self, report_id: int) -> Optional[AnalysisReport]:
        """Retrieves an analysis report by its ID."""
        return self.db.session.get(AnalysisReport, report_id)

    def get_reports_for_user(self, user_id: int) -> List[AnalysisReport]:
        """Retrieves all analysis reports for a specific user."""
        stmt = select(AnalysisReport).filter_by(user_id=user_id).order_by(AnalysisReport.analysis_date.desc())
        result = self.db.session.execute(stmt)
        return result.scalars().all()

    def get_latest_report_for_user_and_transcript(self, user_id: int, transcript_id: int) -> Optional[AnalysisReport]:
        """Retrieves the latest analysis report for a given user and transcript."""
        stmt = select(AnalysisReport).filter_by(
            user_id=user_id,
            transcript_id=transcript_id
        ).order_by(AnalysisReport.analysis_date.desc())
        return self.db.session.execute(stmt).scalars().first()

    def get_reports_for_transcript(self, transcript_id: int) -> List[AnalysisReport]:
        """Retrieves all analysis reports for a specific transcript."""
        stmt = select(AnalysisReport).filter_by(transcript_id=transcript_id).order_by(AnalysisReport.analysis_date.desc())
        result = self.db.session.execute(stmt)
        return result.scalars().all()

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
