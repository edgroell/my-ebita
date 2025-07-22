import json
from typing import Optional, Dict, Any

import requests


class NinjasService:
    """
    A service class to interact with API-Ninjas for various data.
    Currently fetches earnings call transcripts (with speaker splits!)
    and basic company profile information.
    """

    def __init__(self, api_key: str):
        """
        Initializes the NinjasService.

        Args:
            api_key (str): The API key for API-Ninjas.
        """
        if not api_key:
            raise ValueError("API_NINJAS_KEY cannot be empty!")

        self.api_key = api_key
        self.base_url = "https://api.api-ninjas.com/v1/"
        self.headers = {
            'X-Api-Key': self.api_key
        }
        print("NinjasService initialized. Ready to fetch data from API Ninjas!")

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Internal method to construct and send a generic GET API Ninjas request.
        Handles common error logging and JSON parsing.

        Args:
            endpoint (str): The specific API Ninjas endpoint (e.g., "earningstranscript", "logo").
            params (Dict[str, Any]): Dictionary of query parameters for the request.

        Returns:
            Optional[Dict[str, Any]]: The JSON response data as a dictionary, or None if no data.

        Raises:
            requests.exceptions.RequestException: For network-related errors.
            ValueError: For API errors (non-2xx status codes) or unexpected response format.
        """
        url = f"{self.base_url}{endpoint}"
        print(f"Making request to {url} with params: {params}...")
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()

            response_json = response.json()

            if isinstance(response_json, list) and not response_json:
                print(f"No data found for the given parameters.")
                return None
            elif isinstance(response_json, list) and response_json:
                return response_json[0]
            elif isinstance(response_json, dict):
                return response_json
            else:
                raise ValueError(
                    f"Unexpected top-level response format from API Ninjas: {type(response_json)}. Full response: {response.text}")

        except requests.exceptions.HTTPError as e:
            error_message = f"API Ninjas returned an HTTP error {e.response.status_code}: {e.response.text}"
            print(f"Error details from API Ninjas: {e.response.text}")
            raise ValueError(error_message) from e
        except requests.exceptions.ConnectionError as e:
            raise requests.exceptions.RequestException(f"Network connection error to API Ninjas: {e}") from e
        except requests.exceptions.Timeout as e:
            raise requests.exceptions.RequestException(f"API Ninjas request timed out: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from API Ninjas response: {e}. Response: {response.text}") from e
        except Exception as e:
            raise Exception(f"An unknown error occurred during API call to API Ninjas: {e}") from e

    def get_earnings_transcript(self, ticker: str, year: int, quarter: int) -> Optional[Dict[str, Any]]:
        """
        Fetches the earnings call transcript for a given company, year, and quarter,
        including speaker segmentation if available.

        Args:
            ticker (str): The stock ticker symbol (e.g., "AAPL", "MSFT").
            year (int): The fiscal year of the earnings call (e.g., 2024).
            quarter (int): The fiscal quarter (1, 2, 3, or 4).

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the transcript date, full text,
                                      and speaker split (if available).
                                      Example: {
                                          "date": "2024-01-30",
                                          "transcript": "...",
                                          "transcript_split": [{"speaker": "Operator", "text": "..."}, ...]
                                      }
                                      Returns None if the transcript is not found or an error occurs.

        Raises:
            ValueError, RequestException (propagated from _make_request)
        """
        if not (1 <= quarter <= 4):
            raise ValueError("Quarter must be between 1 and 4.")
        if not (1990 <= year <= 2100): # Realistic year range
            raise ValueError("Year seems a bit off. Please provide a realistic year.")

        params = {
            'ticker': ticker.upper(),
            'year': year,
            'quarter': quarter
        }

        transcript_data = self._make_request("earningstranscript", params)

        if not transcript_data:
            print(f"No transcript found for {ticker} Q{quarter} {year}. The silence is still deafening.")
            return None

        date_str = transcript_data.get('date')
        transcript_text = transcript_data.get('transcript')
        transcript_split = transcript_data.get('transcript_split')

        if not transcript_text:
            print(f"Transcript text missing in response for {ticker} Q{quarter} {year}. Just empty words.")
            return None

        return {
            "date": date_str,
            "transcript": transcript_text,
            "transcript_split": transcript_split
        }

    def get_company_profile_basic(self, ticker: str) -> Optional[Dict[str, str]]:
        """
        Fetches basic company profile information (name, symbol, logo URL) using API Ninjas' Logo API.

        Args:
            ticker (str): The stock ticker symbol (e.g., "AAPL", "MSFT").

        Returns:
            Optional[Dict[str, str]]: A dictionary containing 'name', 'symbol' (ticker) and 'logo_url',
                                      or None if the company is not found.
                                      Example: {"name": "Microsoft Corp", "symbol": "MSFT", "logo_url": "..."}

        Raises:
            ValueError, RequestException (propagated from _make_request)
        """
        params = {
            'ticker': ticker.upper()
        }

        company_data = self._make_request("logo", params)

        if not company_data:
            print(f"No basic profile found for ticker {ticker}.")
            return None

        company_name = company_data.get('name')
        company_symbol = company_data.get('ticker')
        company_logo_url = company_data.get('image')

        if not company_name or not company_symbol or not company_logo_url:
            print(f"Missing 'name', 'ticker', or 'image' in basic profile for {ticker}. Incomplete data!")
            return None

        return {
            "name": company_name,
            "symbol": company_symbol,
            "logo_url": company_logo_url,
        }


# TESTING (Only runs when ninjas_service.py is executed directly)
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    API_NINJAS_KEY = os.getenv('API_NINJAS_KEY')

    if not API_NINJAS_KEY:
        print("Error: API_NINJAS_KEY environment variable not set for testing.")
    else:
        ninjas_service = NinjasService(api_key=API_NINJAS_KEY)

        # --- Test Case 1: Fetch a known transcript (e.g., Microsoft Q1 2025) ---
        print("\n--- Testing fetching Microsoft Q1 2025 transcript (with split confirmation) ---")
        try:
            msft_transcript = ninjas_service.get_earnings_transcript(ticker="MSFT", year=2025, quarter=1)
            if msft_transcript:
                print(f"Date: {msft_transcript['date']}")
                print(f"Transcript (first 500 chars):\n{msft_transcript['transcript'][:500]}...\n")
                if msft_transcript['transcript_split']:
                    print(
                        f"Transcript Split Detected! First 3 segments:\n{json.dumps(msft_transcript['transcript_split'][:3], indent=2)}\n")
                else:
                    print("Transcript Split NOT detected (unexpected, but possible for some data points).")
            else:
                print("Failed to retrieve MSFT transcript (no data returned).")
        except (ValueError, requests.exceptions.RequestException, Exception) as e:
            print(f"Error retrieving MSFT transcript: {e}")


        # --- Test Case 2: Fetch basic company profile ---
        print("\n--- Testing fetching basic company profile for Google (GOOGL) ---")
        try:
            googl_profile = ninjas_service.get_company_profile_basic(ticker="GOOG")
            if googl_profile:
                print(f"Company Name: {googl_profile.get('name')}")
                print(f"Ticker Symbol: {googl_profile.get('symbol')}\n")
                print(f"Logo URL: {googl_profile.get('logo_url')}\n")
            else:
                print("Failed to retrieve GOOGL company profile (no data returned).")
        except (ValueError, requests.exceptions.RequestException, Exception) as e:
            print(f"Error retrieving GOOGL company profile: {e}")