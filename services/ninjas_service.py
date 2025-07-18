import os
import json
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()


class NinjasService:
    """
    A service class to interact with the API-Ninjas Earnings Call Transcript API.
    Fetches raw earnings call transcripts for specified companies and quarters,
    now with confirmed speaker split data!
    """

    def __init__(self):
        """
        Initializes the NinjasService.
        """
        self.api_key = os.getenv('API_NINJAS_KEY')
        if not self.api_key:
            raise ValueError("API_NINJAS_KEY not found.")
        self.base_url = "https://api.api-ninjas.com/v1/earningstranscript"
        self.headers = {
            'X-Api-Key': self.api_key
        }
        print("NinjasService initialized. Ready to fetch transcripts (with glorious splits)!")

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
            requests.exceptions.RequestException: For network-related errors.
            ValueError: For API errors (non-2xx status codes) or unexpected response format.
        """
        if not (1 <= quarter <= 4):
            raise ValueError("Quarter must be between 1 and 4.")
        if not (1990 <= year <= 2100):
            raise ValueError("Year seems a bit off. Please provide a realistic year.")

        params = {
            'ticker': ticker.upper(),
            'year': year,
            'quarter': quarter
        }

        print(f"Fetching transcript for {ticker.upper()} Q{quarter} {year} (expecting split!)...")
        try:
            response = requests.get(self.base_url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()

            response_json = response.json()

            if isinstance(response_json, list) and not response_json:
                print(f"No transcript found for {ticker} Q{quarter} {year}. The silence is still deafening.")
                return None
            elif isinstance(response_json, list) and response_json:
                transcript_data = response_json[0]
            elif isinstance(response_json, dict):
                transcript_data = response_json
            else:
                raise ValueError(
                    f"Unexpected top-level response format from API Ninjas: {type(response_json)}. Full response: {response.text}")

            # Extract the necessary fields, including the split
            date = transcript_data.get('date')
            transcript_text = transcript_data.get('transcript')
            transcript_split = transcript_data.get('transcript_split')

            if not transcript_text:
                print(f"Transcript text missing in response for {ticker} Q{quarter} {year}. Just empty words.")
                return None

            return {
                "date": date,
                "transcript": transcript_text,
                "transcript_split": transcript_split
            }

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
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Unexpected response format from API Ninjas. "
                f"Could not extract expected data. Error: {e}. Full response: {json.dumps(data, indent=2)}"
            )
        except Exception as e:
            raise Exception(f"An unknown error occurred during API call to NinjasService: {e}") from e


ninjas_service = NinjasService()
print("\n--- Testing fetching Microsoft Q2 2024 transcript (with split confirmation) ---")
msft_transcript = ninjas_service.get_earnings_transcript(ticker="MSFT", year=2024, quarter=2)
if msft_transcript:
    print(f"Date: {msft_transcript['date']}")
    print(f"Transcript (first 500 chars):\n{msft_transcript['transcript'][:500]}...\n")
    if msft_transcript['transcript_split']:
        print(
            f"Transcript Split Detected! First 3 segments:\n{json.dumps(msft_transcript['transcript_split'][:3], indent=2)}\n")
    else:
        print("Transcript Split NOT detected (unexpected, but possible for some data points).")
else:
    print("Failed to retrieve MSFT transcript.")
