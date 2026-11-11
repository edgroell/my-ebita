"""Client for fetching earnings call transcripts from Alpha Vantage."""

from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse

import requests


class AlphaVantageService:
    """Fetch earnings call transcripts from the Alpha Vantage API."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY cannot be empty.")

        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def _request_json(self, params: Dict[str, str]) -> Dict[str, Any]:
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            response_data = response.json()
        except requests.exceptions.HTTPError as error:
            response_text = error.response.text if error.response is not None else str(error)
            raise ValueError(
                f"Alpha Vantage returned an HTTP error: {response_text}"
            ) from error
        except requests.exceptions.JSONDecodeError as error:
            raise ValueError(
                f"Failed to decode JSON from Alpha Vantage response: {error}"
            ) from error

        if not isinstance(response_data, dict):
            raise ValueError(
                "Unexpected top-level response format from Alpha Vantage: "
                f"{type(response_data).__name__}."
            )

        api_error = (
            response_data.get("Error Message")
            or response_data.get("Note")
            or response_data.get("Information")
        )
        if api_error:
            raise ValueError(f"Alpha Vantage API error: {api_error}")

        return response_data

    @staticmethod
    def is_valid_logo_url(value: Optional[str]) -> bool:
        """Return whether a stored logo value is a usable HTTP(S) URL."""
        if not isinstance(value, str):
            return False
        parsed_url = urlparse(value.strip())
        return parsed_url.scheme in {"http", "https"} and bool(parsed_url.netloc)

    def get_company_profile_basic(self, ticker: str) -> Optional[Dict[str, str]]:
        """Fetch a company's display name and logo URL."""
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty.")

        symbol = ticker.strip().upper()
        overview = self._request_json(
            {
                "function": "OVERVIEW",
                "symbol": symbol,
                "apikey": self.api_key,
            }
        )
        company_name = overview.get("Name")
        if not isinstance(company_name, str) or not company_name.strip():
            return None

        logo = self._request_json(
            {
                "function": "COMPANY_LOGO",
                "symbol": symbol,
                "apikey": self.api_key,
            }
        )
        logo_url = logo.get("logo_url_png") or logo.get("logo_url_svg")
        if not self.is_valid_logo_url(logo_url):
            logo_url = ""
            official_site = overview.get("OfficialSite")
            if isinstance(official_site, str) and official_site.strip():
                official_site = official_site.strip()
                if not urlparse(official_site).scheme:
                    official_site = f"https://{official_site}"
                parsed_site = urlparse(official_site)
                if parsed_site.scheme in {"http", "https"} and parsed_site.netloc:
                    logo_url = (
                        "https://www.google.com/s2/favicons?"
                        + urlencode({"domain": parsed_site.netloc, "sz": "128"})
                    )

        return {
            "name": company_name.strip(),
            "symbol": symbol,
            "logo_url": logo_url if self.is_valid_logo_url(logo_url) else "",
        }

    def get_earnings_transcript(
        self, ticker: str, year: int, quarter: int
    ) -> Optional[Dict[str, Any]]:
        """Fetch and normalize a company's transcript for a fiscal quarter.

        Returns the transcript text and speaker segments in the same shape as
        NinjasService. Alpha Vantage does not include a call date in this
        endpoint's response, so ``date`` is ``None``.
        """
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty.")
        if not 1900 <= year <= 2100:
            raise ValueError("Year seems a bit off. Please provide a realistic year.")
        if not 1 <= quarter <= 4:
            raise ValueError("Quarter must be between 1 and 4.")

        params = {
            "function": "EARNINGS_CALL_TRANSCRIPT",
            "symbol": ticker.strip().upper(),
            "quarter": f"{year}Q{quarter}",
            "apikey": self.api_key,
        }

        response_data = self._request_json(params)

        segments = response_data.get("transcript")
        if not segments:
            return None
        if not isinstance(segments, list) or not all(
            isinstance(segment, dict) for segment in segments
        ):
            raise ValueError("Unexpected transcript format from Alpha Vantage.")

        transcript_split: List[Dict[str, Any]] = []
        transcript_parts: List[str] = []
        for segment in segments:
            content = segment.get("content")
            if not isinstance(content, str) or not content.strip():
                continue

            speaker = segment.get("speaker") or "Unknown"
            title = segment.get("title")
            transcript_split.append(
                {
                    "speaker": speaker,
                    "text": content,
                    **({"title": title} if title else {}),
                    **(
                        {"sentiment": segment["sentiment"]}
                        if segment.get("sentiment") is not None
                        else {}
                    ),
                }
            )
            transcript_parts.append(f"{speaker}: {content}")

        if not transcript_parts:
            return None

        return {
            "date": response_data.get("date"),
            "transcript": "\n\n".join(transcript_parts),
            "transcript_split": transcript_split,
        }
