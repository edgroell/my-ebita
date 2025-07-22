import os
import json

import requests
from dotenv import load_dotenv

load_dotenv()


class GeminiService:
    """
    A service class to interact with the Google Gemini models (e.g., gemini-2.5-flash).
    Designed to provide analysis on earnings call transcripts.
    """

    def __init__(self, default_model: str = "gemini-2.5-flash"):
        """
        Initializes the GeminiService.

        Args:
            default_model (str): The default Gemini model to use for content generation.
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found!")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/"
        self.default_model = default_model
        self.headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': self.api_key
        }

    def _make_request(self, model_name: str, contents: list) -> dict:
        """
        Internal method to construct and send the API request.

        Args:
            model_name (str): The specific Gemini model to target.
            contents (list): The list of content parts for the request body.

        Returns:
            dict: The JSON response from the API.

        Raises:
            requests.exceptions.RequestException: For network-related errors.
            ValueError: For API errors (non-2xx status codes) or unexpected response format.
        """
        url = f"{self.base_url}{model_name}:generateContent"
        payload = {"contents": contents}

        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.HTTPError as e:
            error_message = f"Gemini API returned an HTTP error {e.response.status_code}: {e.response.text}"
            print(f"Error details: {json.dumps(e.response.json(), indent=2)}")
            raise ValueError(error_message) from e
        except requests.exceptions.ConnectionError as e:
            raise requests.exceptions.RequestException(f"Network connection error to Gemini API: {e}") from e
        except requests.exceptions.Timeout as e:
            raise requests.exceptions.RequestException(f"Gemini API request timed out: {e}") from e
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"An unexpected request error occurred: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from Gemini API response: {e}. Response: {response.text}") from e
        except Exception as e:
            raise Exception(f"An unknown error occurred during API call: {e}") from e

    def generate_content(self, prompt_text: str, model_name: str = None) -> str:
        """
        Generates content from a text prompt using the Gemini API.

        Args:
            prompt_text (str): The text input for the AI.
            model_name (str, optional): Override the default model for this specific call.

        Returns:
            str: The generated text content from the AI.

        Raises:
            ValueError: If the API response does not contain expected text.
            (Other exceptions propagated from _make_request)
        """
        model_to_use = model_name if model_name else self.default_model

        contents = [
            {"parts": [{"text": prompt_text}]}
        ]

        response_data = self._make_request(model_to_use, contents)

        try:
            generated_text = response_data['candidates'][0]['content']['parts'][0]['text']

            return generated_text

        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Unexpected response format from Gemini API. "
                f"Could not extract text. Error: {e}. Full response: {json.dumps(response_data, indent=2)}"
            )


# TESTING

Prompt_Content = """
You are a financial analyst AI specializing in dissecting earnings call transcripts. Your goal is to provide concise, factual, and insightful analysis, identifying sentiment, key topics, and any signs of management spin or evasiveness. Focus on the financial implications.
Given the following earnings call transcript: 
CEO: "We've had a truly transformative quarter, navigating significant macroeconomic headwinds with unparalleled agility. Our strategic repositioning initiatives are yielding promising preliminary indicators, suggesting robust potential for enhanced shareholder value in the mid-to-long term."
Analyst: "Can you provide more specific guidance on revenue growth for the next fiscal year, given the recent market volatility?"
CFO: "As we've stated, our focus remains on operational efficiencies and prudently managing our cost structure. While we are observing certain market fluctuations, our internal projections remain cautiously optimistic regarding our capacity to deliver sustainable returns. We are not providing granular forward-looking revenue guidance at this juncture, preferring to allow our ongoing investments in innovation to speak for themselves."
Identify the overall sentiment of the earnings call, specifically noting any corporate jargon or evasive language used by management regarding future guidance.
"""
gemini_service = GeminiService()
ai_explanation = gemini_service.generate_content(Prompt_Content)
print(f"AI Explanation: {ai_explanation}\n")
