import json

import requests


class GeminiService:
    """
    A service class to interact with the Google Gemini models (e.g., gemini-2.5-flash).
    Designed to provide analysis on earnings call transcripts.
    """

    def __init__(self, api_key: str, default_model: str = "gemini-2.5-flash"):
        """
        Initializes the GeminiService.

        Args:
            api_key (str): The API key to authenticate with the Gemini API.
            default_model (str): The default Gemini model to use for content generation.
        """
        if not api_key:
            raise ValueError("Gemini API key cannot be empty.")

        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/"
        self.default_model = default_model
        self.headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': self.api_key
        }
        print(f"GeminiService initialized with model: {self.default_model}")

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

    def analyze_transcript(self, transcript_text: str, user_prompt: str, model_name: str = None) -> dict:
        """
        Sends an earnings call transcript and a user-defined prompt to the Gemini model
        for analysis. This method matches the signature of ChatGPTService.analyze_transcript.

        Args:
            transcript_text (str): The full text of the earnings call transcript.
            user_prompt (str): The specific instruction for the AI (e.g., "Summarize key points,
                                identify overall sentiment, and highlight any evasive language
                                from management regarding future guidance.").
            model_name (str, optional): Override the default model for this specific call.

        Returns:
            dict: A dictionary containing the AI's analysis, or an error message.
                  Example: {"analysis": "...", "model_used": "gemini-2.5-flash", "success": True}
        """
        model_to_use = model_name if model_name else self.default_model

        # Combine transcript and user prompt into a single, comprehensive prompt
        full_prompt = (
            "You are a financial analyst AI specializing in dissecting earnings call transcripts. "
            "Your goal is to provide concise, factual, and insightful analysis, identifying sentiment, "
            "key topics, and any signs of management spin or evasiveness. Focus on the financial implications.\n\n"
            f"Here is an earnings call transcript:\n\n---\n{transcript_text}\n---\n\n"
            f"Based on the transcript, {user_prompt}"
        )

        try:
            generated_text = self.generate_content(full_prompt, model_name=model_to_use)
            return {
                "analysis": generated_text,
                "model_used": model_to_use,
                "success": True
            }
        except (ValueError, requests.exceptions.RequestException, Exception) as e:
            return {"error": f"Failed to get analysis from Gemini: {e}", "success": False}

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


# TESTING (Only runs when gemini_service.py is executed directly)
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set for testing.")
    else:
        print("Testing GeminiService with gemini-2.5-flash...")
        gemini_analyzer = GeminiService(api_key=GEMINI_API_KEY)

        test_transcript = """
        CEO: "We've had a truly transformative quarter, navigating significant macroeconomic headwinds with unparalleled agility. Our strategic repositioning initiatives are yielding promising preliminary indicators, suggesting robust potential for enhanced shareholder value in the mid-to-long term."
        Analyst: "Can you provide more specific guidance on revenue growth for the next fiscal year, given the recent market volatility?"
        CFO: "As we've stated, our focus remains on operational efficiencies and prudently managing our cost structure. While we are observing certain market fluctuations, our internal projections remain cautiously optimistic regarding our capacity to deliver sustainable returns. We are not providing granular forward-looking revenue guidance at this juncture, preferring to allow our ongoing investments in innovation to speak for themselves."
        """
        test_user_prompt = """
        Analyze the transcript. Provide the response as a JSON object with the following keys:
        - "summary": A concise summary of the call (string).
        - "overall_sentiment": "Positive", "Neutral", or "Negative" (string).
        - "management_confidence_score": A score from 0 to 100 for management's confidence (integer).
        - "evasiveness_score_q_a": A score from 0 to 100 for evasiveness in Q&A (integer).
        - "key_topics": A list of 3-5 main topics discussed (array of strings).
        - "red_flags": A list of any specific red flags or evasive phrases identified (array of strings).
        """

        try:
            analysis_result = gemini_analyzer.analyze_transcript(test_transcript, test_user_prompt)
            if analysis_result["success"]:
                print(f"\n--- Gemini Analysis ({analysis_result['model_used']}) ---")
                print(analysis_result["analysis"])
            else:
                print(f"\n--- Gemini Analysis Error ---")
                print(analysis_result["error"])
        except Exception as e:
            print(f"\n--- Unexpected Error During Gemini Analysis ---")
            print(f"An unexpected error occurred: {e}")