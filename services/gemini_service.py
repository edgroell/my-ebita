# Gemini service for interacting with Google Gemini models

import json
import time
import requests
from typing import Any, Optional, Dict

# Attempt to import AnalysisResult from the same directory
try:
    from .analysis_result import AnalysisResult
except Exception:
    import importlib.util
    import sys
    from pathlib import Path

    _analysis_path = Path(__file__).resolve().parent / "analysis_result.py"
    spec = importlib.util.spec_from_file_location("services.analysis_result", str(_analysis_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create a module spec for analysis_result.py at {_analysis_path}")
    analysis_mod = importlib.util.module_from_spec(spec)
    sys.modules["services.analysis_result"] = analysis_mod
    spec.loader.exec_module(analysis_mod)
    AnalysisResult = analysis_mod.AnalysisResult


class GeminiService:
    """
    A service class to interact with the Google Gemini models (e.g., gemini-2.5-flash).
    Uses a prompting pattern that instructs the model to reason carefully (chain-of-thought internally)
    while returning a concise, structured JSON object (without exposing internal chain-of-thought).
    """

    def __init__(self, api_key: str, default_model: str = "gemini-2.5-flash"):
        if not api_key:
            raise ValueError("Gemini API key cannot be empty.")
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/"
        self.default_model = default_model
        self.headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key,
        }
        print(f"GeminiService initialized with model: {self.default_model}")

    def _make_request(self, model_name: str, contents: list[dict[str, Any]], temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Internal method to construct and send the Gemini API request.
        """
        url = f"{self.base_url}{model_name}:generateContent"
        generation_config: dict[str, Any] = {"responseMimeType": "application/json"}
        if temperature is not None:
            generation_config["temperature"] = float(temperature)

        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation_config}

        response = None
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            try:
                details = e.response.json()
            except Exception:
                details = e.response.text if e.response is not None else "No response body"
            raise ValueError(f"Gemini API HTTP error {getattr(e.response, 'status_code', 'N/A')}: {details}") from e
        except requests.exceptions.ConnectionError as e:
            raise requests.exceptions.RequestException(f"Network connection error to Gemini API: {e}") from e
        except requests.exceptions.Timeout as e:
            raise requests.exceptions.RequestException(f"Gemini API request timed out: {e}") from e
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"An unexpected request error occurred: {e}") from e
        except json.JSONDecodeError as e:
            resp_text = response.text if response is not None else "No response"
            raise ValueError(f"Failed to decode JSON from Gemini API response: {e}. Response: {resp_text}") from e
        except Exception as e:
            raise Exception(f"An unknown error occurred during Gemini API call: {e}") from e

    def generate_content(self, prompt_text: str, model_name: Optional[str] = None, temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate content from Gemini and return a dict with:
          - parsed (dict): the parsed JSON object returned by the model
          - api_elapsed_ms (float): time from just before the HTTP request until parsing finished (ms)
          - parse_elapsed_ms (float): time spent parsing/validating the response (ms)
        """
        model_to_use = model_name if model_name else self.default_model
        contents = [{"parts": [{"text": prompt_text}]}]

        api_start = time.perf_counter()
        response_data = self._make_request(model_to_use, contents, temperature=temperature)
        # response received here
        parse_start = time.perf_counter()

        raw_generated_text: Optional[str] = None
        try:
            raw_generated_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
            if raw_generated_text is None:
                raise ValueError("Gemini returned no text content in the candidate response.")
            parsed = json.loads(raw_generated_text)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                parsed = parsed[0]
            if not isinstance(parsed, dict):
                raise ValueError("Gemini returned JSON that is not an object/dict.")
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Unexpected response format from Gemini API. Could not extract text. Error: {e}. Full response: {json.dumps(response_data, indent=2)}"
            ) from e
        except json.JSONDecodeError as e:
            preview = raw_generated_text[:500] if raw_generated_text else "No content available"
            raise ValueError(f"Failed to decode JSON from Gemini response: {e}. Raw content preview: {preview}") from e
        finally:
            parse_end = time.perf_counter()

        api_elapsed_ms = (parse_end - api_start) * 1000.0
        parse_elapsed_ms = (parse_end - parse_start) * 1000.0

        return {"parsed": parsed, "api_elapsed_ms": api_elapsed_ms, "parse_elapsed_ms": parse_elapsed_ms}

    def analyze_transcript(self, transcript_text: str, user_prompt: str, model_name: Optional[str] = None, temperature: float = 0.3) -> AnalysisResult: # type: ignore
        """
        Analyze an earnings call transcript using Gemini and return an AnalysisResult.
        Prompt mirrors ChatGPTService: request internal chain-of-thought reasoning but forbid exposing it,
        require concise_rationale and structured JSON output.
        """
        if not transcript_text:
            return AnalysisResult(success=False, error="Transcript text cannot be empty for analysis.", model_used=model_name or self.default_model)

        model_to_use = model_name if model_name else self.default_model

        system_instructions = (
            "You are a financial analyst AI specializing in dissecting earnings call transcripts. "
            "When analyzing, internally reason step-by-step to improve accuracy (chain-of-thought). "
            "DO NOT reveal detailed internal chain-of-thought or step-by-step working in the output. "
            "Only return a single valid JSON object with the fields described below. "
            "Include a very short 1-2 sentence 'concise_rationale' explaining the key reasons for your conclusions."
        )

        user_instructions = (
            f"Here is an earnings call transcript:\n\n---\n{transcript_text}\n---\n\n"
            f"Based on the transcript, {user_prompt}. "
            "Return ONLY a valid JSON object with these keys: "
            '"summary" (string), '
            '"concise_rationale" (string, 1-2 sentences), '
            '"overall_sentiment" ("Positive", "Neutral", or "Negative"), '
            '"management_confidence_score" (integer 0-100), '
            '"evasiveness_score_q_a" (integer 0-100), '
            '"key_topics" (array of 3-5 strings), '
            '"red_flags" (array of strings). '
            "Do NOT include any extraneous text or internal reasoning. Ensure the JSON is well-formed."
        )

        # Gemini API doesn't use roles the same way as chat models; combine instructions into one prompt
        full_prompt = f"{system_instructions}\n\n{user_instructions}"

        analysis_content: Optional[dict[str, Any]] = None
        api_elapsed = None
        parse_elapsed = None
        start = time.perf_counter()

        try:
            # Start the API timer immediately before making the request / generating content.
            api_start = time.perf_counter()
            gen_result = self.generate_content(full_prompt, model_name=model_to_use, temperature=temperature)
            # generate_content now returns parsed + timing info
            parse_end = time.perf_counter()

            # Use the timings reported by generate_content (keeps parity with ChatGPTService)
            api_elapsed = gen_result.get("api_elapsed_ms")
            parse_elapsed = gen_result.get("parse_elapsed_ms")
            total_elapsed = (time.perf_counter() - start) * 1000.0

            parsed_analysis = gen_result["parsed"]
            if not isinstance(parsed_analysis, dict):
                raise ValueError("Parsed Gemini output is not a dict.")

            result = AnalysisResult.from_dict(parsed_analysis, model_used=model_to_use)
            result.request_ms = api_elapsed
            result.parse_ms = parse_elapsed
            result.total_ms = total_elapsed
            return result

        except Exception as e:
            total_elapsed = (time.perf_counter() - start) * 1000.0
            err_msg = f"Failed to get analysis from Gemini: {e}"
            return AnalysisResult(success=False, error=err_msg, model_used=model_to_use, request_ms=api_elapsed, parse_ms=parse_elapsed, total_ms=total_elapsed)


# TESTING
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set for testing.")
    else:
        print("Testing GeminiService with gemini-2.5-flash...")
        gemini = GeminiService(api_key=GEMINI_API_KEY, default_model="gemini-2.5-flash")

        test_transcript = """
        CEO: "We've had a truly transformative quarter..."
        Analyst: "Can you provide more specific guidance on revenue growth?"
        CFO: "We are not providing granular forward-looking revenue guidance at this juncture..."
        """
        test_user_prompt = """
        Analyze the transcript. Return a structured object with the same keys as ChatGPTService:
        summary, concise_rationale, overall_sentiment, management_confidence_score, evasiveness_score_q_a,
        key_topics, red_flags.
        """

        res = gemini.analyze_transcript(test_transcript, test_user_prompt, temperature=0.3)
        if res.success:
            print(json.dumps(res.to_dict(), indent=2))
        else:
            print(f"Error: {res.error}")
