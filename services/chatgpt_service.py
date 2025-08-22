# ChatGPT service for analyzing earnings call transcripts

from __future__ import annotations

import json
import time
from typing import Any, Optional

from openai import OpenAI
from openai import OpenAIError

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
        raise ImportError(f"Could not load module spec for analysis_result from {_analysis_path}")
    analysis_mod = importlib.util.module_from_spec(spec)
    sys.modules["services.analysis_result"] = analysis_mod
    spec.loader.exec_module(analysis_mod)
    AnalysisResult = analysis_mod.AnalysisResult


class ChatGPTService:
    """
    A service class to interact with OpenAI's ChatGPT models.
    Uses a prompting pattern that instructs the model to reason carefully (chain-of-thought internally)
    while returning a concise, structured JSON object (without exposing internal chain-of-thought).
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        if not api_key:
            raise ValueError("OpenAI API key cannot be empty.")
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        print(f"ChatGPTService initialized with model: {self.model_name}")

    def analyze_transcript(
        self,
        transcript_text: str,
        user_prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.3,
    ) -> "AnalysisResult": # type: ignore
        """
        Sends an earnings call transcript and a user-defined prompt to the ChatGPT model
        for analysis and returns a structured AnalysisResult. The prompt requests careful
        step-by-step internal reasoning (chain-of-thought) but instructs the model NOT to
        output raw internal reasoning. Instead the model must return a valid JSON object
        containing a concise rationale (1-2 sentences) and the structured fields.
        """
        if not transcript_text:
            return AnalysisResult(success=False, error="Transcript text cannot be empty for analysis.", model_used=self.model_name)

        # Prompt uses chain-of-thought technique implicitly: ask model to reason carefully,
        # but explicitly forbid exposing detailed internal chain-of-thought; require concise rationale only.
        system_content = (
            "You are a financial analyst AI specializing in dissecting earnings call transcripts. "
            "When analyzing, internally reason step-by-step to improve accuracy (chain-of-thought). "
            "DO NOT reveal detailed internal chain-of-thought or step-by-step working in the output. "
            "Only return a single valid JSON object with the fields described below. "
            "Include a very short 1-2 sentence 'concise_rationale' explaining the key reasons for your conclusions."
        )

        user_content = (
            f"Here is an earnings call transcript:\n\n---\n{transcript_text}\n---\n\n"
            f"Based on the transcript, {user_prompt}. "
            "Return ONLY a valid JSON object with these keys: "
            '"summary" (string), '
            '"concise_rationale" (string, 1-2 sentences), '
            '"overall_sentiment" (\"Positive\", \"Neutral\", or \"Negative\"), '
            '"management_confidence_score" (integer 0-100), '
            '"evasiveness_score_q_a" (integer 0-100), '
            '"key_topics" (array of 3-5 strings), '
            '"red_flags" (array of strings). '
            "Do NOT include any extraneous text or internal reasoning. Ensure the JSON is well-formed."
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        analysis_content: Optional[Any] = None
        api_elapsed = None
        parse_elapsed = None
        start = time.perf_counter()

        try:
            # Start the API timer immediately before sending the request
            api_start = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
            )

            # Extract the model output safely
            analysis_content = None
            try:
                analysis_content = response.choices[0].message.content
            except Exception:
                try:
                    analysis_content = getattr(response.choices[0].message, "content", None)
                except Exception:
                    analysis_content = None

            if analysis_content is None:
                raise ValueError("Received None for analysis_content, cannot parse JSON.")

            # parse start
            parse_start = time.perf_counter()

            # If the API returned a dict already, use it. Otherwise parse string.
            if isinstance(analysis_content, (dict, list)):
                parsed_content = analysis_content  # type: ignore
            else:
                parsed_content = json.loads(analysis_content)

            # If parsed content is a list with a single dict, accept that too
            if isinstance(parsed_content, list) and parsed_content and isinstance(parsed_content[0], dict):
                parsed_content = parsed_content[0]

            if not isinstance(parsed_content, dict):
                raise ValueError("Parsed AI response is not a JSON object/dict.")

            # parse end
            parse_end = time.perf_counter()

            # api_elapsed is defined as time from api_start until parsing finished
            api_elapsed = (parse_end - api_start) * 1000.0
            parse_elapsed = (parse_end - parse_start) * 1000.0
            total_elapsed = (time.perf_counter() - start) * 1000.0

            result = AnalysisResult.from_dict(parsed_content, model_used=self.model_name)
            result.request_ms = api_elapsed
            result.parse_ms = parse_elapsed
            result.total_ms = total_elapsed
            return result

        except OpenAIError as e:
            total_elapsed = (time.perf_counter() - start) * 1000.0
            print(f"OpenAI API Error: {e}")
            return AnalysisResult(success=False, error=f"Failed to get analysis from ChatGPT: {e}", model_used=self.model_name, request_ms=api_elapsed, parse_ms=parse_elapsed, total_ms=total_elapsed)
        except json.JSONDecodeError as e:
            total_elapsed = (time.perf_counter() - start) * 1000.0
            raw = analysis_content if analysis_content is not None else "Raw response not available"
            print(f"JSON decode error: {e}. Raw: {str(raw)[:400]}")
            return AnalysisResult(success=False, error=f"Failed to parse AI response as JSON: {e}", model_used=self.model_name, request_ms=api_elapsed, parse_ms=parse_elapsed, total_ms=total_elapsed)
        except Exception as e:
            total_elapsed = (time.perf_counter() - start) * 1000.0
            print(f"An unexpected error occurred: {e}")
            return AnalysisResult(success=False, error=f"An unexpected error occurred: {e}", model_used=self.model_name, request_ms=api_elapsed, parse_ms=parse_elapsed, total_ms=total_elapsed)


# TESTING
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set for testing.")
        print("Please set it before running the example: export OPENAI_API_KEY='your_key_here'")
    else:
        print("Testing ChatGPTService with gpt-4o-mini...")
        chatgpt_analyzer = ChatGPTService(api_key=OPENAI_API_KEY, model_name="gpt-4o-mini")

        test_transcript = """
        CEO: "We've had a truly transformative quarter, navigating significant macroeconomic headwinds with unparalleled agility. Our strategic repositioning initiatives are yielding promising preliminary indicators, suggesting robust potential for enhanced shareholder value in the mid-to-long term."
        Analyst: "Can you provide more specific guidance on revenue growth for the next fiscal year, given the recent market volatility?"
        CFO: "As we've stated, our focus remains on operational efficiencies and prudently managing our cost structure. While we are observing certain market fluctuations, our internal projections remain cautiously optimistic regarding our capacity to deliver sustainable returns. We are not providing granular forward-looking revenue guidance at this juncture, preferring to allow our ongoing investments in innovation to speak for themselves."
        """

        test_prompt = """
        Analyze the transcript. Return a structured object with the following keys:
        - "summary": A concise summary of the call (string).
        - "concise_rationale": A 1-2 sentence rationale (string).
        - "overall_sentiment": "Positive", "Neutral", or "Negative" (string).
        - "management_confidence_score": A score from 0 to 100 for management's confidence (integer).
        - "evasiveness_score_q_a": A score from 0 to 100 for evasiveness in Q&A (integer).
        - "key_topics": A list of 3-5 main topics discussed (array of strings).
        - "red_flags": A list of any specific red flags or evasive phrases identified (array of strings).
        """

        analysis_result = chatgpt_analyzer.analyze_transcript(test_transcript, test_prompt, temperature=0.2)

        if analysis_result.success:
            print(f"\n--- ChatGPT Analysis ({analysis_result.model_used}) ---")
            print(json.dumps(analysis_result.to_dict(), indent=2))
        else:
            print("\n--- ChatGPT Analysis Error ---")
            print(analysis_result.error)
