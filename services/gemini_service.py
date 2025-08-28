# Gemini service for interacting with Google Gemini models

import json
import time

from pydantic import BaseModel
from google import genai


class AnalysisResult(BaseModel):
    summary: str
    concise_rationale: str
    overall_sentiment: str
    management_confidence_score: int
    evasiveness_score_q_a: int
    key_topics: list[str]
    red_flags: list[str]
    request_time_ms: float
    model_used: str


class GeminiService:
    """
    A service class to interact with the Google Gemini models (e.g., gemini-2.5-flash).
    Uses a prompting pattern that instructs the model to reason carefully (chain-of-thought internally)
    while returning a concise, structured JSON object (without exposing internal chain-of-thought).
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro"):
        if not api_key:
            raise ValueError("Gemini API key cannot be empty.")
        self.api_key = api_key
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        print(f"GeminiService initialized with model: {self.model_name}")

    def analyze_transcript(
        self,
        transcript_text: str,
        analysis_result_cls: type[AnalysisResult] = AnalysisResult
    ) -> AnalysisResult:
        """
        Sends an earnings call transcript and a user-defined prompt to the ChatGPT model
        for analysis and returns a structured AnalysisResult. The prompt requests careful
        step-by-step internal reasoning (chain-of-thought) but instructs the model NOT to
        output raw internal reasoning. Instead the model must return a valid JSON object
        containing a concise rationale (1-2 sentences) and the structured fields.
        """
        if not transcript_text:
            raise ValueError("Parameter transcript_text cannot be empty.")

        messages_instructions = (
            "You are a financial analyst AI specializing in dissecting earnings call transcripts. "
            "When analyzing, internally reason step-by-step to improve accuracy (chain-of-thought). "
            "DO NOT reveal detailed internal chain-of-thought or step-by-step working in the output. "
            "Include a very short 1-2 sentence 'concise_rationale' explaining the key reasons for your conclusions."
            f"Here is an earnings call transcript:\n\n---\n{transcript_text}\n---\n\n"
            'Based on the transcript, try to return a structured output with these keys: '
            '"summary" (string, 3-5 sentences providing a brief overview of the key points), '
            '"concise_rationale" (string, 1-2 sentences explaining the key reasons for your conclusions), '
            '"overall_sentiment" (' 
            ' \"Positive\" if the outlook of the company is optimistic from an investor point of view,'
            ' \"Neutral\" if the outlook of the company is neither optimistic nor pessimistic from an investor point of view, or '
            ' \"Negative\" if the outlook of the company is pessimistic from an investor point of view'
            '), '
            '"management_confidence_score" (integer 0-100, providing a measure of how confident management is in their guidance, '
            '0 meaning not confident at all and 100 meaning very confident), '
            '"evasiveness_score_q_a" (integer 0-100, assessing how evasive management was during Q&A,'
            '0 meaning not evasive at all and 100 meaning very evasive), '
            '"key_topics" (array of 3-5 strings, highlighting the main topics discussed), '
            '"red_flags" (array of 3-5 strings, noting any potential concerns raised). '
            "Do NOT include any extraneous text or internal reasoning. Ensure the structured output is well-formed."
        )

        request_elapsed = None
        request_start = time.perf_counter()

        params = {
            "model": self.model_name,
            "contents": messages_instructions,  # type: ignore
        }

        try:
            completion = self.client.models.generate_content(
                **params,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": list[analysis_result_cls],
                },
            )

            analysis_content = None
            try:
                parsed = getattr(completion, "parsed", None)
                if isinstance(parsed, list):
                    if not parsed:
                        raise ValueError("Parsed response is an empty list")
                    analysis_content = parsed[0]
                else:
                    analysis_content = parsed

            except Exception as parse_exc:
                raise ValueError(f"Failed to parse structured response: {parse_exc}") from parse_exc

            if analysis_content is None:
                raise ValueError("Received None for analysis_content.")

            request_end = time.perf_counter()
            request_elapsed = (request_end - request_start) * 1000.0

            try:
                setattr(analysis_content, "request_time_ms", request_elapsed)
                setattr(analysis_content, "model_used", self.model_name)
            except Exception:
                pass

            return analysis_content

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse AI response as JSON: {e}")

        except Exception as e:
            raise ValueError(f"Failed to get analysis from Gemini: {e}")
        

# TESTING
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set for testing.")
    else:
        gemini_analyzer = GeminiService(api_key=GEMINI_API_KEY, model_name="gemini-2.5-flash")
        print(f"--- Testing GeminiService with {gemini_analyzer.model_name} ---")

        test_transcript = """
        CEO: "We've had a truly transformative quarter, navigating significant macroeconomic headwinds with unparalleled agility. Our strategic repositioning initiatives are yielding promising preliminary indicators, suggesting robust potential for enhanced shareholder value in the mid-to-long term."
        Analyst: "Can you provide more specific guidance on revenue growth for the next fiscal year, given the recent market volatility?"
        CFO: "As we've stated, our focus remains on operational efficiencies and prudently managing our cost structure. While we are observing certain market fluctuations, our internal projections remain cautiously optimistic regarding our capacity to deliver sustainable returns. We are not providing granular forward-looking revenue guidance at this juncture, preferring to allow our ongoing investments in innovation to speak for themselves."
        """

        try:
            analysis_result = gemini_analyzer.analyze_transcript(
                transcript_text=test_transcript,
            )
        except Exception as exc:
            print("analyze_transcript raised an exception:", exc)
            analysis_result = None

        print("analyze_transcript returned (or failed) -> continue to result handling")

        if analysis_result is None:
            print("No analysis_result object (call failed).")
        else:
            try:
                print(
                    f"\n--- Gemini Analysis {analysis_result.model_used} executed in {analysis_result.request_time_ms:.2f} ms ---"
                    f"\nSummary: {analysis_result.summary}"
                    f"\nRationale: {analysis_result.concise_rationale}"
                    f"\nSentiment: {analysis_result.overall_sentiment}"
                    f"\nConfidence: {analysis_result.management_confidence_score}"
                    f"\nEvasiveness: {analysis_result.evasiveness_score_q_a}"
                    f"\nKey Topics: {', '.join(analysis_result.key_topics)}"
                    f"\nRed Flags: {', '.join(analysis_result.red_flags)}"
                )
                print("\n--- Full Analysis Result ---")
                print(analysis_result)
            except Exception as exc:
                print("Failed to print analysis result:", exc)
