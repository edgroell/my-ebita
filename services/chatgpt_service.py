# ChatGPT service for analyzing earnings call transcripts

import json
import time

from pydantic import BaseModel
from openai import OpenAI
from openai import OpenAIError


class AnalysisResult(BaseModel):
    summary: str = ""
    concise_rationale: str = ""
    overall_sentiment: str = ""
    management_confidence_score: int = 0
    evasiveness_score_q_a: int = 0
    key_topics: list[str] = []
    red_flags: list[str] = []
    request_time_ms: float = 0.00
    model_used: str = ""
    temperature: float = 0.0
    max_tokens: int = 0


class ChatGPTService:
    """
    A service class to interact with OpenAI's ChatGPT models.
    Uses a prompting pattern that instructs the model to reason carefully (chain-of-thought internally)
    while returning a concise, structured JSON object (without exposing internal chain-of-thought).
    """

    def __init__(self, api_key: str, model_name: str = "gpt-5-mini"):
        if not api_key:
            raise ValueError("OpenAI API key cannot be empty.")
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        print(f"ChatGPTService initialized with model: {self.model_name}")

    def _temp_param_number(self, temperature: float) -> float:
        """
        Return the correct temperature parameter number for the current model.
        - Models in the 'gpt-5' family use exclusively 1.0
        - Other models use a float between 0 and 2.0
        """        
        if "gpt-5" in self.model_name.lower():
            return 1.0
        return temperature

    def _token_param_name(self) -> str:
        """
        Return the correct token parameter name for the current model.
        - Models in the 'gpt-5' family use 'max_completion_tokens'
        - Other models use 'max_tokens'
        """
        if "gpt-5" in self.model_name.lower():
            return "max_completion_tokens"
        return "max_tokens"

    def analyze_transcript(
        self,
        transcript_text: str,
        temperature: float = 0.3,
        max_completion_tokens: int = 1000,
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

        system_instructions = (
            "You are a financial analyst AI specializing in dissecting earnings call transcripts. "
            "When analyzing, internally reason step-by-step to improve accuracy (chain-of-thought). "
            "DO NOT reveal detailed internal chain-of-thought or step-by-step working in the output. "
            "Include a very short 1-2 sentence 'concise_rationale' explaining the key reasons for your conclusions."
        )

        user_instructions = (
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

        messages_instructions = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_instructions},
        ]

        request_elapsed = None
        request_start = time.perf_counter()

        temp_param = self._temp_param_number(temperature)
        token_param = self._token_param_name()
        params = {
            "model": self.model_name,
            "messages": messages_instructions,  # type: ignore
            "temperature": temp_param,
            token_param: max_completion_tokens,
        }

        try:
            completion = self.client.chat.completions.parse(
                **params,
                response_format=analysis_result_cls,
            )

            analysis_content = None
            try:
                analysis_content = completion.choices[0].message.parsed
            except Exception as parse_exc:
                raise ValueError(f"Failed to parse structured response: {parse_exc}") from parse_exc

            if analysis_content is None:
                raise ValueError("Received None for analysis_content.")

            request_end = time.perf_counter()
            request_elapsed = (request_end - request_start) * 1000.0

            try:
                setattr(analysis_content, "request_time_ms", request_elapsed)
                setattr(analysis_content, "model_used", self.model_name)
                setattr(analysis_content, "temperature", temp_param)
                setattr(analysis_content, "max_tokens", max_completion_tokens)
            except Exception:
                pass

            return analysis_content
        
        except OpenAIError as oe:
            raise ValueError(f"Failed to get analysis from ChatGPT: {oe}")

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse AI response as JSON: {e}")

        except Exception as e:
            raise ValueError(f"An unexpected error occurred: {e}")

# TESTING
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set for testing.")
    else:
        chatgpt_analyzer = ChatGPTService(api_key=OPENAI_API_KEY, model_name="gpt-5-mini")
        print(f"--- Testing ChatGPTService with {chatgpt_analyzer.model_name} ---")

        test_transcript = """
        CEO: "We've had a truly transformative quarter, navigating significant macroeconomic headwinds with unparalleled agility. Our strategic repositioning initiatives are yielding promising preliminary indicators, suggesting robust potential for enhanced shareholder value in the mid-to-long term."
        Analyst: "Can you provide more specific guidance on revenue growth for the next fiscal year, given the recent market volatility?"
        CFO: "As we've stated, our focus remains on operational efficiencies and prudently managing our cost structure. While we are observing certain market fluctuations, our internal projections remain cautiously optimistic regarding our capacity to deliver sustainable returns. We are not providing granular forward-looking revenue guidance at this juncture, preferring to allow our ongoing investments in innovation to speak for themselves."
        """

        try:
            analysis_result = chatgpt_analyzer.analyze_transcript(
                transcript_text=test_transcript,
                temperature=1.0,
                max_completion_tokens=2500,
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
                    f"\n--- ChatGPT Analysis {analysis_result.model_used} executed in {analysis_result.request_time_ms:.2f} ms ---"
                    f"\nSummary: {analysis_result.summary}"
                    f"\nRationale: {analysis_result.concise_rationale}"
                    f"\nSentiment: {analysis_result.overall_sentiment}"
                    f"\nConfidence: {analysis_result.management_confidence_score}"
                    f"\nEvasiveness: {analysis_result.evasiveness_score_q_a}"
                    f"\nKey Topics: {', '.join(analysis_result.key_topics)}"
                    f"\nRed Flags: {', '.join(analysis_result.red_flags)}"
                    f"\nTemperature: {analysis_result.temperature}"
                    f"\nMax Tokens: {analysis_result.max_tokens}"
                )
                print("\n--- Full Analysis Result ---")
                print(analysis_result)
            except Exception as exc:
                print("Failed to print analysis result:", exc)
