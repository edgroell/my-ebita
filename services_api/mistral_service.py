"""Mistral service for analyzing earnings call transcripts."""

import json
import time
from typing import Any, cast

from mistralai.client import Mistral
from mistralai.client.models.jsonschema import JSONSchema
from mistralai.client.models.responseformat import ResponseFormat
from pydantic import BaseModel, ConfigDict, Field


class KeyMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    management_confidence_score: int = Field(ge=0, le=100)
    evasiveness_score_q_a: int = Field(ge=0, le=100)


class StructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    concise_rationale: str
    overall_sentiment: str
    key_metrics: KeyMetrics
    key_topics: list[str]
    red_flags: list[str]


class AnalysisResult(BaseModel):
    structured_output: StructuredOutput
    request_time_ms: float = 0.0
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    text_parts = []
    for chunk in content:
        if isinstance(chunk, dict):
            chunk_type = chunk.get("type")
            text = chunk.get("text")
        else:
            chunk_type = getattr(chunk, "type", None)
            text = getattr(chunk, "text", None)
        if chunk_type == "text" and isinstance(text, str):
            text_parts.append(text)
    return "".join(text_parts).strip()


class MistralService:
    """Interact with Mistral models for transcript analysis and chat completions."""

    def __init__(self, api_key: str, model_name: str = "mistral-large-4"):
        if not api_key:
            raise ValueError("Mistral API key cannot be empty.")

        self.api_key = api_key
        self.client = Mistral(api_key=self.api_key)
        self.model_name = model_name
        print(f"✓ MistralService initialized with model: {self.model_name}")

    def call_agent(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_completion_tokens: int = 1000,
    ) -> str:
        """Send chat messages to Mistral and return the response text."""
        if not messages or not isinstance(messages, list):
            raise ValueError(
                "Parameter messages must be a non-empty list of message dictionaries."
            )

        completion = self.client.chat.complete(
            model=self.model_name,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_completion_tokens,
        )
        message = completion.choices[0].message
        if message is None:
            raise ValueError("Mistral returned an empty response message.")
        content = _extract_message_text(message.content)
        if not content:
            raise ValueError("Mistral returned an empty or non-text response.")
        return content

    def analyze_transcript(
        self,
        transcript_text: str,
        temperature: float = 0.3,
        max_completion_tokens: int = 2000,
        structured_output_cls: type[StructuredOutput] = StructuredOutput,
        analysis_result_cls: type[AnalysisResult] = AnalysisResult,
    ) -> AnalysisResult:
        """Analyze a transcript and return a structured result."""
        if not transcript_text:
            raise ValueError("Parameter transcript_text cannot be empty.")

        system_instructions = (
            "You are a financial analyst AI specializing in earnings call transcripts. "
            "Analyze the evidence carefully, but do not reveal private chain-of-thought. "
            "Give a concise 1-2 sentence rationale for your conclusions."
        )
        user_instructions = (
            "Analyze the transcript and return JSON matching the supplied schema. "
            "The summary should be 3-5 sentences. concise_rationale is required and "
            "must contain 1-2 non-empty sentences explaining the evidence behind the "
            "conclusion; do not leave it blank. Set overall_sentiment to Positive, "
            "Neutral, or Negative. Provide 3-5 key topics. Populate both "
            "key_metrics.management_confidence_score and "
            "key_metrics.evasiveness_score_q_a with integer scores from 0 to 100. "
            "Provide the red_flags array with 3-5 transcript-supported concerns, or "
            "an empty array only when no concerns are supported. Do not invent details."
        )
        messages = [
            {"role": "system", "content": system_instructions},
            {
                "role": "user",
                "content": (
                    f"Transcript:\n{transcript_text}\n\n"
                    f"Instructions:\n{user_instructions}"
                ),
            },
        ]

        request_time_ms = 0.0
        completion = None
        for token_limit in (max_completion_tokens, max_completion_tokens * 4):
            request_start = time.perf_counter()
            completion = self.client.chat.complete(
                model=self.model_name,
                messages=cast(Any, messages),
                temperature=temperature,
                max_tokens=token_limit,
                response_format=ResponseFormat(
                    type="json_schema",
                    json_schema=JSONSchema(
                        name="earnings_call_analysis",
                        schema_definition=structured_output_cls.model_json_schema(),
                        strict=True,
                    ),
                ),
                reasoning_effort="low",
            )
            request_time_ms += (time.perf_counter() - request_start) * 1000.0
            if not completion.choices or completion.choices[0].finish_reason != "length":
                break

        if completion is None or not completion.choices:
            raise ValueError("Mistral returned no choices for transcript analysis.")

        choice = completion.choices[0]
        if choice.finish_reason == "length":
            raise ValueError(
                "Mistral could not complete its analysis within the token limit "
                f"(finish_reason={choice.finish_reason}, "
                f"max_tokens={max_completion_tokens * 4})."
            )
        message = choice.message
        if message is None:
            raise ValueError(
                "Mistral returned no assistant message for transcript analysis "
                f"(finish_reason={choice.finish_reason})."
            )

        content = _extract_message_text(message.content)
        if not content:
            has_tool_calls = bool(message.tool_calls)
            raise ValueError(
                "Mistral returned no text for transcript analysis "
                f"(finish_reason={choice.finish_reason}, "
                f"content_type={type(message.content).__name__}, "
                f"tool_calls_present={has_tool_calls})."
            )
        try:
            analysis_data: Any = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError(f"Failed to parse Mistral response as JSON: {error}") from error
        if not isinstance(analysis_data, dict):
            raise ValueError("Mistral analysis response must be a JSON object.")

        structured_output = structured_output_cls(**analysis_data)
        usage = completion.usage
        total_tokens = usage.total_tokens if usage is not None else 0
        prompt_tokens = usage.prompt_tokens if usage is not None else 0
        completion_tokens = usage.completion_tokens if usage is not None else 0
        return analysis_result_cls(
            structured_output=structured_output,
            request_time_ms=request_time_ms,
            model=completion.model or self.model_name,
            temperature=temperature,
            max_tokens=max_completion_tokens,
            total_tokens=total_tokens or 0,
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
        )


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        print("Error: MISTRAL_API_KEY environment variable not set for testing.")
    else:
        mistral_service = MistralService(api_key=mistral_api_key)
        print(
            f"--- Testing MistralService with {mistral_service.model_name} ---"
        )
