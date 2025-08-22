# GroqService: A client for interacting with the Groq API

import json
import time
import requests
import re
from typing import Any, Dict, Optional
from pathlib import Path

# Attempt to import AnalysisResult from the same directory
try:
    from .analysis_result import AnalysisResult
except Exception:
    import importlib.util
    import sys

    _analysis_path = Path(__file__).resolve().parent / "analysis_result.py"
    spec = importlib.util.spec_from_file_location("services.analysis_result", str(_analysis_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load analysis_result.py at {_analysis_path}")
    analysis_mod = importlib.util.module_from_spec(spec)
    sys.modules["services.analysis_result"] = analysis_mod
    spec.loader.exec_module(analysis_mod)
    AnalysisResult = analysis_mod.AnalysisResult

# Ensure the name `Groq` is always defined to satisfy static analysis; assign the real class if import succeeds.
Groq: Any = None
try:
    from groq import Groq as _Groq
    Groq = _Groq
    _HAS_GROQ_SDK = True
except Exception:
    Groq = None
    _HAS_GROQ_SDK = False


class GroqService:
    """
    A service class to interact with the Groq models via HTTP (requests).
    Prompts the model to use internal chain-of-thought but returns a concise JSON object.
    Measures api_elapsed_ms (from api_start until parsing finished), parse_elapsed_ms, total_ms.
    """

    def __init__(self, api_key: str, default_model: str = "openai/gpt-oss-20b", base_url: str = "https://api.groq.ai/v1/"):
        if not api_key:
            raise ValueError("Groq API key cannot be empty.")
        self.api_key = api_key
        self.default_model = default_model
        self.base_url = base_url.rstrip("/") + "/"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        # Use a requests.Session for potential connection reuse
        self.session = requests.Session()
        print(f"GroqService initialized with model: {self.default_model}")

        if _HAS_GROQ_SDK:
            try:
                self.sdk_client = Groq(api_key=self.api_key)
            except Exception as e:
                # keep SDK off if init fails
                self.sdk_client = None
                print(f"Groq SDK init failed, falling back to HTTP client: {e}")

    def _make_request(self, model_name: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        # Use SDK if present
        if _HAS_GROQ_SDK and getattr(self, "sdk_client", None) is not None:
            try:
                # Use a local variable so static type checkers understand it is not None
                sdk = getattr(self, "sdk_client", None)
                if sdk is None:
                    # If sdk_client is unexpectedly None, raise to allow fallback to HTTP path
                    raise RuntimeError("Groq SDK client is not initialized")
                sdk_payload = {
                    "messages": payload.get("messages") or [{"role":"user","content": payload.get("prompt","")}],
                    "model": model_name,
                    "stream": False,
                }
                return sdk.chat.completions.create(**sdk_payload)
            except Exception as e:
                # Provide clearer guidance when model is not found / inaccessible
                err_text = str(e)
                if "model_not_found" in err_text or "does not exist" in err_text or getattr(e, "status_code", None) == 404:
                    raise ValueError(
                        f"Groq SDK error: model '{model_name}' not found or not accessible for this API key. "
                        "Check your Groq Console for available models and update default_model. "
                        f"Underlying error: {e}"
                    ) from e
                # If the SDK client was unexpectedly not available, allow falling back to the HTTP path below
                if isinstance(e, RuntimeError):
                    pass
                else:
                    raise ValueError(f"Groq SDK error: {e}") from e

        # fallback HTTP path (unchanged)
        url = f"{self.base_url}models/{model_name}/invoke"
        resp = self.session.post(url, headers=self.headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def generate_content(self, prompt_text: str, model_name: Optional[str] = None, temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Sends prompt to Groq and returns:
          - parsed (dict): parsed JSON object returned by model
          - api_elapsed_ms (float)
          - parse_elapsed_ms (float)
        """
        model_to_use = model_name or self.default_model
        payload: Dict[str, Any] = {
            "prompt": prompt_text,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)

        api_start = time.perf_counter()
        response_data = self._make_request(model_to_use, payload)
        parse_start = time.perf_counter()

        raw_text: Optional[str] = None
        parsed: Dict[str, Any]
        try:
            # Try common response shapes to extract text content
            raw_text = None

            # SDK object shape: ChatCompletion with .choices -> Choice -> .message -> .content
            # Support both dict-shaped HTTP responses and SDK objects by extracting choices safely.
            choices = None
            if isinstance(response_data, dict):
                choices = response_data.get("choices")
            else:
                choices = getattr(response_data, "choices", None)

            if choices:
                candidate = choices[0]
                # candidate may be an object or a dict
                if isinstance(candidate, dict):
                    # dict candidate may contain "message" or "text"
                    msg = candidate.get("message") or candidate.get("text")
                elif hasattr(candidate, "message"):
                    msg = candidate.message
                else:
                    msg = candidate

                # msg may be a plain string, an object with .content, or a dict with "content" / "text"
                if isinstance(msg, str):
                    raw_text = msg
                else:
                    raw_text = (
                        getattr(msg, "content", None)
                        or (msg.get("content") if isinstance(msg, dict) else None)
                        or (msg.get("text") if isinstance(msg, dict) else None)
                    )

            # Fallback dict HTTP shapes
            if raw_text is None and isinstance(response_data, dict):
                if "outputs" in response_data and isinstance(response_data["outputs"], list) and response_data["outputs"]:
                    candidate = response_data["outputs"][0]
                    raw_text = candidate.get("text") or candidate.get("content") or json.dumps(candidate)
                elif "choices" in response_data and isinstance(response_data["choices"], list) and response_data["choices"]:
                    candidate = response_data["choices"][0]
                    if isinstance(candidate, dict) and isinstance(candidate.get("message"), dict):
                        raw_text = candidate["message"].get("content")
                    else:
                        raw_text = candidate.get("text") or json.dumps(candidate)
                elif "result" in response_data:
                    raw_text = response_data["result"] if isinstance(response_data["result"], str) else json.dumps(response_data["result"])
                elif "text" in response_data:
                    raw_text = response_data["text"]
                else:
                    raw_text = json.dumps(response_data)

            # As a last resort, stringify the object
            if raw_text is None:
                raw_text = str(response_data)

            # Attempt to parse JSON text returned by the model
            if raw_text is None:
                raise ValueError("Groq returned no text content to parse.")

            try:
                parsed_candidate = json.loads(raw_text)
            except json.JSONDecodeError:
                # Try to recover by extracting the first JSON object in the string
                m = re.search(r'(\{.*\})', raw_text, re.DOTALL)
                if m:
                    try:
                        parsed_candidate = json.loads(m.group(1))
                    except json.JSONDecodeError:
                        preview = raw_text[:500] if raw_text else "No content available"
                        raise ValueError(f"Failed to decode JSON from Groq response (after extraction). Raw preview: {preview}")
                else:
                    preview = raw_text[:500] if raw_text else "No content available"
                    raise ValueError(f"Failed to decode JSON from Groq response. Raw preview: {preview}")

            if isinstance(parsed_candidate, list) and parsed_candidate and isinstance(parsed_candidate[0], dict):
                parsed = parsed_candidate[0]
            elif isinstance(parsed_candidate, dict):
                parsed = parsed_candidate
            else:
                raise ValueError("Groq returned JSON that is not an object/dict.")
        except json.JSONDecodeError as e:
            preview = raw_text[:500] if raw_text else "No content available"
            raise ValueError(f"Failed to decode JSON from Groq response: {e}. Raw preview: {preview}") from e
        except Exception as e:
            raise ValueError(f"Unexpected format from Groq response: {e}. Full response: {json.dumps(response_data, indent=2)}") from e
        finally:
            parse_end = time.perf_counter()

        api_elapsed_ms = (parse_end - api_start) * 1000.0
        parse_elapsed_ms = (parse_end - parse_start) * 1000.0
        return {"parsed": parsed, "api_elapsed_ms": api_elapsed_ms, "parse_elapsed_ms": parse_elapsed_ms}

    def analyze_transcript(self, transcript_text: str, user_prompt: str, model_name: Optional[str] = None, temperature: float = 0.3) -> AnalysisResult: # type: ignore
        if not transcript_text:
            return AnalysisResult(success=False, error="Transcript text cannot be empty for analysis.", model_used=model_name or self.default_model)

        model_to_use = model_name or self.default_model

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
            '"overall_sentiment" (\"Positive\", \"Neutral\", or \"Negative\"), '
            '"management_confidence_score" (integer 0-100), '
            '"evasiveness_score_q_a" (integer 0-100), '
            '"key_topics" (array of 3-5 strings), '
            '"red_flags" (array of strings). '
            "Do NOT include any extraneous text or internal reasoning. Ensure the JSON is well-formed."
        )

        full_prompt = f"{system_instructions}\n\n{user_instructions}"
        start = time.perf_counter()
        api_elapsed = None
        parse_elapsed = None

        try:
            api_start = time.perf_counter()
            gen = self.generate_content(full_prompt, model_name=model_to_use, temperature=temperature)
            parse_end = time.perf_counter()

            api_elapsed = gen.get("api_elapsed_ms")
            parse_elapsed = gen.get("parse_elapsed_ms")
            total_elapsed = (time.perf_counter() - start) * 1000.0

            parsed_analysis = gen["parsed"]
            if not isinstance(parsed_analysis, dict):
                raise ValueError("Parsed Groq output is not a dict.")

            result = AnalysisResult.from_dict(parsed_analysis, model_used=model_to_use)
            result.request_ms = api_elapsed
            result.parse_ms = parse_elapsed
            result.total_ms = total_elapsed
            return result
        except Exception as e:
            total_elapsed = (time.perf_counter() - start) * 1000.0
            return AnalysisResult(success=False, error=f"Failed to get analysis from Groq: {e}", model_used=model_to_use, request_ms=api_elapsed, parse_ms=parse_elapsed, total_ms=total_elapsed)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    # Locate project root
    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / ".env"

    if dotenv_path.exists():
        load_dotenv(dotenv_path=str(dotenv_path))
    else:
        # fallback to default behavior (current working directory)
        load_dotenv()

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not GROQ_API_KEY:
        print(f"Error: GROQ_API_KEY environment variable not set for testing. Attempted to load .env from: {dotenv_path}")
    else:
        svc = GroqService(api_key=GROQ_API_KEY)
        test_transcript = "CEO: We had a solid quarter. Analyst: Any guidance? CFO: Not at this time."
        test_prompt = "Analyze the transcript and return the structured object as used by other services."
        res = svc.analyze_transcript(test_transcript, test_prompt, temperature=0.2)
        if res.success:
            print(json.dumps(res.to_dict(), indent=2))
        else:
            print(f"Error: {res.error}")
