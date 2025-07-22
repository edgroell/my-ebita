import json

from openai import OpenAI
from openai import OpenAIError


class ChatGPTService:
    """
    A service class to interact with OpenAI's ChatGPT models (e.g., gpt-4o-mini, gpt-4.1-mini).
    Designed to provide analysis on earnings call transcripts.
    """

    def __init__(self, api_key: str,
                 model_name: str = "gpt-4o-mini"):
        """
        Initializes the ChatGPTService with an API key and specified model.

        Args:
            api_key (str): The API key to authenticate with the OpenAI API.
            model_name (str): The name of the ChatGPT model to use (e.g., "gpt-4o-mini", "gpt-4.1-mini").
        """
        if not api_key:
            raise ValueError("OpenAI API key cannot be empty.")
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        print(f"ChatGPTService initialized with model: {self.model_name}")

    def analyze_transcript(self, transcript_text: str, user_prompt: str, max_tokens: int = 2500) -> dict:
        """
        Sends an earnings call transcript and a user-defined prompt to the ChatGPT model
        for analysis.

        Args:
            transcript_text (str): The full text of the earnings call transcript.
            user_prompt (str): The specific instruction for the AI (e.g., "Summarize key points,
                                identify overall sentiment, and highlight any evasive language
                                from management regarding future guidance.").
            max_tokens (int): The maximum number of tokens for the model's response.

        Returns:
            dict: A dictionary containing the AI's analysis, or an error message.
                  Example: {"analysis": {...}, "model_used": "gpt-4o-mini", "success": True}
        """
        if not transcript_text:
            return {"error": "Transcript text cannot be empty for analysis.", "success": False}

        messages = [
            {"role": "system",
             "content": "You are a financial analyst AI specializing in dissecting earnings call transcripts. Your goal is to provide concise, factual, and insightful analysis, identifying sentiment, key topics, and any signs of management spin or evasiveness. Focus on the financial implications. **Always return your analysis as a JSON object.**"},
            {"role": "user",
             "content": f"Here is an earnings call transcript:\n\n---\n{transcript_text}\n---\n\nBased on the transcript, {user_prompt}. **Ensure the output is valid JSON.**"}
        ]

        try:
            print(f"Sending request to OpenAI model: {self.model_name}...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            analysis_content = response.choices[0].message.content

            parsed_content = json.loads(analysis_content)
            return {"analysis": parsed_content, "model_used": self.model_name, "success": True}

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error (critical): {e}. Raw response: {analysis_content}")
            return {"error": f"Failed to parse AI response as JSON (critical): {e}. Raw: {analysis_content[:200]}...",
                    "success": False}
        except OpenAIError as e:
            print(f"OpenAI API Error: {e}")
            return {"error": f"Failed to get analysis from ChatGPT: {e}", "success": False}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"error": f"An unexpected error occurred: {e}", "success": False}


# TESTING (Only runs when chatgpt_service.py is executed directly)
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
        Analyze the transcript. Provide the response as a JSON object with the following keys:
        - "summary": A concise summary of the call (string).
        - "overall_sentiment": "Positive", "Neutral", or "Negative" (string).
        - "management_confidence_score": A score from 0 to 100 for management's confidence (integer).
        - "evasiveness_score_q_a": A score from 0 to 100 for evasiveness in Q&A (integer).
        - "key_topics": A list of 3-5 main topics discussed (array of strings).
        - "red_flags": A list of any specific red flags or evasive phrases identified (array of strings).
        """

        analysis_result = chatgpt_analyzer.analyze_transcript(test_transcript, test_prompt)

        if analysis_result["success"]:
            print(f"\n--- ChatGPT Analysis ({analysis_result['model_used']}) ---")
            print(json.dumps(analysis_result["analysis"], indent=2))
        else:
            print("\n--- ChatGPT Analysis Error ---")
            print(analysis_result["error"])
