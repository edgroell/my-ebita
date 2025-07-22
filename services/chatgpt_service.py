import os
import json

from openai import OpenAI
from openai import OpenAIError
from dotenv import load_dotenv

load_dotenv()


class ChatGPTService:
    """
    A service class to interact with OpenAI's ChatGPT models (e.g., gpt-4o-mini, gpt-4.1-mini).
    Designed to provide analysis on earnings call transcripts.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initializes the ChatGPTService with an API key and specified model.

        Args:
            api_key (str): Your OpenAI API key.
            model_name (str): The name of the ChatGPT model to use (e.g., "gpt-4o-mini", "gpt-4.1-mini").
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found!")
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        print(f"ChatGPTService initialized with model: {self.model_name}")

    def analyze_transcript(self, transcript_text: str, user_prompt: str, max_tokens: int = 1000) -> dict:
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
                  Example: {"analysis": "...", "model_used": "gpt-4o-mini", "success": True}
        """
        if not transcript_text:
            return {"error": "Transcript text cannot be empty for analysis.", "success": False}
        if not user_prompt:
            return {"error": "User prompt cannot be empty for analysis.", "success": False}

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

            try:
                parsed_content = json.loads(analysis_content)
                return {"analysis": parsed_content, "model_used": self.model_name, "success": True}
            except json.JSONDecodeError:
                return {"analysis": analysis_content, "model_used": self.model_name, "success": True,
                        "warning": "Expected JSON but got plain text."}

            # if type is text
            # return {
            #     "analysis": analysis_content,
            #     "model_used": self.model_name,
            #     "success": True
            # }

        except OpenAIError as e:
            print(f"OpenAI API Error: {e}")
            return {"error": f"Failed to get analysis from ChatGPT: {e}", "success": False}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"error": f"An unexpected error occurred: {e}", "success": False}


# TESTING

print("Testing ChatGPTService with gpt-4.1-mini...")
chatgpt_analyzer = ChatGPTService(model_name="gpt-4.1-mini")

test_transcript = """
CEO: "We've had a truly transformative quarter, navigating significant macroeconomic headwinds with unparalleled agility. Our strategic repositioning initiatives are yielding promising preliminary indicators, suggesting robust potential for enhanced shareholder value in the mid-to-long term."
Analyst: "Can you provide more specific guidance on revenue growth for the next fiscal year, given the recent market volatility?"
CFO: "As we've stated, our focus remains on operational efficiencies and prudently managing our cost structure. While we are observing certain market fluctuations, our internal projections remain cautiously optimistic regarding our capacity to deliver sustainable returns. We are not providing granular forward-looking revenue guidance at this juncture, preferring to allow our ongoing investments in innovation to speak for themselves."
"""
test_prompt = "Identify the overall sentiment of the earnings call, specifically noting any corporate jargon or evasive language used by management regarding future guidance."

analysis_result = chatgpt_analyzer.analyze_transcript(test_transcript, test_prompt)

if analysis_result["success"]:
    print("\n--- ChatGPT Analysis (gpt-4.1-mini) ---")
    for k, v in analysis_result["analysis"].items():
        print(f"{k}: {v}")

else:
    print("\n--- ChatGPT Analysis Error ---")
    print(analysis_result["error"])