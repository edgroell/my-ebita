# Agentic bot using pydantic_ai with RAG, web search, and conversation history.

from dotenv import load_dotenv
import os
from typing import List, Dict
import asyncio
import requests
from datetime import datetime

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel

from .rag_manager import RAGManager

load_dotenv(".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST")


class ConversationTurn(BaseModel):
    """Structured Q&A turn in conversation"""
    question: str
    answer: str


class AgenticBot:
    """
    Agent using pydantic_ai with conversation history.
    - RAG search scoped to a specific earnings call transcript
    - Optional Google Custom Search tool
    - Conversation history tracking
    - Langfuse observability
    """

    def __init__(
        self,
        rag_manager: RAGManager,
        transcript_id: int,
        openai_model: str = "gpt-4o-mini",
        google_api_key: str | None = None,
        google_cse_id: str | None = None,
        user_id: str | None = None,
    ):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")

        self.rag = rag_manager
        self.transcript_id = transcript_id
        self.user_id = user_id
        
        # Conversation history
        self.conversation_history: List[ConversationTurn] = []
        
        # Tool call tracking per query
        self._tool_calls_this_query: Dict[str, int] = {}
        
        # Initialize Langfuse (optional)
        self._lf = None
        if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY and LANGFUSE_HOST:
            try:
                from langfuse import Langfuse
                import langfuse
                
                self._lf = Langfuse(
                    public_key=LANGFUSE_PUBLIC_KEY,
                    secret_key=LANGFUSE_SECRET_KEY,
                    host=LANGFUSE_HOST
                )
                # Test the connection
                self._lf.auth_check()

                print(f"✓ Langfuse observability enabled (version: {langfuse.__version__ if hasattr(langfuse, '__version__') else 'unknown'})") #type: ignore
                # print(f"  Available methods: {[m for m in dir(self._lf) if not m.startswith('_')]}")
                
            except Exception as e:
                print(f"⚠ Langfuse initialization failed: {e}")
                self._lf = None

        # Optional Google Custom Search tool
        self._google_api_key = google_api_key or GOOGLE_API_KEY
        self._google_cse_id = google_cse_id or GOOGLE_CSE_ID

        # Build the pydantic_ai Agent using OpenAI model
        if self._google_api_key and self._google_cse_id:
            default_prompt = (
                f"You are a financial analysis assistant for earnings call transcript {transcript_id}.\n\n"
                "CRITICAL TOOL USAGE RULES (FOLLOW EXACTLY):\n\n"
                "1. TRANSCRIPT QUESTIONS → Use 'search_transcript' tool ONCE:\n"
                "   - Revenue, earnings, margins, guidance mentioned in the call\n"
                "   - Management comments, strategies, business updates\n"
                "   - Q&A responses from the earnings call\n\n"
                "2. CURRENT/MARKET DATA → Use 'search_web' tool ONCE:\n"
                "   - Current stock prices (ALWAYS use web search)\n"
                "   - Recent news or market conditions\n"
                "   - Competitor information\n"
                "   - Industry trends\n\n"
                "3. MANDATORY LIMITS - VIOLATING THESE IS AN ERROR:\n"
                "   - Each tool can be called EXACTLY ONCE per question\n"
                "   - After receiving tool results, you MUST formulate an answer immediately\n"
                "   - DO NOT call the same tool again with rephrased queries\n"
                "   - DO NOT call tools sequentially to 'verify' or 'improve' results\n"
                "   - If a tool returns results, those ARE the final results - use them\n"
                "   - If a tool returns 'No results found', accept that and tell the user\n\n"
                "4. Response Strategy:\n"
                "   - If transcript search finds nothing → tell user and suggest they might need web search\n"
                "   - If web search returns results → summarize them immediately\n"
                "   - If web search returns 'No results found' → tell user no information available\n"
                "   - DO NOT try alternative queries or call tools multiple times\n\n"
                "5. Keep answers concise and based strictly on the tool results received.\n"
            )
        else:
            default_prompt = (
                f"You are a financial analysis assistant for earnings call transcript {transcript_id}.\n\n"
                "IMPORTANT LIMITATIONS:\n"
                "- You can ONLY search the earnings call transcript using 'search_transcript' tool\n"
                "- Web search is NOT available\n\n"
                "TOOL USAGE RULES:\n"
                "1. Use 'search_transcript' ONCE to find information in the earnings call\n"
                "2. If the information is not in the transcript (e.g., current stock prices, recent news), respond:\n"
                "   'I can only search the earnings call transcript. Current market data like stock prices is not available. Please ask about information discussed in the earnings call.'\n"
                "3. DO NOT repeatedly call the tool with different queries\n"
                "4. Keep answers concise and based on transcript content only.\n\n"
                "GOOD QUESTIONS:\n"
                "- What was the revenue this quarter?\n"
                "- What guidance did management provide?\n"
                "- What were the key highlights?\n\n"
                "BAD QUESTIONS (cannot answer):\n"
                "- What is the current stock price?\n"
                "- What's the latest news?\n"
                "- How do they compare to competitors?\n"
            )

        self.agent = Agent(
            model=f"openai:{openai_model}",
            system_prompt=default_prompt,
            retries=1,
        )
        self._model_name = f"openai:{openai_model}"

        # Register tools
        @self.agent.tool
        def search_transcript(ctx: RunContext[None], query: str) -> str:
            """
            Search the earnings call transcript for information mentioned during the call.
            
            USE THIS FOR: Revenue, earnings, guidance, management comments, business updates.
            DO NOT USE FOR: Current stock prices, recent news, market data.
            
            Args:
                query: What to search for in the transcript
                
            Returns:
                Relevant excerpts from the earnings call transcript
            """
            return self._rag_search_impl(query)
        
        if self._google_api_key and self._google_cse_id:
            @self.agent.tool
            def search_web(ctx: RunContext[None], query: str) -> str:
                """
                Search the web for current market data and recent information.
                
                USE THIS FOR: Current stock prices, recent news, competitor info, market trends.
                DO NOT USE FOR: Information from the earnings call transcript.
                
                Args:
                    query: What to search for on the web
                    
                Returns:
                    Web search results with current information
                """
                return self._google_search_impl(query)
            
            print("✓ Google Custom Search enabled")
        else:
            print("⚠ Google Custom Search NOT enabled (missing API keys)")
        
        print(f"✓ Agent initialized with model: {openai_model}")

    def _log_to_langfuse(self, event_name: str, metadata: dict):
        """Log a small span as an event under the current trace context."""
        if not self._lf:
            return
        try:
            with self._lf.start_as_current_span(name=event_name):
                try:
                    self._lf.update_current_span(metadata=metadata)
                except Exception:
                    pass
        except Exception as e:
            print(f"[Langfuse] start_as_current_span failed: {e}")

    def _build_conversation_context(self) -> str:
        """Build conversation history context for the prompt"""
        if not self.conversation_history:
            return ""
        
        context_parts = ["Previous conversation:"]
        for i, turn in enumerate(self.conversation_history[-2:], 1):
            context_parts.append(f"\nQ{i}: {turn.question}")
            answer = turn.answer[:150] + "..." if len(turn.answer) > 150 else turn.answer
            context_parts.append(f"A{i}: {answer}")
        
        return "\n".join(context_parts) + "\n\nCurrent question:"

    # --- Tool Implementation Methods ---
    
    def _rag_search_impl(self, query: str) -> str:
        """RAG search implementation"""
        # Check if already called
        if self._tool_calls_this_query.get("search_transcript", 0) >= 1:
            print("⚠️ [RAG Search] Already called once - blocking repeated call")
            return "ERROR: search_transcript was already called. You must use the previous results."
        
        self._tool_calls_this_query["search_transcript"] = self._tool_calls_this_query.get("search_transcript", 0) + 1
        
        try:
            print(f"🔍 [RAG Search] Query: '{query}'")
            
            self._log_to_langfuse("rag_search", {
                "query": query,
                "transcript_id": self.transcript_id
            })
            
            docs = self.rag.retrieve_context(query, transcript_id=self.transcript_id)
            ctx_text = self.rag.build_context_text(docs)
            
            MAX_CONTEXT_CHARS = 6000
            if len(ctx_text) > MAX_CONTEXT_CHARS:
                ctx_text = ctx_text[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated...]"
                print(f"⚠️ [RAG Search] Context truncated to {MAX_CONTEXT_CHARS} chars")
            
            print(f"✅ [RAG Search] Retrieved {len(docs)} docs, {len(ctx_text)} chars")
            
            self._log_to_langfuse("rag_search_complete", {
                "query": query,
                "docs_count": len(docs),
                "text_length": len(ctx_text)
            })
            
            if not ctx_text.strip() or len(docs) == 0:
                if self._google_api_key and self._google_cse_id:
                    return "No information found in the earnings call transcript for this query. Consider using web search for current data."
                else:
                    return "No information found in the earnings call transcript for this query. Web search is not available, so I can only answer questions about the earnings call content."
            
            return ctx_text
            
        except Exception as e:
            print(f"❌ [RAG Search] Error: {e}")
            self._log_to_langfuse("rag_search_error", {"query": query, "error": str(e)})
            return f"Error searching transcript: {e}"

    def _google_search_impl(self, query: str, max_results: int = 5) -> str:
        """Google search implementation"""
        # Check if already called
        if self._tool_calls_this_query.get("search_web", 0) >= 1:
            print("⚠️ [Google Search] Already called once - blocking repeated call")
            return "ERROR: search_web was already called. You must use the previous results and provide an answer now."
        
        self._tool_calls_this_query["search_web"] = self._tool_calls_this_query.get("search_web", 0) + 1
        
        try:
            print(f"🌐 [Google Search] Query: '{query}'")
            
            self._log_to_langfuse("google_search", {
                "query": query,
                "max_results": max_results
            })
            
            params = {
                "key": self._google_api_key,
                "cx": self._google_cse_id,
                "q": query,
                "num": max(1, min(int(max_results), 10)),
                "safe": "off",
            }
            resp = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            
            if not items:
                print(f"⚠️ [Google Search] No results found")
                self._log_to_langfuse("google_search_no_results", {"query": query})
                return "No web results found for this query."

            lines = []
            for i, it in enumerate(items, start=1):
                title = it.get("title", "Untitled")
                link = it.get("link", "")
                snippet = it.get("snippet", "")
                
                if len(title) > 80:
                    title = title[:80] + "..."
                
                lines.append(f"Result {i}:\nTitle: {title}\nURL: {link}\nContent: {snippet}\n")

            result = "\n".join(lines)
            
            MAX_TOTAL_LENGTH = 12000
            if len(result) > MAX_TOTAL_LENGTH:
                result = result[:MAX_TOTAL_LENGTH] + "\n[Results truncated - showing first portion]"
                print(f"⚠️ [Google Search] Results truncated to {MAX_TOTAL_LENGTH} chars")
            
            print(f"✅ [Google Search] Found {len(items)} results, {len(result)} chars")
            
            self._log_to_langfuse("google_search_complete", {
                "query": query,
                "results_count": len(items),
                "text_length": len(result)
            })
            
            return result
            
        except Exception as e:
            print(f"❌ [Google Search] Error: {e}")
            self._log_to_langfuse("google_search_error", {"query": query, "error": str(e)})
            return f"Error searching the web: {e}"

    # --- Agent Invocation Methods ---

    async def invoke_async(self, user_message: str) -> str:
        """Invoke the agent asynchronously with conversation history"""
        start_time = datetime.now()
        
        # Reset tool call counter for new query
        self._tool_calls_this_query = {}

        try:
            print(f"\n{'='*80}")
            print(f"💬 [Bot] Processing: {user_message[:100]}...")
            print(f"{'='*80}")

            if self._lf:
                try:
                    with self._lf.start_as_current_span(name="chatbot_query"):
                        try:
                            self._lf.update_current_trace(
                                name="chatbot_query",
                                user_id=self.user_id or "anonymous",
                                metadata={
                                    "transcript_id": self.transcript_id,
                                    "conversation_turn": len(self.conversation_history) + 1
                                },
                            )
                        except Exception as e:
                            print(f"[Langfuse] update_current_trace failed: {e}")

                        self._log_to_langfuse("chatbot_query_start", {
                            "transcript_id": self.transcript_id,
                            "message": user_message[:200],
                            "user_id": self.user_id,
                            "timestamp": start_time.isoformat()
                        })

                        conversation_context = self._build_conversation_context()
                        full_message = f"{conversation_context}\n{user_message}" if conversation_context else user_message

                        output_text: str
                        if self._lf:
                            try:
                                with self._lf.start_as_current_generation(
                                    name="agent_response",
                                    model=self._model_name,
                                ):
                                    try:
                                        self._lf.update_current_generation(input=full_message)
                                    except Exception:
                                        pass

                                    result = await asyncio.wait_for(
                                        self.agent.run(full_message),
                                        timeout=45.0
                                    )

                                    output_text = str(getattr(result, "output", result))
                                    if output_text.startswith("AgentRunResult("):
                                        import re
                                        m = re.search(r'output="([^"]*(?:\\.[^"]*)*)"', output_text) or \
                                            re.search(r"output='([^']*(?:\\.[^']*)*)'", output_text)
                                        if m:
                                            output_text = m.group(1)

                                    output_text = output_text.replace('\\n', '\n').replace("\\'", "'").replace('\\"', '"')

                                    try:
                                        self._lf.update_current_generation(output=output_text)
                                    except Exception:
                                        pass
                            except Exception as e:
                                print(f"[Langfuse] start_as_current_generation failed: {e}")
                                # Fallback: run without Langfuse generation
                                result = await asyncio.wait_for(self.agent.run(full_message), timeout=30.0)
                                output_text = str(getattr(result, "output", result))
                                if output_text.startswith("AgentRunResult("):
                                    import re
                                    m = re.search(r'output="([^"]*(?:\\.[^"]*)*)"', output_text) or \
                                        re.search(r"output='([^']*(?:\\.[^']*)*)'", output_text)
                                    if m:
                                        output_text = m.group(1)
                                output_text = output_text.replace('\\n', '\n').replace("\\'", "'").replace('\\"', '"')
                        else:
                            # No Langfuse: normal execution
                            result = await asyncio.wait_for(self.agent.run(full_message), timeout=30.0)
                            output_text = str(getattr(result, "output", result))
                            if output_text.startswith("AgentRunResult("):
                                import re
                                m = re.search(r'output="([^"]*(?:\\.[^"]*)*)"', output_text) or \
                                    re.search(r"output='([^']*(?:\\.[^']*)*)'", output_text)
                                if m:
                                    output_text = m.group(1)
                            output_text = output_text.replace('\\n', '\n').replace("\\'", "'").replace('\\"', '"')

                        print(f"✅ [Bot] Response generated: {len(output_text)} chars")

                        elapsed = (datetime.now() - start_time).total_seconds()
                        self._log_to_langfuse("chatbot_query_complete", {
                            "transcript_id": self.transcript_id,
                            "user_id": self.user_id,
                            "response_length": len(output_text),
                            "elapsed_seconds": elapsed,
                            "conversation_turn": len(self.conversation_history) + 1
                        })

                        turn = ConversationTurn(question=user_message, answer=output_text)
                        self.conversation_history.append(turn)

                        if self._lf:
                            try:
                                self._lf.flush()
                                print("✓ Langfuse data flushed")
                            except Exception as e:
                                print(f"⚠ Langfuse flush failed: {e}")

                        return output_text

                except Exception as e:
                    print(f"[Langfuse] root span failed: {e}")

            # Fallback path (Langfuse unavailable or failed early)
            conversation_context = self._build_conversation_context()
            full_message = f"{conversation_context}\n{user_message}" if conversation_context else user_message

            result = await asyncio.wait_for(self.agent.run(full_message), timeout=45.0)
            output_text = str(getattr(result, "output", result))
            if output_text.startswith("AgentRunResult("):
                import re
                m = re.search(r'output="([^"]*(?:\\.[^"]*)*)"', output_text) or \
                    re.search(r"output='([^']*(?:\\.[^']*)*)'", output_text)
                if m:
                    output_text = m.group(1)
            output_text = output_text.replace('\\n', '\n').replace("\\'", "'").replace('\\"', '"')

            print(f"✅ [Bot] Response generated: {len(output_text)} chars")
            turn = ConversationTurn(question=user_message, answer=output_text)
            self.conversation_history.append(turn)
            return output_text

        except asyncio.TimeoutError:
            print("❌ [Bot] Timeout - agent may be calling tools repeatedly")
            print(f"   Tool calls this query: {self._tool_calls_this_query}")
            self._log_to_langfuse("chatbot_timeout", {
                "transcript_id": self.transcript_id,
                "user_id": self.user_id,
                "tool_calls": self._tool_calls_this_query
            })
            if self._lf:
                self._lf.flush()
            return "Sorry, the request took too long. This usually happens when searching for information that isn't available. Please try rephrasing your question or ask about information from the earnings call transcript."
        except Exception as e:
            print(f"❌ [Bot] Error: {e}")
            self._log_to_langfuse("chatbot_error", {
                "transcript_id": self.transcript_id,
                "user_id": self.user_id,
                "error": str(e)
            })
            if self._lf:
                self._lf.flush()
            raise
        finally:
            if hasattr(self, "_current_trace_id"):
                delattr(self, "_current_trace_id")

    def invoke(self, user_message: str) -> str:
        """Synchronous wrapper for invoke_async"""
        return asyncio.run(self.invoke_async(user_message))

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history as list of dicts"""
        return [{"question": turn.question, "answer": turn.answer} for turn in self.conversation_history]

    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        print("[Bot] Conversation history cleared")
        
        self._log_to_langfuse("conversation_cleared", {
            "transcript_id": self.transcript_id,
            "user_id": self.user_id
        })
        
        if self._lf:
            self._lf.flush()

    async def stream_async(self, user_message: str):
        """Stream the agent response asynchronously"""
        accumulated = []
        
        try:
            print(f"[Bot Stream] Processing query: {user_message[:100]}...")
            
            conversation_context = self._build_conversation_context()
            full_message = f"{conversation_context}\n{user_message}" if conversation_context else user_message
            
            async with self.agent.run_stream(full_message) as result:
                async for message in result.stream():
                    text = str(message)
                    accumulated.append(text)
                    yield message
                    
            final = "".join(accumulated)
            
            turn = ConversationTurn(question=user_message, answer=final)
            self.conversation_history.append(turn)
            
            print(f"[Bot Stream] Total response: {len(final)} chars")
                
        except Exception as e:
            print(f"[Bot Stream] Error: {e}")
            raise

    def stream(self, user_message: str):
        """Synchronous wrapper for stream_async"""
        async def _stream():
            messages = []
            async for msg in self.stream_async(user_message):
                messages.append(str(msg))
            return "".join(messages)
        return asyncio.run(_stream())


# TESTING
if __name__ == "__main__":
    from rag_manager import RAGManager
    from vector_store import ChromaVectorStore

    print("Initializing vector store and RAG manager...")
    vector_store = ChromaVectorStore()
    rag_manager = RAGManager(vector_store=vector_store)

    print("\nCreating agentic bot...")
    bot = AgenticBot(rag_manager=rag_manager, transcript_id=123, user_id="dev-test")
    
    # Test conversation
    queries = [
        "What were the key highlights from this earnings call?",
        "Can you elaborate on the revenue figures?",
        "What about the guidance for next quarter?"
    ]
    
    for query in queries:
        print(f"\n{'='*80}")
        print(f"Q: {query}")
        print('='*80)
        response = bot.invoke(query)
        print(f"A: {response}")
    
    print("\n" + "="*80)
    print("Conversation History:")
    print("="*80)
    for i, turn in enumerate(bot.get_conversation_history(), 1):
        print(f"\nTurn {i}:")
        print(f"Q: {turn['question']}")
        print(f"A: {turn['answer'][:100]}...")
