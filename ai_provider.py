"""
AI Provider Module
------------------
Isolates AI API integration from the application logic.
Supports multiple configurable AI providers (Gemini, Groq, OpenRouter) through a common interface.

Configuration is loaded from environment variables:
- AI_PROVIDER: 'gemini', 'groq', or 'openrouter'
- AI_API_KEY: Provider API key
- AI_MODEL: (Optional) Model name to use
"""

import os
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import requests


# ---------------------------------------------------------------------------
# Strict Legal Grounding System Instruction
# ---------------------------------------------------------------------------
LEGAL_SYSTEM_INSTRUCTION = (
    "You are an academic IPC (Indian Penal Code) Legal Assistant designed for college students "
    "studying Cyber Laws and Information Crime. Your duty is to explain legal provisions accurately, "
    "clearly, and objectively.\n\n"
    "STRICT LEGAL ACCURACY RULES:\n"
    "1. PRIMARY SOURCE: Rely strictly on the provided IPC sections below. Do NOT invent, assume, or fabricate "
    "any section numbers, offenses, or punishments.\n"
    "2. FACT VERIFICATION: Distinguish explicitly between what is directly codified in the retrieved text "
    "and any unverified general legal principles. If a specific detail is not stated in the text, explicitly state so.\n"
    "3. HISTORICAL CONTEXT & BNS TRANSITION: When answering, note that the Indian Penal Code (IPC), 1860 was replaced "
    "by the Bharatiya Nyaya Sanhita (BNS), 2023 on 1 July 2024. Clarify that IPC applies to offences committed on or before "
    "30 June 2024 (subject to transitional provisions), while BNS governs offences committed on or after 1 July 2024.\n"
    "4. CITATION: Explicitly cite the IPC Section number(s) and title(s) in your explanation.\n"
    "5. DISCLAIMER: Always maintain an educational tone and remind the user that this does not constitute formal legal counsel."
)


def format_sections_for_context(sections: List[Dict[str, Any]]) -> str:
    """Formats retrieved section dictionaries into a clean text block for context."""
    if not sections:
        return "No relevant sections retrieved from dataset."
    blocks = []
    for s in sections:
        b = (
            f"--- IPC SECTION {s.get('Section')} ---\n"
            f"Title: {s.get('section_title')}\n"
            f"Chapter {s.get('chapter')}: {s.get('chapter_title')}\n"
            f"Description: {s.get('section_desc')}\n"
        )
        blocks.append(b)
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Base Provider Interface
# ---------------------------------------------------------------------------
class BaseAIProvider(ABC):
    """Abstract Base Class for AI model providers."""

    def __init__(self, api_key: str, model: Optional[str] = None):
        self.api_key = (api_key or "").strip()
        self.model = (model or "").strip() or self.default_model

    @property
    @abstractmethod
    def name(self) -> str:
        """User-friendly name of the provider."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model used if none is specified."""
        pass

    def is_configured(self) -> bool:
        """Returns True if the provider has an API key configured."""
        return bool(self.api_key)

    def build_user_prompt(self, user_query: str, retrieved_sections: List[Dict[str, Any]]) -> str:
        """Builds user prompt content containing the retrieved IPC context."""
        context_text = format_sections_for_context(retrieved_sections)
        return (
            f"=== RETRIEVED IPC SECTIONS FROM DATASET ===\n"
            f"{context_text}\n"
            f"===========================================\n\n"
            f"USER QUESTION: {user_query}\n\n"
            f"Please provide a structured, simple explanation:"
        )

    @abstractmethod
    def generate_response(
        self,
        user_query: str,
        retrieved_sections: List[Dict[str, Any]],
        sleep_fn: Any = time.sleep
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generates an answer based on the query and retrieved IPC sections.
        Returns:
            (response_text, error_message)
        """
        pass


# ---------------------------------------------------------------------------
# Gemini Provider Adapter
# ---------------------------------------------------------------------------
class GeminiProvider(BaseAIProvider):
    """Google Gemini AI Provider."""

    @property
    def name(self) -> str:
        return "Google Gemini"

    @property
    def default_model(self) -> str:
        return "gemini-2.5-flash"

    def _is_http_status_code(self, error: Exception, target_code: int) -> bool:
        """Checks for specific HTTP status codes on exceptions."""
        for attr in ("code", "status_code", "http_status"):
            val = getattr(error, attr, None)
            if val == target_code or val == str(target_code):
                return True
        response = getattr(error, "response", None)
        if response is not None and getattr(response, "status_code", None) == target_code:
            return True
        err_str = str(error).lower()
        code_str = str(target_code)
        if target_code == 503:
            return code_str in err_str or "unavailable" in err_str or "overloaded" in err_str
        elif target_code == 429:
            return code_str in err_str or "resource_exhausted" in err_str or "quota" in err_str or "rate limit" in err_str
        elif target_code == 401 or target_code == 403:
            return code_str in err_str or "api_key_invalid" in err_str or "unauthenticated" in err_str
        return False

    def _call_gemini_single_attempt(self, prompt: str) -> str:
        """Calls Gemini using modern google.genai with fallback to google.generativeai."""
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            if response and response.text:
                return response.text
            raise ValueError("Empty response received from Gemini API.")
        except Exception as primary_err:
            if self._is_http_status_code(primary_err, 503) or self._is_http_status_code(primary_err, 429):
                raise primary_err
            # Try fallback to legacy SDK
            try:
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=self.api_key)
                legacy_model = genai_legacy.GenerativeModel("gemini-1.5-flash")
                response = legacy_model.generate_content(prompt)
                if response and response.text:
                    return response.text
                raise ValueError("Empty response from Gemini fallback model.")
            except Exception:
                raise primary_err

    def generate_response(
        self,
        user_query: str,
        retrieved_sections: List[Dict[str, Any]],
        sleep_fn: Any = time.sleep
    ) -> Tuple[Optional[str], Optional[str]]:
        if not self.is_configured():
            return None, "Gemini API key is not configured. Add AI_API_KEY (or GEMINI_API_KEY) in .env."

        user_content = self.build_user_prompt(user_query, retrieved_sections)
        combined_prompt = f"{LEGAL_SYSTEM_INSTRUCTION}\n\n{user_content}"

        retry_delays = [2, 4, 8]
        max_retries = len(retry_delays)

        for attempt in range(max_retries + 1):
            try:
                response_text = self._call_gemini_single_attempt(combined_prompt)
                return response_text, None
            except Exception as err:
                if self._is_http_status_code(err, 503):
                    if attempt < max_retries:
                        sleep_fn(retry_delays[attempt])
                        continue
                    return None, (
                        "The Gemini AI service is temporarily overloaded or unavailable (HTTP 503). "
                        "All 3 automatic retry attempts have failed. Please try again shortly."
                    )
                if self._is_http_status_code(err, 429):
                    return None, (
                        "Gemini API rate limit exceeded (HTTP 429). The request quota has been reached. "
                        "Please wait a moment before sending another query."
                    )
                if self._is_http_status_code(err, 401) or "api_key" in str(err).lower():
                    return None, (
                        "Invalid Gemini API key. Please check your key configuration in .env."
                    )
                return None, (
                    "Unable to generate AI response due to a connection or service issue with Gemini. "
                    "Please verify your internet connection and API key."
                )

        return None, "Gemini service is currently unavailable. Please try again later."


# ---------------------------------------------------------------------------
# Groq Provider Adapter
# ---------------------------------------------------------------------------
class GroqProvider(BaseAIProvider):
    """Groq Cloud AI Provider."""

    @property
    def name(self) -> str:
        return "Groq"

    @property
    def default_model(self) -> str:
        return "qwen/qwen3.8-27b"

    def generate_response(
        self,
        user_query: str,
        retrieved_sections: List[Dict[str, Any]],
        sleep_fn: Any = time.sleep
    ) -> Tuple[Optional[str], Optional[str]]:
        if not self.is_configured():
            return None, "Groq API key is not configured. Add AI_API_KEY (or GROQ_API_KEY) in .env."

        try:
            from groq import (
                Groq,
                RateLimitError,
                AuthenticationError,
                APITimeoutError,
                InternalServerError,
                NotFoundError,
                BadRequestError
            )
        except ImportError:
            return None, "The 'groq' package is not installed. Run 'pip install groq'."

        user_content = self.build_user_prompt(user_query, retrieved_sections)
        messages = [
            {"role": "system", "content": LEGAL_SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content}
        ]

        retry_delays = [2, 4, 8]
        max_retries = len(retry_delays)

        for attempt in range(max_retries + 1):
            try:
                client = Groq(api_key=self.api_key)
                
                # Attempt with the configured model
                model_to_use = self.model
                try:
                    completion = client.chat.completions.create(
                        model=model_to_use,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=1024,
                    )
                except NotFoundError:
                    # If the requested model is not found on Groq, attempt fallback to available active models
                    fallback_models = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
                    successful_fallback = None
                    for alt_model in fallback_models:
                        if alt_model == model_to_use:
                            continue
                        try:
                            completion = client.chat.completions.create(
                                model=alt_model,
                                messages=messages,
                                temperature=0.2,
                                max_tokens=1024,
                            )
                            successful_fallback = alt_model
                            break
                        except Exception:
                            continue
                    if not successful_fallback:
                        return None, (
                            f"Groq Model Error (HTTP 404): The requested model '{model_to_use}' was not found. "
                            "Please set AI_MODEL='qwen/qwen3.8-27b' or 'openai/gpt-oss-120b' in .env."
                        )

                if completion and completion.choices and completion.choices[0].message:
                    text = completion.choices[0].message.content
                    if text:
                        return text, None
                return None, "Empty response received from Groq API."

            except AuthenticationError:
                return None, "Invalid Groq API key (HTTP 401). Please check your key in .env."
            except RateLimitError:
                return None, (
                    "Groq API rate limit exceeded (HTTP 429). The request quota has been reached. "
                    "Please wait a few seconds before asking another question."
                )
            except APITimeoutError:
                return None, "Request to Groq API timed out. Please verify your internet connection."
            except BadRequestError as bad_req:
                return None, f"Groq Bad Request Error (HTTP 400): {str(bad_req)}"
            except InternalServerError as server_err:
                if attempt < max_retries:
                    sleep_fn(retry_delays[attempt])
                    continue
                return None, (
                    "Groq AI service is currently unavailable or experiencing high load (HTTP 500/503). "
                    "All retry attempts failed. Please try again shortly."
                )
            except Exception as err:
                err_str = str(err).lower()
                if "503" in err_str or "unavailable" in err_str:
                    if attempt < max_retries:
                        sleep_fn(retry_delays[attempt])
                        continue
                    return None, "Groq service is temporarily unavailable (HTTP 503). Please retry in a moment."
                if "429" in err_str:
                    return None, "Groq rate limit exceeded (HTTP 429). Please wait a moment."
                return None, f"Groq Service Error: {str(err)}"

        return None, "Groq service unavailable after multiple retries."


# ---------------------------------------------------------------------------
# OpenRouter Provider Adapter
# ---------------------------------------------------------------------------
class OpenRouterProvider(BaseAIProvider):
    """OpenRouter Unified API Provider."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    @property
    def name(self) -> str:
        return "OpenRouter"

    @property
    def default_model(self) -> str:
        return "meta-llama/llama-3.3-70b-instruct:free"

    def generate_response(
        self,
        user_query: str,
        retrieved_sections: List[Dict[str, Any]],
        sleep_fn: Any = time.sleep
    ) -> Tuple[Optional[str], Optional[str]]:
        if not self.is_configured():
            return None, "OpenRouter API key is not configured. Add AI_API_KEY (or OPENROUTER_API_KEY) in .env."

        user_content = self.build_user_prompt(user_query, retrieved_sections)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": LEGAL_SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.2,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "IPC Legal Assistant",
            "Content-Type": "application/json"
        }

        retry_delays = [2, 4, 8]
        max_retries = len(retry_delays)

        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    self.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=35
                )

                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices and "message" in choices[0]:
                        content = choices[0]["message"].get("content", "")
                        if content:
                            return content, None
                    return None, "Empty response received from OpenRouter API."

                elif resp.status_code == 401:
                    return None, "Invalid OpenRouter API key (HTTP 401). Please verify your key in .env."

                elif resp.status_code == 429:
                    return None, (
                        "OpenRouter API rate limit exceeded (HTTP 429). Free-tier or model limit reached. "
                        "Please wait a few moments before trying again."
                    )

                elif resp.status_code in (502, 503, 504):
                    if attempt < max_retries:
                        sleep_fn(retry_delays[attempt])
                        continue
                    return None, (
                        f"OpenRouter upstream provider is temporarily unavailable (HTTP {resp.status_code}). "
                        "All retry attempts failed. Please try again shortly."
                    )

                else:
                    return None, f"OpenRouter API returned HTTP {resp.status_code}. Please check your model or account."

            except requests.exceptions.Timeout:
                return None, "Request to OpenRouter timed out. The upstream model took too long to respond."
            except requests.exceptions.ConnectionError:
                return None, "Could not establish a connection to OpenRouter. Please check your internet connectivity."
            except Exception as err:
                return None, f"OpenRouter communication error: {str(err)}"

        return None, "OpenRouter service unavailable after multiple retries."


# ---------------------------------------------------------------------------
# Unsupported / Invalid Provider Placeholder
# ---------------------------------------------------------------------------
class InvalidConfigProvider(BaseAIProvider):
    """Fallback provider when an invalid AI_PROVIDER value is provided in .env."""

    def __init__(self, requested_name: str):
        super().__init__(api_key="", model="")
        self._requested_name = requested_name

    @property
    def name(self) -> str:
        return f"Unknown Provider ('{self._requested_name}')"

    @property
    def default_model(self) -> str:
        return "none"

    def is_configured(self) -> bool:
        return False

    def generate_response(
        self,
        user_query: str,
        retrieved_sections: List[Dict[str, Any]],
        sleep_fn: Any = time.sleep
    ) -> Tuple[Optional[str], Optional[str]]:
        return None, (
            f"Unsupported AI_PROVIDER '{self._requested_name}'. "
            "Supported providers are: 'gemini', 'groq', or 'openrouter'. "
            "Please update AI_PROVIDER in your .env file."
        )


# ---------------------------------------------------------------------------
# Factory & Environment Resolver
# ---------------------------------------------------------------------------
def get_provider_config() -> Dict[str, str]:
    """
    Reads provider configuration from environment variables.
    Supports AI_PROVIDER, AI_API_KEY, AI_MODEL with provider-specific key fallbacks.
    """
    provider_name = os.getenv("AI_PROVIDER", "").strip().lower()

    # Auto-detect provider if AI_PROVIDER is not set
    if not provider_name:
        if os.getenv("GROQ_API_KEY"):
            provider_name = "groq"
        elif os.getenv("OPENROUTER_API_KEY"):
            provider_name = "openrouter"
        elif os.getenv("GEMINI_API_KEY"):
            provider_name = "gemini"
        elif os.getenv("AI_API_KEY"):
            provider_name = "gemini"
        else:
            provider_name = "gemini"

    # Resolve API Key
    api_key = os.getenv("AI_API_KEY", "").strip()
    if not api_key:
        if provider_name == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
        elif provider_name == "groq":
            api_key = os.getenv("GROQ_API_KEY", "").strip()
        elif provider_name == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    # Resolve Model
    model = os.getenv("AI_MODEL", "").strip()

    return {
        "provider": provider_name,
        "api_key": api_key,
        "model": model,
    }


def get_ai_provider() -> BaseAIProvider:
    """
    Instantiates and returns the configured AI provider object based on .env settings.
    """
    config = get_provider_config()
    provider_key = config["provider"]
    api_key = config["api_key"]
    model = config["model"] or None

    if provider_key in ("gemini", "google", "google-genai"):
        return GeminiProvider(api_key=api_key, model=model)
    elif provider_key in ("groq", "groqcloud"):
        return GroqProvider(api_key=api_key, model=model)
    elif provider_key in ("openrouter", "open-router"):
        return OpenRouterProvider(api_key=api_key, model=model)
    else:
        return InvalidConfigProvider(requested_name=provider_key)
