import httpx
from typing import Any, Dict, List, Optional, Union
from app.core.config import settings
from app.prompts.enhancer import HUMANINTENT_SYSTEM_PROMPT


class OllamaError(Exception):
    """Base exception for Ollama interactions."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama service cannot be reached."""
    pass


class OllamaTimeoutError(OllamaError):
    """Raised when an Ollama request times out."""
    pass


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested model is not installed in Ollama."""
    pass


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an unexpected or malformed response."""
    pass


class OllamaClient:
    """HTTP client service for communicating directly with the local Ollama instance."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout

    async def check_health(self) -> bool:
        """Check if the local Ollama instance is reachable."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError):
            return False

    async def list_models(self) -> List[str]:
        """Fetch the list of installed local Ollama models."""
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.base_url}. Make sure Ollama is running locally."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Request to Ollama timed out while listing models ({url})."
            ) from exc
        except httpx.RequestError as exc:
            raise OllamaConnectionError(
                f"Failed to communicate with Ollama: {exc}"
            ) from exc

        if response.status_code != 200:
            raise OllamaResponseError(
                f"Ollama returned HTTP {response.status_code} when listing models."
            )

        try:
            data = response.json()
            models_data = data.get("models", [])
            return [m.get("name", "") for m in models_data if m.get("name")]
        except Exception as exc:
            raise OllamaResponseError(
                f"Failed to parse models list from Ollama response: {exc}"
            ) from exc

    async def chat(
        self,
        messages: List[Dict[str, str]],
        format: Optional[Union[str, Dict[str, Any]]] = None,
        model_override: Optional[str] = None,
    ) -> str:
        """Reusable low-level transport for Ollama chat completion API."""
        model_to_use = model_override or self.model
        url = f"{self.base_url}/api/chat"

        payload: Dict[str, Any] = {
            "model": model_to_use,
            "messages": messages,
            "stream": False,
        }
        if format:
            payload["format"] = format

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.base_url}. Make sure Ollama is running locally."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Ollama took longer than {self.timeout}s to respond. Try a smaller model or increase timeout."
            ) from exc
        except httpx.RequestError as exc:
            raise OllamaConnectionError(
                f"Failed to communicate with Ollama: {exc}"
            ) from exc

        if response.status_code == 404:
            try:
                error_body = response.json().get("error", "")
            except Exception:
                error_body = response.text
            raise OllamaModelNotFoundError(
                f"Configured model '{model_to_use}' is not installed in Ollama. "
                f"Please run 'ollama run {model_to_use}' locally to download it. ({error_body})"
            )

        if response.status_code != 200:
            try:
                err_detail = response.json().get("error", response.text)
            except Exception:
                err_detail = response.text
            raise OllamaResponseError(
                f"Ollama returned status code {response.status_code}: {err_detail}"
            )

        try:
            data = response.json()
            content = data.get("message", {}).get("content", "").strip()
            if not content:
                raise OllamaResponseError("Ollama returned an empty response.")
            return content
        except OllamaResponseError:
            raise
        except Exception as exc:
            raise OllamaResponseError(
                f"Failed to parse Ollama response: {exc}"
            ) from exc

    async def enhance_prompt(self, user_prompt: str, model_override: Optional[str] = None) -> str:
        """Send prompt to Ollama chat API with basic HumanIntent system instructions."""
        messages = [
            {"role": "system", "content": HUMANINTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return await self.chat(messages=messages, model_override=model_override)


ollama_client = OllamaClient()
