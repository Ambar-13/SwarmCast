"""OpenAI-compatible LLM backend for Concordia.

Wraps any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, Together, etc.)
into the Concordia LanguageModel interface.

Includes defensive sanitization to prevent malformed API requests from
crashing ensemble runs. Every known failure mode is handled with retries
and fallbacks.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Collection, Mapping, Sequence

import openai
from concordia.language_model import language_model


# Characters that are safe in JSON string values.
# Strip everything else to prevent serialization failures.
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize_prompt(text: str, max_length: int = 100_000) -> str:
    """Clean a prompt string so it won't break JSON serialization.

    Handles: null bytes, control characters, invalid UTF-8,
    excessive length, and embedded surrogate pairs.
    """
    # 1. Force valid UTF-8
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

    # 2. Strip control characters (keep newline \n, carriage return \r, tab \t)
    text = _CONTROL_CHAR_RE.sub("", text)

    # 3. Remove surrogate pairs that slip through
    text = text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")

    # 4. Truncate if too long (keep start and end for context)
    if len(text) > max_length:
        half = max_length // 2
        text = text[:half] + "\n...[context truncated]...\n" + text[-half:]

    return text.strip()


def _safe_float(value: float, default: float = 0.7) -> float:
    """Convert to a JSON-safe float. Handles NaN, Infinity, numpy types."""
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


class OpenAIModel(language_model.LanguageModel):
    """Wraps any OpenAI-compatible API endpoint (OpenAI, Ollama, vLLM, Together) into Concordia's LanguageModel interface. Handles sanitization, retries, and graceful fallback so a single bad request never crashes an ensemble run."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        base_url: str | None = None,
        default_temperature: float = 0.7,
        max_retries: int = 3,
    ):
        """Configure the OpenAI client, target model, default sampling temperature, and retry limit."""
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model = model_name
        self._default_temperature = default_temperature
        self._max_retries = max_retries

    def set_temperature(self, temperature: float) -> None:
        """Set the default temperature for subsequent calls.

        Used by the ensemble runner to vary temperature across runs.
        Concordia calls sample_text() without a temperature argument,
        so this is the only way to control it per-run.
        """
        self._default_temperature = _safe_float(temperature, 0.7)

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 5000,
        terminators: Collection[str] = (),
        temperature: float = -1.0,  # sentinel: -1 means "use default"
        top_p: float = 0.95,
        top_k: int = 64,  # accepted but not sent to OpenAI (unsupported)
        timeout: float = 60,
        seed: int | None = None,
    ) -> str:
        """Send a prompt to the model and return the response text. Retries up to max_retries times with exponential backoff, applying progressive ASCII sanitization on BadRequestError, and returns an empty string if all attempts fail."""
        # Use the instance default unless the caller passes a value
        if temperature < 0:
            temperature = self._default_temperature

        # Sanitize everything
        prompt = _sanitize_prompt(prompt)
        temperature = _safe_float(temperature, self._default_temperature)
        top_p = _safe_float(top_p, 0.95)
        max_tokens = int(max_tokens)

        stop = list(terminators) if terminators else None

        # Retry with exponential backoff and progressive sanitization
        last_error = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    seed=seed,
                )
                return resp.choices[0].message.content or ""

            except openai.BadRequestError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    # Progressive sanitization: each retry strips more
                    prompt = prompt.encode("ascii", errors="replace").decode("ascii")
                    if len(prompt) > 50_000:
                        prompt = prompt[:25_000] + "\n...[truncated]...\n" + prompt[-25_000:]
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break

            except openai.RateLimitError:
                # Back off and retry
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
                continue

            except openai.APITimeoutError:
                if attempt < self._max_retries - 1:
                    time.sleep(1)
                    continue
                break

            except openai.APIConnectionError:
                if attempt < self._max_retries - 1:
                    time.sleep(2)
                    continue
                break

            except Exception as e:
                # Catch-all for any unexpected error (including ones from
                # the openai library we didn't anticipate). Log and retry
                # rather than crashing the entire simulation run.
                last_error = e
                if attempt < self._max_retries - 1:
                    prompt = prompt.encode("ascii", errors="replace").decode("ascii")
                    time.sleep(1)
                    continue
                break

        # All retries exhausted — return a safe fallback instead of crashing
        # the entire ensemble run. The action parser will classify this as
        # do_nothing, which is better than losing the whole run.
        return ""

    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, Mapping[str, any]]:
        """Ask the model to pick one of the provided response strings and return its index, text, and an empty metadata dict."""
        augmented = prompt + "\nChoices:\n"
        for i, r in enumerate(responses):
            augmented += f"  {i}: {r}\n"
        augmented += "Reply with ONLY the number of your choice."

        text = self.sample_text(augmented, max_tokens=8, temperature=0.0, seed=seed)
        for i, r in enumerate(responses):
            if str(i) in text or r.lower() in text.lower():
                return i, r, {}
        return 0, responses[0], {}
