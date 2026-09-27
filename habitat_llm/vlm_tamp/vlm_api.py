import base64
import copy
import hashlib
import json
import os
import requests
import signal
from datetime import datetime
from pprint import pprint

import dotenv

from habitat_llm.utils.llm_usage import TokenUsageTracker

dotenv.load_dotenv()


_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


def _openai_sampling_payload(model_name, temperature, n=1, reasoning_effort=None):
    """GPT-5 models only accept the default temperature (1); omit it otherwise."""
    payload = {}
    model = str(model_name or "")
    if not model.lower().startswith("gpt-5"):
        payload["temperature"] = temperature
    if n is not None and n != 1:
        payload["n"] = n
    if reasoning_effort and model.lower().startswith("gpt-5"):
        effort_name = str(reasoning_effort).strip().lower()
        if effort_name in _REASONING_EFFORTS:
            payload["reasoning_effort"] = effort_name
    return payload


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class VLMApi:
    name = "VLM API"
    api_key = None
    model_name = None

    def __init__(self, model_name=None, cache_path=None):
        self.model_name = model_name
        self.context = []
        self.cache_path = cache_path
        self.last_response_metadata: dict = {}
        self.token_usage = TokenUsageTracker(model_name)
        self._cache: dict = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _cache_key(self, prompt: str, image_urls) -> str:
        if image_urls is None:
            key_img = ""
        elif isinstance(image_urls, (list, tuple)):
            key_img = "\n".join(str(u) for u in image_urls)
        else:
            key_img = str(image_urls)
        payload = str(prompt) + key_img + str(self.model_name)
        return hashlib.md5(payload.encode()).hexdigest()

    def _save_cache(self):
        if self.cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)

    def _append_api_prompt_log(self, prompt, image_data_urls=None):
        log_path = os.getenv("VLM_API_PROMPT_LOG_PATH", "vlm_api_prompts.txt")
        log_dir = os.path.dirname(os.path.abspath(log_path))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        urls = list(image_data_urls) if image_data_urls else []
        block = (
            "================================================================\n"
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n"
            f"api: {self.name}\n"
            f"model: {self.model_name}\n"
            f"image_count: {len(urls)}\n"
            "----------------------------------------------------------------\n"
            f"{prompt}\n\n"
        )
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(block)

    def new_session(self):
        self.context = []

    def ask(
        self,
        prompt,
        image_data_url=None,
        image_data_urls=None,
        image_path=None,
        use_cache=True,
        **kwargs,
    ):
        self.last_response_metadata = {}
        if image_data_url is None and image_data_urls is None and image_path is not None:
            image_data_url = f"data:image/jpeg;base64,{encode_image(image_path)}"

        urls = None
        if image_data_urls:
            urls = list(image_data_urls)
        elif image_data_url is not None:
            urls = [image_data_url]

        if use_cache and self.cache_path:
            key = self._cache_key(prompt, urls)
            if key in self._cache:
                cached = self._cache[key]
                self.last_response_metadata = {
                    "provider": self.name,
                    "model": self.model_name,
                    "status": "cache_hit",
                    "empty_response": not bool(str(cached or "").strip()),
                }
                # Replay cached assistant message into context so multi-turn still works
                self.context.append({"role": "user", "content": prompt})
                self.context.append({"role": "assistant", "content": str(cached)})
                return cached

        result = self._ask(prompt, image_data_urls=urls, **kwargs)
        if not self.last_response_metadata:
            self.last_response_metadata = {
                "provider": self.name,
                "model": self.model_name,
                "status": "ok",
                "empty_response": not bool(str(result or "").strip()),
            }
        else:
            self.last_response_metadata.setdefault(
                "empty_response", not bool(str(result or "").strip())
            )

        if use_cache and self.cache_path and result is not None:
            key = self._cache_key(prompt, urls)
            self._cache[key] = result
            self._save_cache()

        return result

    def get_last_response_metadata(self):
        return copy.deepcopy(self.last_response_metadata)

    def _ask(self, prompt, image_data_urls=None, **kwargs):
        raise NotImplementedError


class GPT4vApi(VLMApi):
    name = "GPT-4V"

    def __init__(self, model_name="gpt-5.6-luna", cache_path=None):
        super().__init__(model_name=model_name, cache_path=cache_path)
        self.api_key = os.getenv("OPENAI_API_KEY", None)
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")

    def _ask(
        self,
        prompt,
        image_data_urls=None,
        max_completion_tokens=800,
        temperature=0.2,
        n=1,
        reasoning_effort=None,
        **kwargs,
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        urls = list(image_data_urls) if image_data_urls else []
        content: list = [{"type": "text", "text": prompt}]
        for i, url in enumerate(urls):
            if i > 0:
                content.append({
                    "type": "text",
                    "text": "[Scene view 2 — remaining object labels]",
                })
            content.append({
                "type": "image_url",
                "image_url": {"url": url},
            })

        messages = copy.deepcopy(self.context)
        messages.append({"role": "user", "content": content})

        if reasoning_effort is None:
            reasoning_effort = getattr(self, "reasoning_effort", None)
        if reasoning_effort:
            self.token_usage.set_reasoning_effort(reasoning_effort)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
        }
        payload.update(
            _openai_sampling_payload(
                self.model_name,
                temperature,
                n,
                reasoning_effort=reasoning_effort,
            )
        )
        self._append_api_prompt_log(prompt, urls)

        def timeout_handler(num, stack):
            raise Exception("TIMEOUT")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(180)
        http_response = None
        try:
            http_response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        except Exception as e:
            self.last_response_metadata = {
                "provider": "openai",
                "model": self.model_name,
                "status": "request_error",
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                },
                "max_completion_tokens": max_completion_tokens,
                "reasoning_effort": reasoning_effort,
                "empty_response": True,
            }
            print(f"\033[31m[VLM API ERROR] GPT4v request failed: {e}\033[0m")
            return ""
        finally:
            signal.alarm(0)

        http_status = getattr(http_response, "status_code", None)
        response_headers = getattr(http_response, "headers", {}) or {}
        request_id = response_headers.get("x-request-id")
        try:
            response = http_response.json()
        except Exception as e:
            self.last_response_metadata = {
                "provider": "openai",
                "model": self.model_name,
                "status": "response_decode_error",
                "http_status": http_status,
                "request_id": request_id,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                },
                "max_completion_tokens": max_completion_tokens,
                "reasoning_effort": reasoning_effort,
                "empty_response": True,
            }
            print(f"\033[31m[VLM API ERROR] GPT4v response was not JSON: {e}\033[0m")
            return ""

        if not isinstance(response, dict):
            self.last_response_metadata = {
                "provider": "openai",
                "model": self.model_name,
                "status": "response_shape_error",
                "http_status": http_status,
                "request_id": request_id,
                "error": {
                    "type": type(response).__name__,
                    "message": "Expected a JSON object from the API",
                },
                "max_completion_tokens": max_completion_tokens,
                "reasoning_effort": reasoning_effort,
                "empty_response": True,
            }
            print("\033[31m[VLM API ERROR] GPT4v response JSON was not an object\033[0m")
            return ""

        usage = response.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        choices = response.get("choices")
        self.last_response_metadata = {
            "provider": "openai",
            "model": response.get("model") or self.model_name,
            "response_id": response.get("id"),
            "status": "ok" if isinstance(choices, list) and choices else "api_error",
            "http_status": http_status,
            "request_id": request_id,
            "finish_reasons": [
                choice.get("finish_reason")
                for choice in choices or []
                if isinstance(choice, dict)
            ],
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": prompt_details.get("cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": completion_details.get("reasoning_tokens"),
            "accepted_prediction_tokens": completion_details.get(
                "accepted_prediction_tokens"
            ),
            "rejected_prediction_tokens": completion_details.get(
                "rejected_prediction_tokens"
            ),
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": reasoning_effort,
        }
        api_error = response.get("error")
        if isinstance(api_error, dict):
            self.last_response_metadata["error"] = {
                key: api_error.get(key)
                for key in ("type", "code", "param", "message")
                if api_error.get(key) is not None
            }

        if not isinstance(choices, list) or not choices:
            self.last_response_metadata["empty_response"] = True
            print(f"\033[31m[VLM API ERROR] Unexpected response (no 'choices'):\033[0m")
            pprint(response)
            return ""

        if any(
            not isinstance(choice, dict)
            or not isinstance(choice.get("message"), dict)
            for choice in choices
        ):
            self.last_response_metadata["status"] = "response_shape_error"
            self.last_response_metadata["error"] = {
                "type": "invalid_choices",
                "message": "A response choice did not contain a message object",
            }
            self.last_response_metadata["empty_response"] = True
            print("\033[31m[VLM API ERROR] GPT4v choices were malformed\033[0m")
            return ""

        self.token_usage.record_openai(response, model=self.model_name)
        answers = [ans["message"] for ans in choices]
        self.context.append({"role": "user", "content": prompt})
        self.context.extend(answers)

        contents = []
        for answer in answers:
            content = answer["content"]
            if isinstance(content, str) and content.startswith("```json"):
                content = content.replace("\n", "").replace("```json{", "{").replace("}```", "}")
                try:
                    content = json.loads(content)
                except Exception:
                    pass
            contents.append(content)

        result = contents[0] if len(contents) == 1 else contents
        self.last_response_metadata["empty_response"] = not bool(
            str(result or "").strip()
        )
        return result


class Claude3Api(VLMApi):
    name = "Claude-3"

    def __init__(self, model_name="claude-3-opus-20240229", cache_path=None):
        super().__init__(model_name=model_name, cache_path=cache_path)
        self.api_key = os.getenv("ANTHROPIC_API_KEY", None)
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.client = None
        self.messages = []

    def new_session(self):
        super().new_session()
        self.messages = []

    def _ask(
        self,
        prompt,
        image_data_urls=None,
        max_completion_tokens=800,
        temperature=0.2,
        **kwargs,
    ):
        try:
            import anthropic
        except Exception as e:
            raise ImportError("anthropic is required for Claude3Api") from e

        if self.client is None:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.messages = []

        urls = list(image_data_urls) if image_data_urls else []
        if not urls:
            content = prompt
        else:
            content = []
            for i, image_data_url in enumerate(urls):
                if i > 0:
                    content.append({
                        "type": "text",
                        "text": "[Scene view 2 — remaining object labels]",
                    })
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data_url.split(",")[-1],
                    },
                })
            content.append({"type": "text", "text": prompt})

        self.messages.append({"role": "user", "content": content})
        self._append_api_prompt_log(prompt, urls)
        message = self.client.messages.create(
            model=self.model_name,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            messages=self.messages,
        )
        self.messages.append({"role": "assistant", "content": message.content})
        usage = getattr(message, "usage", None)
        self.token_usage.record(
            model=self.model_name,
            prompt_tokens=getattr(usage, "input_tokens", 0) if usage is not None else 0,
            completion_tokens=getattr(usage, "output_tokens", 0) if usage is not None else 0,
            source="api",
        )
        # Also update shared context so caching replay works
        self.context.append({"role": "user", "content": prompt})
        self.context.append({"role": "assistant", "content": message.content[0].text if message.content else ""})
        responses = [c.text for c in message.content]
        return responses[0] if len(responses) == 1 else responses
