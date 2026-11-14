"""极简 OpenAI 兼容客户端（只用标准库，保持仓库零第三方依赖）。

为什么自己写而不引 openai 包：
  1) 评测镜像里只装 numpy，多一个依赖就多一条出错路径；
  2) 我们只需要 /v1/chat/completions 一个接口，urllib 足够；
  3) 断网环境下要能"快速失败"——超时、拒连、格式错都必须变成可捕获的异常，
     让上层立刻降级回规则，而不是卡住整个窗口。

关于 thinking：
  vLLM 部署的 Qwen3 系列默认会先"想"一长段再出答案，实测单次 446 tokens、
  content 甚至可能为空（全在 reasoning 里）。我们只要一个 JSON，所以默认
  用 chat_template_kwargs.enable_thinking=False 把思考关掉（实测 6 tokens、0.2s）。
  但这个参数不是所有 OpenAI 兼容端点都认，遇到 400 会自动去掉重试一次。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    """任何"这次没调通"的情况：连不上、超时、HTTP 非 200、返回体不是 JSON。"""


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout: float = 20.0, no_think: bool = True):
        if not base_url:
            raise LLMError("base_url 为空")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.no_think = no_think
        self.n_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0

    def _post(self, messages: list[dict], temperature: float, max_tokens: int,
              no_think: bool) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if no_think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST")
        self.n_calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMError(f"HTTP {e.code}: {e.read()[:200]!r}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"连接失败: {e}") from e
        except json.JSONDecodeError as e:
            raise LLMError(f"返回不是 JSON: {e}") from e

    def chat(self, messages: list[dict], temperature: float = 0.0,
             max_tokens: int = 256) -> str:
        try:
            payload = self._post(messages, temperature, max_tokens, self.no_think)
        except LLMError as e:
            # 端点不认 chat_template_kwargs 时（400），去掉该参数重试一次。
            if self.no_think and "HTTP 400" in str(e):
                payload = self._post(messages, temperature, max_tokens, False)
            else:
                raise

        usage = payload.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        self.reasoning_tokens += int(details.get("reasoning_tokens") or 0)

        try:
            msg = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"返回体缺 choices[0].message: {str(payload)[:200]}") from e

        text = msg.get("content")
        if not text:                      # thinking 没关掉时，答案可能全在 reasoning 里
            text = msg.get("reasoning") or ""
        if not text:
            raise LLMError(f"content 与 reasoning 都为空: {str(payload)[:200]}")
        return text

    def stats(self) -> dict:
        return {
            "llm_calls": self.n_calls,
            "llm_prompt_tokens": self.prompt_tokens,
            "llm_completion_tokens": self.completion_tokens,
            "llm_reasoning_tokens": self.reasoning_tokens,
            "llm_total_tokens": self.prompt_tokens + self.completion_tokens,
        }
