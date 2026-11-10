"""Opt-in Phase B generation using owned, immutable PEFT snapshots.

The engine must be dedicated, trusted and mounted on the same adapter files.
No engine is started, no base weights are changed, and errors never fall back.
"""

import json
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import cast
from urllib.parse import urlsplit
from uuid import uuid4

__all__ = ["VllmBackend", "VllmBackendError"]


class VllmBackendError(RuntimeError):
    """Generation failed or remote adapter ownership became uncertain."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise VllmBackendError("vLLM redirects are not accepted")


class VllmBackend:
    def __init__(
        self,
        endpoint: str,
        local_adapter_dir: Path,
        server_adapter_dir: str,
        *,
        timeout: float = 300,
        api_key: str | None = None,
    ) -> None:
        url = urlsplit(endpoint)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or url.path.rstrip("/") != "/v1"
        ):
            raise ValueError("vLLM endpoint must be an explicit HTTP(S) /v1 URL")
        self.root = local_adapter_dir.resolve(strict=True)
        self.server_root = PurePosixPath(server_adapter_dir)
        if not self.root.is_dir() or not self.server_root.is_absolute():
            raise ValueError(
                "vLLM adapter directories must be existing local/absolute server paths"
            )
        if not 0 < timeout <= 3600:
            raise ValueError("vLLM timeout must be in (0, 3600] seconds")
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        if api_key is not None and ("\n" in api_key or "\r" in api_key):
            raise ValueError("vLLM API key contains invalid header characters")
        self.api_key = api_key
        # Never forward local administrative credentials through ambient proxies.
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )
        self.poisoned = False

    def _post(self, route: str, payload: dict[str, object]) -> bytes:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint + route, json.dumps(payload).encode(), headers, method="POST"
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                body = response.read(16 * 1024 * 1024 + 1)
                if len(body) > 16 * 1024 * 1024:
                    raise VllmBackendError("vLLM response exceeds 16 MiB")
                return body
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            # Do not echo response bodies, prompts, keys or credential-bearing URLs.
            raise VllmBackendError(
                f"vLLM {route} failed ({type(error).__name__})"
            ) from error

    @staticmethod
    def _choices(body: bytes, count: int) -> list[list[int]]:
        try:
            result = json.loads(body)
            if not isinstance(result, dict) or not isinstance(
                result.get("choices"), list
            ):
                raise TypeError("missing choices")
            choices = result["choices"]
            ordered: dict[int, list[int]] = {}
            for choice in choices:
                if not isinstance(choice, dict):
                    raise TypeError("invalid choice")
                index, tokens = choice.get("index"), choice.get("token_ids")
                if type(index) is not int or not 0 <= index < count or index in ordered:
                    raise ValueError("invalid or duplicate response index")
                if not isinstance(tokens, list) or any(
                    type(t) is not int or t < 0 for t in tokens
                ):
                    raise ValueError("engine must return generated token_ids")
                ordered[index] = cast(list[int], tokens)
            if set(ordered) != set(range(count)):
                raise ValueError("incomplete responses")
            return [ordered[index] for index in range(count)]
        except (ValueError, TypeError) as error:
            raise VllmBackendError("invalid vLLM completion response") from error

    def generate(
        self,
        prompt_ids: list[list[int]],
        export_adapter: Callable[[Path], None],
        *,
        base_model: str,
        max_tokens: int,
        eos_token_ids: list[int],
        batch_size: int,
    ) -> list[list[int]]:
        if self.poisoned:
            raise VllmBackendError(
                "previous adapter unload failed; reconcile the owned engine first"
            )
        if not prompt_ids:
            return []
        if batch_size < 1 or max_tokens < 1:
            raise ValueError("generation sizes must be positive")
        name = "heretic-" + uuid4().hex
        snapshot = Path(tempfile.mkdtemp(prefix=name + "-", dir=self.root))
        attempted = False
        try:
            export_adapter(snapshot)
            config = json.loads((snapshot / "adapter_config.json").read_text())
            if (
                not isinstance(config, dict)
                or config.get("base_model_name_or_path") != base_model
                or config.get("peft_type") != "LORA"
            ):
                raise VllmBackendError(
                    "snapshot is not a LoRA adapter for the configured base model"
                )
            if not (snapshot / "adapter_model.safetensors").is_file():
                raise VllmBackendError("snapshot must contain safe adapter weights")
            attempted = True
            self.poisoned = True
            self._post(
                "/load_lora_adapter",
                {
                    "lora_name": name,
                    "lora_path": str(self.server_root / snapshot.name),
                },
            )
            outputs: list[list[int]] = []
            for start in range(0, len(prompt_ids), batch_size):
                batch = prompt_ids[start : start + batch_size]
                body = self._post(
                    "/completions",
                    {
                        "model": name,
                        "prompt": batch,
                        "n": 1,
                        "stream": False,
                        "temperature": 0,
                        "max_tokens": max_tokens,
                        "stop_token_ids": eos_token_ids,
                        "return_token_ids": True,
                        "add_special_tokens": False,
                        "cache_salt": name,
                    },
                )
                outputs.extend(self._choices(body, len(batch)))
            return outputs
        finally:
            if attempted:
                try:
                    self._post("/unload_lora_adapter", {"lora_name": name})
                except VllmBackendError as error:
                    raise VllmBackendError(
                        f"owned adapter {name} unload unconfirmed; retained snapshot {snapshot}"
                    ) from error
                self.poisoned = False
            shutil.rmtree(snapshot)
