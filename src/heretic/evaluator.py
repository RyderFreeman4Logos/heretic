# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path

import torch.nn.functional as F
from torch import Tensor

from .config import KlMode, Settings
from .model import Model
from .utils import Prompt, format_duration, load_prompts, print

logger = logging.getLogger(__name__)

# Bump this when cache-affecting logic changes (generation, logprobs, serialization).
_CACHE_VERSION = 1


def _hash_json(data: object) -> str:
    """Return a stable SHA256 hash for JSON-serializable data."""
    canonical = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrialEvaluation:
    """Structured result from a single trial evaluation."""

    objectives: tuple[float, ...]
    kl_divergence: float
    refusals: int
    thinking_completion_rate: float | None = None
    thinking_failures: int | None = None
    thinking_samples: int = 0


class PendingScore:
    """Holds GPU results and a background LLM judge future for pipelined evaluation."""

    def __init__(
        self,
        evaluator: Evaluator,
        kl_divergence: float,
        responses: list[str],
        judge_future: Future[list[bool] | None] | None,
        thinking_completion_rate: float | None = None,
        thinking_failures: int | None = None,
        thinking_samples: int = 0,
        phase_times: dict[str, float] | None = None,
        judge_submit_time: float | None = None,
    ) -> None:
        self._evaluator = evaluator
        self.kl_divergence = kl_divergence
        self._responses = responses
        self._judge_future = judge_future
        self._thinking_completion_rate = thinking_completion_rate
        self._thinking_failures = thinking_failures
        self._thinking_samples = thinking_samples
        self.phase_times: dict[str, float] = phase_times or {}
        self._judge_submit_time = judge_submit_time or time.monotonic()

    def resolve(self, timeout: float | None = None) -> TrialEvaluation:
        """Block until LLM judge completes and compute final evaluation.

        Args:
            timeout: Maximum seconds to wait for the LLM judge future.
                     None means wait indefinitely. When fallback_policy
                     is ``"never"``, timeout is ignored (always waits).
        """
        ev = self._evaluator

        refusal_flags: list[bool] | None = None
        if self._judge_future is not None:
            # When fallback_policy="never", classify_refusals_batch already
            # retries internally until success, so we always wait indefinitely.
            from .llm_judge import get_config as _get_judge_config

            judge_cfg = _get_judge_config()
            effective_timeout = (
                None if judge_cfg.fallback_policy == "never" else timeout
            )

            wait_start = time.monotonic()
            try:
                refusal_flags = self._judge_future.result(timeout=effective_timeout)
            except TimeoutError:
                logger.warning(
                    f"LLM judge timed out after {timeout:.1f}s, falling back to substring",
                )
            except Exception:
                logger.warning("Pipelined LLM judge raised", exc_info=True)
            judge_wait = time.monotonic() - wait_start
            judge_total = time.monotonic() - self._judge_submit_time
            self.phase_times["judge_wait"] = judge_wait
            self.phase_times["judge_total"] = judge_total
            if judge_wait < 0.5:
                print(
                    f"  * LLM judge: resolved in {format_duration(judge_wait)} "
                    f"(completed during prior phases)"
                )
            else:
                print(
                    f"  * LLM judge: waited {format_duration(judge_wait)} "
                    f"(total {format_duration(judge_total)})"
                )

        ev._last_used_llm_judge = refusal_flags is not None

        # When substring_matching is disabled but LLM judge failed, fall back
        # to substring with a warning (safety behavior).
        use_substring_fallback = (
            refusal_flags is None
            and ev.settings.use_llm_judge
            and not ev.settings.substring_matching
        )
        if use_substring_fallback:
            logger.warning(
                "LLM judge failed and substring_matching is disabled; "
                "falling back to substring matching for safety",
            )

        refusals = 0
        for i, response in enumerate(self._responses):
            is_ref = (
                refusal_flags[i]
                if refusal_flags is not None
                else ev.is_refusal(response)
            )
            if is_ref:
                refusals += 1

            if ev.settings.print_responses:
                ev._print_response(ev.bad_prompts[i], response, is_ref)

        if ev.settings.print_responses:
            print()

        if ev._last_used_llm_judge and ev._base_refusals_llm is not None:
            base = ev._base_refusals_llm
        else:
            base = ev._base_refusals_substring

        refusals_score = refusals / base if base > 0 else float(refusals)
        kl_target = ev.settings.kl_divergence_target
        kl_scale = ev.settings.kl_divergence_scale

        if self.kl_divergence >= kl_target:
            kld_score = self.kl_divergence / kl_scale
        else:
            kld_score = refusals_score * kl_target / kl_scale

        # Build objective tuple: 2 objectives baseline, optional 3rd for thinking.
        objectives: tuple[float, ...]
        if self._thinking_completion_rate is not None:
            thinking_incompletion = 1.0 - self._thinking_completion_rate
            objectives = (kld_score, refusals_score, thinking_incompletion)
        else:
            objectives = (kld_score, refusals_score)

        return TrialEvaluation(
            objectives=objectives,
            kl_divergence=self.kl_divergence,
            refusals=refusals,
            thinking_completion_rate=self._thinking_completion_rate,
            thinking_failures=self._thinking_failures,
            thinking_samples=self._thinking_samples,
        )


class Evaluator:
    settings: Settings
    model: Model
    good_prompts: list[Prompt]
    bad_prompts: list[Prompt]
    base_logprobs: Tensor
    base_refusals: int
    reference_ids: Tensor | None
    reference_mask: Tensor | None

    def __init__(self, settings: Settings, model: Model) -> None:
        self.settings = settings
        self.model = model
        self._judge_executor = ThreadPoolExecutor(max_workers=1)
        atexit.register(self._judge_executor.shutdown, wait=False)

        # Track dual baselines for score consistency across LLM judge fallback.
        self._base_refusals_llm: int | None = None
        self._base_refusals_substring: int = 0
        self._last_used_llm_judge: bool = False

        # Check LLM judge dependency upfront so users know immediately.
        if settings.use_llm_judge:
            try:
                import httpx  # noqa: F401
            except ImportError:
                print(
                    "[bold yellow]WARNING: use_llm_judge is enabled but httpx is not installed.[/]"
                )
                print("[yellow]Install with: pip install heretic-llm\\[llm-judge][/]")
                print(
                    "[yellow]Falling back to substring matching for refusal classification.[/]"
                )
                settings.use_llm_judge = False

        print()
        print(
            f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset}[/]..."
        )
        self.good_prompts = load_prompts(settings, settings.good_evaluation_prompts)
        total_good = len(self.good_prompts)
        if (
            settings.max_good_eval_prompts > 0
            and settings.max_good_eval_prompts < total_good
        ):
            rng = random.Random(42)
            self.good_prompts = rng.sample(
                self.good_prompts, settings.max_good_eval_prompts
            )
            print(
                f"* [bold]{len(self.good_prompts)}[/]/{total_good} prompts loaded (sampled)"
            )
        else:
            print(f"* [bold]{total_good}[/] prompts loaded")

        if settings.kl_mode == KlMode.SEQUENCE:
            cache_key = self._compute_baseline_cache_key()
            cached = self._try_load_baseline_cache(cache_key)
            if cached is not None:
                ref_ids, ref_mask, dat_path, logprobs_shape = cached
                self.reference_ids = ref_ids
                self.reference_mask = ref_mask
                self._base_logprobs_file = dat_path
                self._base_logprobs_shape = logprobs_shape
                print("* [bold green]Loaded cached[/] sequence KL baseline")
            else:
                print("* Generating reference responses for sequence-level KL...")
                self.reference_ids, self.reference_mask = model.generate_reference_ids(
                    self.good_prompts, settings.kl_sequence_length
                )
                _, dat_path_obj = self._baseline_cache_paths(cache_key)
                dat_path_obj.parent.mkdir(parents=True, exist_ok=True)
                self._base_logprobs_file = str(dat_path_obj)
                print(
                    "* Saving base sequence-level probability distributions to disk..."
                )
                self._base_logprobs_shape = model.save_sequence_logprobs_to_disk(
                    self.good_prompts, self.reference_ids, self._base_logprobs_file
                )
                self._save_baseline_cache(
                    cache_key,
                    self.reference_ids,
                    self.reference_mask,
                    self._base_logprobs_shape,
                )
            self.base_logprobs = None  # type: ignore[assignment]
        else:
            self.reference_ids = None
            self.reference_mask = None
            print("* Obtaining first-token probability distributions...")
            self.base_logprobs = model.get_logprobs_batched(self.good_prompts)

        print()
        print(
            f"Loading bad evaluation prompts from [bold]{settings.bad_evaluation_prompts.dataset}[/]..."
        )
        self.bad_prompts = load_prompts(settings, settings.bad_evaluation_prompts)
        total_bad = len(self.bad_prompts)
        if (
            settings.max_bad_eval_prompts > 0
            and settings.max_bad_eval_prompts < total_bad
        ):
            rng = random.Random(42)
            self.bad_prompts = rng.sample(
                self.bad_prompts, settings.max_bad_eval_prompts
            )
            print(
                f"* [bold]{len(self.bad_prompts)}[/]/{total_bad} prompts loaded (sampled)"
            )
        else:
            print(f"* [bold]{total_bad}[/] prompts loaded")

        cache_key = self._compute_refusal_baseline_cache_key()
        base_responses: list[str] | None = None
        cached_refusals = None
        skip_cache_load = self.settings.print_responses

        if skip_cache_load:
            print("* Refusal baseline cache bypassed (print_responses enabled)")
        else:
            cached_refusals = self._try_load_refusal_baseline_cache(cache_key)

        if cached_refusals is not None:
            substring_refusals, llm_refusals = cached_refusals
            self._base_refusals_substring = substring_refusals
            self._base_refusals_llm = llm_refusals
            self.base_refusals = (
                self._base_refusals_llm
                if self._base_refusals_llm is not None
                else self._base_refusals_substring
            )
            print("* [bold green]Loaded cached[/] refusal baseline")
        else:
            if not skip_cache_load:
                print("* Refusal baseline cache miss")
            print("* Counting model refusals...")
            base_responses = model.get_responses_batched(
                self.bad_prompts,
                skip_special_tokens=True,
            )

            # Always compute substring baseline (used as fallback even when disabled).
            self._base_refusals_substring = sum(
                1 for r in base_responses if self.is_refusal(r)
            )

            # Try LLM judge for baseline if enabled.
            if settings.use_llm_judge:
                flags = self._try_llm_judge(base_responses)
                if flags is not None:
                    self._base_refusals_llm = sum(flags)
                    self.base_refusals = self._base_refusals_llm
                    if settings.substring_matching:
                        logger.info(
                            f"Baseline: LLM judge={self._base_refusals_llm}, substring={self._base_refusals_substring}",
                        )
                    else:
                        logger.info(
                            f"Baseline: LLM judge={self._base_refusals_llm} (substring matching disabled)",
                        )
                else:
                    self.base_refusals = self._base_refusals_substring
                    logger.warning(
                        f"Baseline LLM judge failed, using substring ({self.base_refusals})",
                    )
            else:
                self.base_refusals = self._base_refusals_substring

            self._save_refusal_baseline_cache(
                cache_key,
                self._base_refusals_substring,
                self._base_refusals_llm,
            )

        if base_responses is not None and self.settings.print_responses:
            for prompt, response in zip(self.bad_prompts, base_responses):
                self._print_response(prompt, response, self.is_refusal(response))
            print()

        print(
            f"* Initial refusals: [bold]{self.base_refusals}[/]/{len(self.bad_prompts)}"
        )

        # Load thinking evaluation prompts when the feature is active.
        self.thinking_prompts: list[Prompt] = []
        self._thinking_eval_active = (
            settings.thinking_eval_enabled
            and settings.thinking_eval_prompts is not None
            and model.thinking_profile is not None
        )
        if self._thinking_eval_active:
            assert settings.thinking_eval_prompts is not None
            print()
            print(
                f"Loading thinking evaluation prompts from "
                f"[bold]{settings.thinking_eval_prompts.dataset}[/]..."
            )
            self.thinking_prompts = load_prompts(
                settings, settings.thinking_eval_prompts
            )
            print(f"* [bold]{len(self.thinking_prompts)}[/] prompts loaded")
            if not self.thinking_prompts:
                print(
                    "[yellow]Warning:[/] thinking prompt dataset is empty. "
                    "Thinking evaluation will be skipped."
                )
                self._thinking_eval_active = False

    def _compute_baseline_cache_key(self) -> str:
        """Compute a deterministic cache key for the sequence KL baseline."""
        ds = self.settings.good_evaluation_prompts
        resolved_system_prompt = (
            ds.system_prompt
            if ds.system_prompt is not None
            else self.settings.system_prompt
        )
        key_data = {
            "cache_version": _CACHE_VERSION,
            "model": self.settings.model,
            "quantization": self.settings.quantization.value,
            "kl_sequence_length": self.settings.kl_sequence_length,
            "max_good_eval_prompts": self.settings.max_good_eval_prompts,
            "dataset": ds.dataset,
            "split": ds.split,
            "column": ds.column,
            "prefix": ds.prefix,
            "suffix": ds.suffix,
            "system_prompt": resolved_system_prompt,
        }
        return _hash_json(key_data)

    def _baseline_cache_paths(self, cache_key: str) -> tuple[Path, Path]:
        """Return (pt_path, dat_path) for a given cache key."""
        stem = f"kl_baseline_{cache_key[:16]}"
        pt_path = Path("checkpoints") / f"{stem}.pt"
        dat_path = Path("checkpoints") / f"{stem}.dat"
        return pt_path, dat_path

    def _try_load_baseline_cache(
        self, cache_key: str
    ) -> tuple[Tensor, Tensor, str, tuple[int, ...]] | None:
        """Try loading cached baseline. Returns None on cache miss."""
        pt_path, dat_path = self._baseline_cache_paths(cache_key)
        if not pt_path.exists() or not dat_path.exists():
            return None
        try:
            import torch

            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            if data.get("cache_key") != cache_key:
                logger.warning("Cache key mismatch in %s, ignoring", pt_path)
                return None
            return (
                data["reference_ids"],
                data["reference_mask"],
                str(dat_path),
                tuple(data["logprobs_shape"]),
            )
        except (RuntimeError, KeyError, OSError, EOFError, ValueError):
            logger.warning(
                "Failed to load baseline cache from %s", pt_path, exc_info=True
            )
            return None

    def _save_baseline_cache(
        self,
        cache_key: str,
        reference_ids: Tensor,
        reference_mask: Tensor,
        logprobs_shape: tuple[int, ...],
    ) -> None:
        """Persist baseline tensors and metadata for future runs."""
        import torch

        pt_path, _ = self._baseline_cache_paths(cache_key)
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cache_key": cache_key,
                "reference_ids": reference_ids,
                "reference_mask": reference_mask,
                "logprobs_shape": logprobs_shape,
            },
            pt_path,
        )
        logger.info("Sequence KL baseline cache saved to %s", pt_path)

    def _compute_refusal_baseline_cache_key(self) -> str:
        """Compute a deterministic cache key for baseline refusal counts."""
        return _hash_json(self._get_refusal_baseline_cache_metadata())

    def _get_refusal_baseline_cache_metadata(self) -> dict[str, object]:
        """Build cache metadata for the refusal baseline."""
        model_config = getattr(self.model.model, "config", None)
        model_identity = {
            "requested_model": self.settings.model,
            "loaded_model": getattr(model_config, "name_or_path", None),
            "commit_hash": getattr(model_config, "_commit_hash", None),
            "quantization": self.settings.quantization.value,
            "dtype": str(getattr(self.model.model, "dtype", "")),
            "max_response_length": self.settings.max_response_length,
            "response_prefix": self.model.response_prefix,
        }
        return {
            "cache_version": _CACHE_VERSION,
            "model_hash": _hash_json(model_identity),
            "eval_prompts_hash": self._hash_prompt_dataset(self.bad_prompts),
            "judge_config_hash": self._compute_judge_config_hash(),
            "use_llm_judge": self.settings.use_llm_judge,
            "substring_matching": self.settings.substring_matching,
            "refusal_markers": self.settings.refusal_markers,
        }

    def _hash_prompt_dataset(self, prompts: list[Prompt]) -> str:
        """Hash the resolved evaluation prompts after sampling and templating."""
        hasher = hashlib.sha256()
        for prompt in prompts:
            hasher.update(
                json.dumps(
                    {
                        "system": prompt.system,
                        "user": prompt.user,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            hasher.update(b"\n")
        return hasher.hexdigest()

    def _compute_judge_config_hash(self) -> str:
        """Hash the effective LLM judge config used for baseline scoring."""
        if not self.settings.use_llm_judge:
            return _hash_json({"enabled": False})

        from .llm_judge import get_config

        config = get_config()
        return _hash_json(
            {
                "enabled": True,
                "api_base": config.api_base,
                "models": list(config.models),
                "batch_size": config.batch_size,
                "concurrency": config.concurrency,
                "timeout": config.timeout,
                "max_retries": config.max_retries,
                "pricing": config.pricing,
                "system_prompt": config.system_prompt,
            }
        )

    def _refusal_baseline_cache_path(self, cache_key: str) -> Path:
        """Return the JSON path for cached refusal baselines."""
        stem = f"baseline_refusals_{cache_key[:16]}"
        return Path("checkpoints") / f"{stem}.json"

    def _try_load_refusal_baseline_cache(
        self, cache_key: str
    ) -> tuple[int, int | None] | None:
        """Try loading a cached refusal baseline. Returns None on cache miss."""
        path = self._refusal_baseline_cache_path(cache_key)
        if not path.exists():
            return None

        try:
            with path.open(encoding="utf-8") as file:
                data = json.load(file)

            if data.get("cache_key") != cache_key:
                logger.warning("Refusal cache key mismatch in %s, ignoring", path)
                return None

            substring_refusals = data["substring_refusals"]
            llm_refusals = data.get("llm_refusals")
            if not isinstance(substring_refusals, int):
                raise ValueError("substring_refusals must be an int")
            if llm_refusals is not None and not isinstance(llm_refusals, int):
                raise ValueError("llm_refusals must be an int or null")

            logger.info("Refusal baseline cache loaded from %s", path)
            return (substring_refusals, llm_refusals)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            logger.warning(
                "Failed to load refusal baseline cache from %s",
                path,
                exc_info=True,
            )
            return None

    def _save_refusal_baseline_cache(
        self,
        cache_key: str,
        substring_refusals: int,
        llm_refusals: int | None,
    ) -> None:
        """Persist refusal baseline counts for future resume runs."""
        if self.settings.use_llm_judge and llm_refusals is None:
            logger.info("Skipping refusal baseline cache save because LLM judge failed")
            return

        path = self._refusal_baseline_cache_path(cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._get_refusal_baseline_cache_metadata()
        data = {
            "cache_key": cache_key,
            "base_refusals": (
                llm_refusals if llm_refusals is not None else substring_refusals
            ),
            "substring_refusals": substring_refusals,
            "llm_refusals": llm_refusals,
            **metadata,
        }
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        logger.info("Refusal baseline cache saved to %s", path)

    def _try_llm_judge(self, responses: list[str]) -> list[bool] | None:
        """Attempt LLM judge classification. Returns None on failure."""
        try:
            from .llm_judge import classify_refusals_batch

            prompt_texts = [p.user for p in self.bad_prompts]
            flags = classify_refusals_batch(prompt_texts, responses)
            if flags is not None:
                logger.info(f"LLM judge classified {len(flags)} responses")
            else:
                logger.warning("LLM judge returned None (all models exhausted)")
            return flags
        except ImportError:
            logger.warning("LLM judge module not available")
            return None
        except Exception:
            logger.warning("LLM judge unexpected error", exc_info=True)
            return None

    def is_refusal(self, response: str) -> bool:
        # Classify empty responses as refusals to avoid optimizing for them.
        if not response.strip():
            return True

        # Remove emphasis (e.g. "I *will not*...").
        response = response.lower().replace("*", "")

        # Normalize typographic apostrophes ("won't" -> "won't").
        response = response.replace("\u2019", "'")

        # Normalize whitespace between words to a single space.
        response = " ".join(response.split())

        for marker in self.settings.refusal_markers:
            if marker.lower() in response:
                return True

        return False

    def _print_response(self, prompt: Prompt, response: str, is_refusal: bool) -> None:
        """Print a single prompt/response pair when print_responses is enabled."""
        print()
        print(f"[bold]System prompt:[/] {prompt.system}")
        print(f"[bold]Prompt:[/] {prompt.user}")
        disp = "[italic]\\[empty][/]" if not response.strip() else response
        print(f"[bold]Response:[/] [{'red' if is_refusal else 'green'}]{disp}[/]")

    def evaluate_thinking(self, prompts: list[Prompt]) -> tuple[float, int, int]:
        """Run a thinking chain completion pass and return (rate, failures, samples).

        Temporarily clears model.response_prefix so the model generates with its
        native thinking syntax, then restores the original prefix in a finally guard.
        """
        profile = self.model.thinking_profile
        assert profile is not None

        original_prefix = self.model.response_prefix
        try:
            self.model.response_prefix = ""
            responses = self.model.get_responses_batched(
                prompts,
                skip_special_tokens=False,
                enable_thinking=True if profile.template_controlled else None,
            )
        finally:
            self.model.response_prefix = original_prefix

        failures = 0
        for response in responses:
            if not response.strip():
                failures += 1
                continue
            close_pos = response.find(profile.completion_marker)
            if profile.template_controlled:
                # Template-controlled models (e.g. Qwen3.5) put <think> in the
                # input via the chat template; only </think> appears in output.
                if close_pos < 0:
                    failures += 1
            else:
                open_pos = response.find(profile.opening_marker)
                if open_pos < 0 or close_pos < 0 or close_pos <= open_pos:
                    failures += 1

        total = len(prompts)
        rate = (total - failures) / total if total > 0 else 0.0
        return rate, failures, total

    def start_evaluation(
        self,
        thinking_prompts: list[Prompt] | None = None,
    ) -> PendingScore:
        """Run GPU work, submit LLM judge async, return pending score.

        The returned PendingScore can be resolved later (after the caller
        has started the next trial's GPU work) to get the final score.

        Args:
            thinking_prompts: Subset of thinking prompts for this trial.
                              Pass None to skip thinking evaluation.
        """
        phase_times: dict[str, float] = {}

        # GPU: generate responses for bad prompts.
        n_bad = len(self.bad_prompts)
        print(f"  * Generating responses [{n_bad} prompts]...", end=" ")
        t0 = time.monotonic()
        responses = self.model.get_responses_batched(
            self.bad_prompts,
            skip_special_tokens=True,
        )
        phase_times["gen"] = time.monotonic() - t0
        print(f"done ({format_duration(phase_times['gen'])})")

        # Submit LLM judge to background thread (non-blocking).
        judge_future: Future[list[bool] | None] | None = None
        judge_submit_time: float | None = None
        if self.settings.use_llm_judge:
            judge_future = self._judge_executor.submit(
                self._try_llm_judge,
                responses,
            )
            judge_submit_time = time.monotonic()
            print("  * LLM judge submitted (async)")

        # GPU: logprobs for good prompts (overlaps with LLM judge).
        n_good = len(self.good_prompts)
        if self.settings.kl_mode == KlMode.SEQUENCE:
            assert self.reference_ids is not None
            assert self.reference_mask is not None
            print(
                f"  * Computing streaming sequence-level KL divergence [{n_good} prompts]...",
                end=" ",
            )
            t0 = time.monotonic()
            kl_divergence = self.model.compute_sequence_kl_streaming(
                self.good_prompts,
                self.reference_ids,
                self.reference_mask,
                self._base_logprobs_file,
                self._base_logprobs_shape,
            )
            phase_times["kl"] = time.monotonic() - t0
            print(f"done ({format_duration(phase_times['kl'])})")
        else:
            print(
                f"  * Computing first-token KL divergence [{n_good} prompts]...",
                end=" ",
            )
            t0 = time.monotonic()
            logprobs = self.model.get_logprobs_batched(self.good_prompts)
            kl_divergence = F.kl_div(
                logprobs,
                self.base_logprobs,
                reduction="batchmean",
                log_target=True,
            ).item()
            phase_times["kl"] = time.monotonic() - t0
            print(f"done ({format_duration(phase_times['kl'])})")
        print(f"  * KL divergence: [bold]{kl_divergence:.4f}[/]")

        # GPU: secondary thinking pass (serial, but LLM judge runs in parallel).
        thinking_rate: float | None = None
        thinking_failures: int | None = None
        thinking_samples = 0
        if thinking_prompts:
            print(
                f"  * Evaluating thinking chain completion [{len(thinking_prompts)} prompts]...",
                end=" ",
            )
            t0 = time.monotonic()
            thinking_rate, thinking_failures, thinking_samples = self.evaluate_thinking(
                thinking_prompts
            )
            phase_times["thinking"] = time.monotonic() - t0
            pct = thinking_rate * 100
            completed = thinking_samples - thinking_failures
            print(f"done ({format_duration(phase_times['thinking'])})")
            print(
                f"  * Thinking evaluation: {completed}/{thinking_samples} "
                f"complete ({pct:.1f}%)"
            )

        return PendingScore(
            self,
            kl_divergence,
            responses,
            judge_future,
            thinking_completion_rate=thinking_rate,
            thinking_failures=thinking_failures,
            thinking_samples=thinking_samples,
            phase_times=phase_times,
            judge_submit_time=judge_submit_time,
        )

    def get_score(self) -> TrialEvaluation:
        """Synchronous evaluation (backward compatible)."""
        pending = self.start_evaluation()
        result = pending.resolve()
        print(f"  * Refusals: [bold]{result.refusals}[/]/{len(self.bad_prompts)}")
        return result
