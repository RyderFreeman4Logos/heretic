# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from __future__ import annotations

import atexit
import logging
import os
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor

from .config import KlMode, Settings
from .model import Model
from .utils import Prompt, format_duration, load_prompts, print

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._evaluator = evaluator
        self.kl_divergence = kl_divergence
        self._responses = responses
        self._judge_future = judge_future
        self._thinking_completion_rate = thinking_completion_rate
        self._thinking_failures = thinking_failures
        self._thinking_samples = thinking_samples
        self.phase_times: dict[str, float] = phase_times or {}
        self._judge_submit_time = time.monotonic()

    def resolve(self, timeout: float | None = None) -> TrialEvaluation:
        """Block until LLM judge completes and compute final evaluation.

        Args:
            timeout: Maximum seconds to wait for the LLM judge future.
                     None means wait indefinitely. On timeout, falls back
                     to substring matching.
        """
        ev = self._evaluator

        refusal_flags: list[bool] | None = None
        if self._judge_future is not None:
            wait_start = time.monotonic()
            try:
                refusal_flags = self._judge_future.result(timeout=timeout)
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
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

        if settings.kl_mode == KlMode.SEQUENCE:
            print("* Generating reference responses for sequence-level KL...")
            self.reference_ids, self.reference_mask = model.generate_reference_ids(
                self.good_prompts, settings.kl_sequence_length
            )
            print("* Saving base sequence-level probability distributions to disk...")
            fd, self._base_logprobs_file = tempfile.mkstemp(
                suffix=".dat", prefix="heretic_base_logprobs_"
            )
            os.close(fd)
            self._base_logprobs_shape = model.save_sequence_logprobs_to_disk(
                self.good_prompts, self.reference_ids, self._base_logprobs_file
            )
            atexit.register(self._cleanup_logprobs_file)
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
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

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

        if self.settings.print_responses:
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

    def _cleanup_logprobs_file(self) -> None:
        path = getattr(self, "_base_logprobs_file", None)
        if path and os.path.exists(path):
            os.unlink(path)

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
        if self.settings.use_llm_judge:
            judge_future = self._judge_executor.submit(
                self._try_llm_judge,
                responses,
            )
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
        )

    def get_score(self) -> TrialEvaluation:
        """Synchronous evaluation (backward compatible)."""
        pending = self.start_evaluation()
        result = pending.resolve()
        print(f"  * Refusals: [bold]{result.refusals}[/]/{len(self.bad_prompts)}")
        return result
