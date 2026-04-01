# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# ruff: noqa: E402

from .progress import patch_tqdm

# This patches tqdm class definitions, which must happen
# before any other module imports tqdm.
patch_tqdm()

import hashlib
import json
import logging
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict
from importlib.metadata import version
from os.path import commonprefix
from pathlib import Path
from typing import Any

import huggingface_hub
import lm_eval
import numpy as np
import optuna
import questionary
import torch
import torch.nn.functional as F
import transformers
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)
from huggingface_hub import ModelCard, ModelCardData
from lm_eval.models.huggingface import HFLM
from optuna import Trial
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.study import StudyDirection
from optuna.trial import TrialState
from pydantic import ValidationError
from questionary import Choice, Style
from rich.table import Table
from rich.traceback import install

from .analyzer import Analyzer
from .config import DirectionMethod, QuantizationMethod, Settings
from .evaluator import Evaluator, PendingScore
from .model import AbliterationParameters, Model, ThinkingProfile, get_model_class
from .utils import (
    Prompt,
    empty_cache,
    format_duration,
    get_readme_intro,
    get_trial_parameters,
    load_prompts,
    print,
    print_memory_usage,
    prompt_password,
    prompt_path,
    prompt_select,
    prompt_text,
)

logger = logging.getLogger(__name__)

_RESIDUAL_CACHE_VERSION = 1


def _hash_json(data: object) -> str:
    """Return a stable SHA256 hash for JSON-serializable data."""
    canonical = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_prompt_dataset(prompts: list[Prompt]) -> str:
    """Hash the resolved prompts after all dataset transforms are applied."""
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


def _get_residual_cache_metadata(
    settings: Settings,
    model: Model,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
) -> dict[str, object]:
    """Build cache metadata for reusable residual tensors."""
    model_config = getattr(model.model, "config", None)
    return {
        "cache_version": _RESIDUAL_CACHE_VERSION,
        "model_id": settings.model,
        "loaded_model": getattr(model_config, "name_or_path", None),
        "commit_hash": getattr(model_config, "_commit_hash", None),
        "quantization": settings.quantization.value,
        "batch_size": settings.batch_size,
        "winsorization_quantile": settings.winsorization_quantile,
        "good_dataset_path": settings.good_prompts.dataset,
        "good_dataset_checksum": _hash_prompt_dataset(good_prompts),
        "bad_dataset_path": settings.bad_prompts.dataset,
        "bad_dataset_checksum": _hash_prompt_dataset(bad_prompts),
    }


def _residual_cache_path(cache_key: str) -> Path:
    """Return the cache file path for a residual cache key."""
    return Path("checkpoints") / f"residuals_{cache_key[:16]}.pt"


def _try_load_residual_cache(
    cache_key: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Try loading cached residual tensors. Returns None on cache miss."""
    path = _residual_cache_path(cache_key)
    if not path.exists():
        return None

    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(data, dict):
            raise ValueError("Residual cache payload must be a dict")
        if data.get("cache_key") != cache_key:
            logger.warning("Residual cache key mismatch in %s, ignoring", path)
            return None

        good_residuals = data["good_residuals"]
        bad_residuals = data["bad_residuals"]
        if not isinstance(good_residuals, torch.Tensor):
            raise ValueError("good_residuals must be a tensor")
        if not isinstance(bad_residuals, torch.Tensor):
            raise ValueError("bad_residuals must be a tensor")

        logger.info("Residual cache loaded from %s", path)
        return (good_residuals, bad_residuals)
    except (RuntimeError, KeyError, OSError, EOFError, ValueError):
        logger.warning("Failed to load residual cache from %s", path, exc_info=True)
        return None


def _save_residual_cache(
    cache_key: str,
    metadata: dict[str, object],
    good_residuals: torch.Tensor,
    bad_residuals: torch.Tensor,
) -> None:
    """Persist residual tensors for faster resume runs."""
    path = _residual_cache_path(cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache_key": cache_key,
            **metadata,
            "good_residuals": good_residuals,
            "bad_residuals": bad_residuals,
        },
        path,
    )
    logger.info("Residual cache saved to %s", path)


def obtain_merge_strategy(settings: Settings) -> str | None:
    """
    Prompts the user for how to proceed with saving the model.
    Provides info to the user if the model is quantized on memory use.
    Returns "merge", "adapter", or None (if cancelled/invalid).
    """

    if settings.quantization == QuantizationMethod.BNB_4BIT:
        print()
        print(
            "Model was loaded with quantization. Merging requires reloading the base model."
        )
        print(
            "[yellow]WARNING: CPU merging requires dequantizing the entire model to system RAM.[/]"
        )
        print("[yellow]This can lead to system freezes if you run out of memory.[/]")

        try:
            # Estimate memory requirements by loading the model structure on the "meta" device.
            # This doesn't consume actual RAM but allows us to inspect the parameter count/dtype.
            #
            # Suppress warnings during meta device loading (e.g., "Some weights were not initialized").
            # These are expected and harmless since we're only inspecting model structure, not running inference.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta_model = get_model_class(settings.model).from_pretrained(
                    settings.model,
                    device_map="meta",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                footprint_bytes = meta_model.get_memory_footprint()
                footprint_gb = footprint_bytes / (1024**3)
                print(
                    f"[yellow]Estimated RAM required (excluding overhead): [bold]~{footprint_gb:.2f} GB[/][/]"
                )
        except Exception:
            # Fallback if meta loading fails (e.g. owing to custom model code
            # or bitsandbytes quantization config issues on the meta device).
            print(
                "[yellow]Rule of thumb: You need approximately 3x the parameter count in GB RAM.[/]"
            )
            print(
                "[yellow]Example: A 27B model requires ~80GB RAM. A 70B model requires ~200GB RAM.[/]"
            )
        print()

        strategy = prompt_select(
            "How do you want to proceed?",
            choices=[
                Choice(
                    title="Merge LoRA into full model"
                    + (
                        ""
                        if settings.quantization == QuantizationMethod.NONE
                        else " (requires sufficient RAM)"
                    ),
                    value="merge",
                ),
                Choice(
                    title="Cancel",
                    value="cancel",
                ),
            ],
        )

        if strategy == "cancel":
            return None

        return strategy
    else:
        return "merge"


def detect_thinking_profile(prefix: str) -> ThinkingProfile | None:
    """Detect a known thinking syntax from a common response prefix."""
    if prefix.startswith("<think>"):
        return ThinkingProfile(
            name="think",
            opening_marker="<think>",
            completion_marker="</think>",
            suppressed_prefix="<think></think>",
        )
    if prefix.startswith("<|channel|>analysis<|message|>"):
        return ThinkingProfile(
            name="gpt-oss",
            opening_marker="<|channel|>analysis<|message|>",
            completion_marker="<|end|><|start|>assistant<|channel|>final<|message|>",
            suppressed_prefix=(
                "<|channel|>analysis<|message|>"
                "<|end|><|start|>assistant<|channel|>final<|message|>"
            ),
        )
    if prefix.startswith("<thought>"):
        return ThinkingProfile(
            name="thought",
            opening_marker="<thought>",
            completion_marker="</thought>",
            suppressed_prefix="<thought></thought>",
        )
    if prefix.startswith("[THINK]"):
        return ThinkingProfile(
            name="think-bracket",
            opening_marker="[THINK]",
            completion_marker="[/THINK]",
            suppressed_prefix="[THINK][/THINK]",
        )
    return None


def run():
    # Enable expandable segments to reduce memory fragmentation on multi-GPU setups.
    if (
        "PYTORCH_ALLOC_CONF" not in os.environ
        and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    ):
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Modified "Pagga" font from https://budavariam.github.io/asciiart-text/
    print(f"[cyan]█░█░█▀▀░█▀▄░█▀▀░▀█▀░█░█▀▀[/]  v{version('heretic-llm')}")
    print("[cyan]█▀█░█▀▀░█▀▄░█▀▀░░█░░█░█░░[/]")
    print(
        "[cyan]▀░▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀▀▀[/]  [blue underline]https://github.com/p-e-w/heretic[/]"
    )
    print()

    if (
        # There is at least one argument (argv[0] is the program name).
        len(sys.argv) > 1
        # No model has been explicitly provided.
        and "--model" not in sys.argv
        # The last argument is a parameter value rather than a flag (such as "--help").
        and not sys.argv[-1].startswith("-")
    ):
        # Assume the last argument is the model.
        sys.argv.insert(-1, "--model")

    try:
        # The required argument "model" must be provided by the user,
        # either on the command line or in the configuration file.
        settings = Settings()  # ty:ignore[missing-argument]
    except ValidationError as error:
        print(f"[red]Configuration contains [bold]{error.error_count()}[/] errors:[/]")

        for error in error.errors():
            print(f"[bold]{error['loc'][0]}[/]: [yellow]{error['msg']}[/]")

        print()
        print(
            "Run [bold]heretic --help[/] or see [bold]config.default.toml[/] for details about configuration parameters."
        )
        return

    # Adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/env.py
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        total_vram = sum(torch.cuda.mem_get_info(i)[1] for i in range(count))
        print(
            f"Detected [bold]{count}[/] CUDA device(s) ({total_vram / (1024**3):.2f} GB total VRAM):"
        )
        for i in range(count):
            vram = torch.cuda.mem_get_info(i)[1] / (1024**3)
            print(
                f"* GPU {i}: [bold]{torch.cuda.get_device_name(i)}[/] ({vram:.2f} GB)"
            )
    elif is_xpu_available():
        count = torch.xpu.device_count()
        print(f"Detected [bold]{count}[/] XPU device(s):")
        for i in range(count):
            print(f"* XPU {i}: [bold]{torch.xpu.get_device_name(i)}[/]")
    elif is_mlu_available():
        count = torch.mlu.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] MLU device(s):")
        for i in range(count):
            print(f"* MLU {i}: [bold]{torch.mlu.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_sdaa_available():
        count = torch.sdaa.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] SDAA device(s):")
        for i in range(count):
            print(f"* SDAA {i}: [bold]{torch.sdaa.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_musa_available():
        count = torch.musa.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] MUSA device(s):")
        for i in range(count):
            print(f"* MUSA {i}: [bold]{torch.musa.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_npu_available():
        print(f"NPU detected (CANN version: [bold]{torch.version.cann}[/])")  # ty:ignore[unresolved-attribute]
    elif torch.backends.mps.is_available():
        print("Detected [bold]1[/] MPS device (Apple Metal)")
    else:
        print(
            "[bold yellow]No GPU or other accelerator detected. Operations will be slow.[/]"
        )

    # We don't need gradients as we only do inference.
    torch.set_grad_enabled(False)

    # While determining the optimal batch size, we will try many different batch sizes,
    # resulting in many computation graphs being compiled. Raising the limit (default = 8)
    # avoids errors from TorchDynamo assuming that something is wrong because we
    # recompile too often.
    torch._dynamo.config.cache_size_limit = 64

    # Enable INFO logging for LLM judge and evaluator monitoring.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libraries.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Silence warning spam from Transformers.
    # In my entire career I've never seen a useful warning from that library.
    transformers.logging.set_verbosity_error()

    # Another library that generates warning spam.
    logging.getLogger("lm_eval").setLevel(logging.ERROR)

    # We do our own trial logging, so we don't need the INFO messages.
    # about parameters and results.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Silence the warning about multivariate TPE being experimental.
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    os.makedirs(settings.study_checkpoint_dir, exist_ok=True)

    study_checkpoint_file = os.path.join(
        settings.study_checkpoint_dir,
        "".join(
            [(c if (c.isalnum() or c in ["_", "-"]) else "--") for c in settings.model]
        )
        + ".jsonl",
    )

    lock_obj = JournalFileOpenLock(study_checkpoint_file)
    backend = JournalFileBackend(study_checkpoint_file, lock_obj=lock_obj)
    storage = JournalStorage(backend)

    try:
        existing_study = storage.get_all_studies()[0]
    except IndexError:
        existing_study = None

    if existing_study is not None and settings.evaluate_model is None:
        choices = []

        if existing_study.user_attrs["finished"]:
            print()
            print(
                (
                    "[green]You have already processed this model.[/] "
                    "You can show the results from the previous run, allowing you to export models or to run additional trials. "
                    "Alternatively, you can ignore the previous run and start from scratch. "
                    "This will delete the checkpoint file and all results from the previous run."
                )
            )
            choices.append(
                Choice(
                    title="Show the results from the previous run",
                    value="continue",
                )
            )
        else:
            print()
            print(
                (
                    "[yellow]You have already processed this model, but the run was interrupted.[/] "
                    "You can continue the previous run from where it stopped. This will override any specified settings. "
                    "Alternatively, you can ignore the previous run and start from scratch. "
                    "This will delete the checkpoint file and all results from the previous run."
                )
            )
            choices.append(
                Choice(
                    title="Continue the previous run",
                    value="continue",
                )
            )

        choices.append(
            Choice(
                title="Ignore the previous run and start from scratch",
                value="restart",
            )
        )

        choices.append(
            Choice(
                title="Exit program",
                value="",
            )
        )

        print()
        if not sys.stdin.isatty():
            # Auto-continue in non-interactive mode (e.g. nohup).
            if existing_study.user_attrs["finished"]:
                if settings.headless:
                    # Headless mode: skip to post-optimization trial selection.
                    choice = "finished_headless"
                else:
                    print(
                        "[yellow]Study already finished. Run interactively to select a trial.[/]"
                    )
                    return
            else:
                choice = "continue"
                print(
                    "[green]Auto-continuing interrupted run (non-interactive mode).[/]"
                )
        else:
            choice = prompt_select("How would you like to proceed?", choices)

        if choice == "continue" or choice == "finished_headless":
            # Preserve current run overrides across checkpoint restoration.
            headless_override = settings.headless
            output_dir_override = settings.output_dir
            trial_override = settings.trial
            n_trials_override = settings.n_trials
            settings = Settings.model_validate_json(
                existing_study.user_attrs["settings"]
            )
            if settings.n_trials != n_trials_override:
                print(
                    "[yellow]Warning:[/] n_trials from the current config "
                    f"([bold]{n_trials_override}[/]) differs from the checkpoint "
                    f"([bold]{settings.n_trials}[/]). Using the current config value."
                )
            settings.headless = headless_override
            settings.output_dir = output_dir_override
            settings.trial = trial_override
            settings.n_trials = n_trials_override
        elif choice == "restart":
            os.unlink(study_checkpoint_file)
            backend = JournalFileBackend(study_checkpoint_file, lock_obj=lock_obj)
            storage = JournalStorage(backend)
        elif choice is None or choice == "":
            return

    model = Model(settings)
    print()
    print_memory_usage()

    print()
    print(f"Loading good prompts from [bold]{settings.good_prompts.dataset}[/]...")
    good_prompts = load_prompts(settings, settings.good_prompts)
    print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

    print()
    print(f"Loading bad prompts from [bold]{settings.bad_prompts.dataset}[/]...")
    bad_prompts = load_prompts(settings, settings.bad_prompts)
    print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

    if settings.batch_size == 0:
        print()
        print("Determining optimal batch size...")

        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(
                f"* GPU memory: [bold]{total_mem / 1024**3:.1f}[/] GB total, "
                f"[bold]{free_mem / 1024**3:.1f}[/] GB free"
            )

        batch_size = 1
        best_batch_size = -1
        best_performance = -1.0
        regressions = 0

        while batch_size <= settings.max_batch_size:
            print(f"* Trying batch size [bold]{batch_size}[/]... ", end="")

            prompts = good_prompts * math.ceil(batch_size / len(good_prompts))
            prompts = prompts[:batch_size]

            try:
                # Warmup run to build the computation graph so that part isn't benchmarked.
                model.get_responses(prompts)

                start_time = time.perf_counter()
                responses = model.get_responses(prompts)
                end_time = time.perf_counter()
            except Exception as error:
                if batch_size == 1:
                    # Even a batch size of 1 already fails.
                    # We cannot recover from this.
                    raise

                empty_cache()
                print(f"[red]Failed[/] ({error})")
                break

            response_lengths = [
                len(model.tokenizer.encode(response)) for response in responses
            ]
            performance = sum(response_lengths) / (end_time - start_time)

            print(f"[green]Ok[/] ([bold]{performance:.0f}[/] tokens/s)")

            if performance > best_performance:
                best_batch_size = batch_size
                best_performance = performance
                regressions = 0
            else:
                regressions += 1
                if regressions >= 2:
                    print("* Throughput declining, stopping search")
                    break

            batch_size *= 2

        settings.batch_size = best_batch_size
        print(f"* Chosen batch size: [bold]{settings.batch_size}[/]")

    print()
    print("Checking for common response prefix...")
    prefix_check_prompts = good_prompts[:100] + bad_prompts[:100]
    responses = model.get_responses_batched(prefix_check_prompts)

    # Despite being located in os.path, commonprefix actually performs
    # a naive string operation without any path-specific logic,
    # which is exactly what we need here. Trailing spaces are removed
    # to avoid issues where multiple different tokens that all start
    # with a space character lead to the common prefix ending with
    # a space, which would result in an uncommon tokenization.
    model.response_prefix = commonprefix(responses).rstrip(" ")

    # Detect thinking profile and suppress CoT output.
    recheck_prefix = False
    if model.response_prefix:
        profile = detect_thinking_profile(model.response_prefix)
        if profile is not None:
            model.thinking_profile = profile
            model.response_prefix = profile.suppressed_prefix
            # Recheck: the predefined prefix may be missing a trailing newline.
            recheck_prefix = True

    if model.response_prefix:
        print(f"* Prefix found: [bold]{model.response_prefix!r}[/]")
    else:
        print("* None found")

    if recheck_prefix:
        print("* Rechecking with prefix...")
        responses = model.get_responses_batched(prefix_check_prompts)
        additional_prefix = commonprefix(responses).rstrip(" ")
        if additional_prefix:
            model.response_prefix += additional_prefix
            print(f"* Extended prefix found: [bold]{model.response_prefix!r}[/]")

    # Fallback: detect thinking model from chat template (e.g. Qwen3.5).
    # These models control thinking via the template's enable_thinking parameter
    # rather than emitting <think> as part of the model output.
    if (
        model.thinking_profile is None
        and hasattr(model.tokenizer, "chat_template")
        and model.tokenizer.chat_template
        and "enable_thinking" in model.tokenizer.chat_template
    ):
        model.thinking_profile = ThinkingProfile(
            name="think",
            opening_marker="<think>",
            completion_marker="</think>",
            suppressed_prefix="<think></think>",
            template_controlled=True,
        )
        print(
            "* Thinking model detected from chat template "
            f"(profile: [bold]{model.thinking_profile.name}[/], template-controlled)"
        )

    if settings.thinking_eval_enabled and model.thinking_profile is None:
        print(
            "[yellow]Warning:[/] thinking_eval_enabled is true but no supported "
            "thinking prefix was detected. Thinking evaluation will be skipped."
        )

    evaluator = Evaluator(settings, model)

    if settings.evaluate_model is not None:
        print()
        print(f"Loading model [bold]{settings.evaluate_model}[/]...")
        settings.model = settings.evaluate_model
        model.reset_model()
        print("* Evaluating...")
        evaluator.get_score()
        return

    print()
    print("Calculating per-layer refusal directions...")
    residual_cache_metadata = _get_residual_cache_metadata(
        settings,
        model,
        good_prompts,
        bad_prompts,
    )
    residual_cache_key = _hash_json(residual_cache_metadata)
    cached_residuals = _try_load_residual_cache(residual_cache_key)
    if cached_residuals is not None:
        good_residuals, bad_residuals = cached_residuals
        print("* [bold green]Loaded cached[/] residual tensors")
    else:
        print("* Residual cache miss")
        print("* Obtaining residuals for good prompts...")
        good_residuals = model.get_residuals_batched(good_prompts)
        print("* Obtaining residuals for bad prompts...")
        bad_residuals = model.get_residuals_batched(bad_prompts)
        _save_residual_cache(
            residual_cache_key,
            residual_cache_metadata,
            good_residuals,
            bad_residuals,
        )

    if settings.direction_method == DirectionMethod.GEOMETRIC_MEDIAN:
        try:
            from geom_median.torch import (  # ty:ignore[unresolved-import]
                compute_geometric_median,
            )
        except ImportError:
            raise ImportError(
                'direction_method = "geometric_median" requires the geom-median package. '
                "Install it with: uv pip install heretic-llm[research]"
            ) from None

        def _per_layer_geometric_median(residuals: torch.Tensor) -> torch.Tensor:
            device = residuals.device
            return torch.stack(
                [
                    compute_geometric_median(residuals[:, i, :].detach().cpu()).median
                    for i in range(residuals.shape[1])
                ]
            ).to(device)

        good_center = _per_layer_geometric_median(good_residuals)
        bad_center = _per_layer_geometric_median(bad_residuals)
    else:
        good_center = good_residuals.mean(dim=0)
        bad_center = bad_residuals.mean(dim=0)

    refusal_directions = F.normalize(bad_center - good_center, p=2, dim=1)

    if settings.orthogonalize_direction:
        # Implements https://huggingface.co/blog/grimjim/projected-abliteration
        # Adjust the refusal directions so that only the component that is
        # orthogonal to the good direction is subtracted during abliteration.
        good_directions = F.normalize(good_center, p=2, dim=1)
        projection_vector = torch.sum(refusal_directions * good_directions, dim=1)
        refusal_directions = (
            refusal_directions - projection_vector.unsqueeze(1) * good_directions
        )
        refusal_directions = F.normalize(refusal_directions, p=2, dim=1)

    analyzer = Analyzer(settings, model, good_residuals, bad_residuals)

    if settings.print_residual_geometry:
        analyzer.print_residual_geometry()

    if settings.plot_residuals:
        analyzer.plot_residuals()

    # We don't need the residuals after computing refusal directions.
    del good_residuals, bad_residuals, analyzer
    empty_cache()

    trial_index = 0
    start_index = 0

    last_layer_index = len(model.get_layers()) - 1

    def suggest_and_abliterate(trial: Trial, trial_idx: int) -> None:
        """Suggest parameters, reset model, and run abliteration (GPU)."""
        direction_scope = trial.suggest_categorical(
            "direction_scope",
            [
                "global",
                "per layer",
            ],
        )

        # Discrimination between "harmful" and "harmless" inputs is usually strongest
        # in layers slightly past the midpoint of the layer stack. See the original
        # abliteration paper (https://arxiv.org/abs/2406.11717) for a deeper analysis.
        #
        # Note that we always sample this parameter even though we only need it for
        # the "global" direction scope. The reason is that multivariate TPE doesn't
        # work with conditional or variable-range parameters.
        direction_index = trial.suggest_float(
            "direction_index",
            0.4 * last_layer_index,
            0.9 * last_layer_index,
        )

        if direction_scope == "per layer":
            direction_index = None

        parameters = {}

        for component in model.get_abliterable_components():
            # The parameter ranges are based on experiments with various models
            # and much wider ranges. They are not set in stone and might have to be
            # adjusted for future models.
            max_weight = trial.suggest_float(
                f"{component}.max_weight",
                0.8,
                1.5,
            )
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * last_layer_index,
                1.0 * last_layer_index,
            )
            # For sampling purposes, min_weight is expressed as a fraction of max_weight,
            # again because multivariate TPE doesn't support variable-range parameters.
            # The value is transformed into the actual min_weight value below.
            min_weight = trial.suggest_float(
                f"{component}.min_weight",
                0.0,
                1.0,
            )
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                0.6 * last_layer_index,
            )

            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=(min_weight * max_weight),
                min_weight_distance=min_weight_distance,
            )

        trial.set_user_attr("direction_index", direction_index)
        trial.set_user_attr("parameters", {k: asdict(v) for k, v in parameters.items()})

        print()
        print(f"Running trial [bold]{trial_idx}[/] of [bold]{settings.n_trials}[/]...")
        print("* Parameters:")
        for name, value in get_trial_parameters(trial).items():
            print(f"  * {name} = [bold]{value}[/]")
        print("* Resetting model...", end=" ")
        t0 = time.monotonic()
        model.reset_model()
        trial_phase_times["reset"] = time.monotonic() - t0
        print(f"done ({format_duration(trial_phase_times['reset'])})")
        print("* Abliterating...", end=" ")
        t0 = time.monotonic()
        model.abliterate(refusal_directions, direction_index, parameters)
        trial_phase_times["abliterate"] = time.monotonic() - t0
        print(f"done ({format_duration(trial_phase_times['abliterate'])})")

    def resolve_pending(
        pending: tuple[PendingScore, Trial, int, dict[str, float]] | None,
        timeout: float | None = None,
    ) -> None:
        """Resolve a pipelined evaluation and report score to Optuna."""
        if pending is None:
            return
        pending_score, prev_trial, prev_idx, phases = pending
        result = pending_score.resolve(timeout=timeout)
        print(f"  * Refusals: [bold]{result.refusals}[/]/{len(evaluator.bad_prompts)}")

        # Merge evaluator phase times into the trial phases.
        phases.update(pending_score.phase_times)

        # Print trial timing summary.
        # Exclude judge_total — it overlaps with kl/thinking (concurrent).
        sequential_keys = ("gen", "kl", "reset", "abliterate", "thinking", "judge_wait")
        total = sum(phases.get(k, 0.0) for k in sequential_keys)
        if total > 0:
            parts: list[str] = []
            for label in ("gen", "kl", "reset", "abliterate", "thinking", "judge_wait"):
                if label in phases:
                    pct = phases[label] / total * 100
                    parts.append(f"{label}: {pct:.0f}%")
            other = total - sum(
                phases.get(k, 0.0)
                for k in ("gen", "kl", "reset", "abliterate", "thinking", "judge_wait")
            )
            if other > 0.5:
                parts.append(f"other: {other / total * 100:.0f}%")
            print(
                f"* Trial {prev_idx} total: [bold]{format_duration(total)}[/] "
                f"({', '.join(parts)})"
            )

        elapsed_time = time.perf_counter() - start_time
        print()
        print(f"[grey50]Elapsed time: [bold]{format_duration(elapsed_time)}[/][/]")
        completed = prev_idx - start_index
        if completed > 0 and prev_idx < settings.n_trials:
            remaining_time = (elapsed_time / completed) * (settings.n_trials - prev_idx)
            print(
                f"[grey50]Estimated remaining time: [bold]{format_duration(remaining_time)}[/][/]"
            )
        print_memory_usage()

        prev_trial.set_user_attr("kl_divergence", result.kl_divergence)
        prev_trial.set_user_attr("refusals", result.refusals)
        if result.thinking_completion_rate is not None:
            prev_trial.set_user_attr(
                "thinking_completion_rate", result.thinking_completion_rate
            )
            prev_trial.set_user_attr("thinking_failures", result.thinking_failures)
            prev_trial.set_user_attr("thinking_samples", result.thinking_samples)
        study.tell(prev_trial, result.objectives)

    # Run baseline sanity check BEFORE creating the study so that the objective
    # shape is final. This avoids the bug where a 3-objective study is created
    # but baseline failure downgrades to 2-objective evaluations at runtime.
    thinking_eval_active = evaluator._thinking_eval_active
    trial_thinking_prompts: list[Prompt] = []
    thinking_eval_seed: int | None = None
    thinking_eval_indices: list[int] | None = None

    if thinking_eval_active and evaluator.thinking_prompts:
        # Select prompt subset for baseline check.
        seed = random.randint(0, 2**31 - 1)
        rng = random.Random(seed)
        indices = list(range(len(evaluator.thinking_prompts)))
        rng.shuffle(indices)
        n_samples = min(settings.thinking_eval_samples, len(indices))
        selected_indices = sorted(indices[:n_samples])
        trial_thinking_prompts = [
            evaluator.thinking_prompts[i] for i in selected_indices
        ]
        thinking_eval_seed = seed
        thinking_eval_indices = selected_indices
        print(f"* Thinking evaluation: using {n_samples} prompts (seed={seed})")

        # Baseline sanity check: evaluate thinking on the unablated model.
        print("* Running thinking baseline sanity check...")
        baseline_rate, baseline_failures, baseline_total = evaluator.evaluate_thinking(
            trial_thinking_prompts
        )
        pct = baseline_rate * 100
        print(
            f"  * Baseline thinking: {baseline_total - baseline_failures}/{baseline_total} "
            f"complete ({pct:.1f}%)"
        )
        if baseline_rate < 0.5:
            print(
                "[yellow]Warning:[/] baseline thinking completion rate is below 50%. "
                "The unablated model may not support this thinking syntax reliably. "
                "Disabling thinking evaluation for this run."
            )
            thinking_eval_active = False
            trial_thinking_prompts = []

    # Determine study objective shape AFTER baseline check is final.
    if thinking_eval_active:
        directions = [
            StudyDirection.MINIMIZE,
            StudyDirection.MINIMIZE,
            StudyDirection.MINIMIZE,
        ]
        objective_names = ["kld_score", "refusals_score", "thinking_incompletion"]
    else:
        directions = [StudyDirection.MINIMIZE, StudyDirection.MINIMIZE]
        objective_names = ["kld_score", "refusals_score"]

    study = optuna.create_study(
        sampler=TPESampler(
            n_startup_trials=settings.n_startup_trials,
            n_ei_candidates=128,
            multivariate=True,
        ),
        directions=directions,
        storage=storage,
        study_name="heretic",
        load_if_exists=True,
    )

    # Resume validation: objective shape must match between checkpoint and config.
    stored_n_objectives = study.user_attrs.get("n_objectives")
    if stored_n_objectives is not None and stored_n_objectives != len(directions):
        print(
            f"[red]Error:[/] checkpoint has {stored_n_objectives} objectives but "
            f"current config expects {len(directions)}. Delete the checkpoint or "
            f"change thinking_eval_enabled to match."
        )
        sys.exit(1)

    study.set_user_attr("settings", settings.model_dump_json())
    study.set_user_attr("finished", False)
    study.set_user_attr("n_objectives", len(directions))
    study.set_user_attr("objective_names", objective_names)
    study.set_user_attr("thinking_eval_active", thinking_eval_active)

    # Persist per-study prompt subset for reproducibility (B1).
    if thinking_eval_active and thinking_eval_seed is not None:
        # Use stored seed if resuming an existing study.
        stored_seed = study.user_attrs.get("thinking_eval_seed")
        if stored_seed is not None and stored_seed != thinking_eval_seed:
            # Re-derive subset from the stored seed for consistency.
            rng = random.Random(stored_seed)
            indices = list(range(len(evaluator.thinking_prompts)))
            rng.shuffle(indices)
            n_samples = min(settings.thinking_eval_samples, len(indices))
            selected_indices = sorted(indices[:n_samples])
            trial_thinking_prompts = [
                evaluator.thinking_prompts[i] for i in selected_indices
            ]
            thinking_eval_seed = stored_seed
            thinking_eval_indices = selected_indices

        study.set_user_attr("thinking_eval_seed", thinking_eval_seed)
        study.set_user_attr("thinking_eval_prompt_indices", thinking_eval_indices)

    def count_completed_trials() -> int:
        # Count number of complete trials to compute trials to run.
        return sum([(1 if t.state == TrialState.COMPLETE else 0) for t in study.trials])

    start_index = trial_index = count_completed_trials()
    start_time = time.perf_counter()
    if start_index > 0:
        print()
        print("Resuming existing study.")

    # Pipelined ask/tell loop: trial N's LLM judge runs concurrently with
    # trial N+1's GPU work (reset + abliterate + generate + logprobs).
    # Tuple: (PendingScore, Trial, trial_index, trial_phase_times)
    pending: tuple[PendingScore, Trial, int, dict[str, float]] | None = None
    trial_phase_times: dict[str, float] = {}
    # Track the current trial separately so we can fail it on interrupt.
    current_trial: Trial | None = None

    def _fail_outstanding_trials() -> None:
        """Fail any trials left in RUNNING state after interruption or error."""
        nonlocal pending, current_trial
        if pending is not None:
            _, pending_trial, _, _ = pending
            try:
                resolve_pending(pending, timeout=5.0)
            except Exception:
                study.tell(pending_trial, state=TrialState.FAIL)
                logger.warning(
                    "Failed to resolve pending evaluation, marked trial as FAIL",
                    exc_info=True,
                )
            pending = None

        if current_trial is not None:
            study.tell(current_trial, state=TrialState.FAIL)
            current_trial = None

    def _run_trial_loop() -> None:
        """Execute pipelined ask/tell loop for remaining trials."""
        nonlocal pending, current_trial, trial_index, trial_phase_times
        pending = None
        current_trial = None
        try:
            n_remaining = settings.n_trials - count_completed_trials()
            for _ in range(n_remaining):
                trial_phase_times = {}
                current_trial = study.ask()
                trial_index += 1
                current_trial.set_user_attr("index", trial_index)

                suggest_and_abliterate(current_trial, trial_index)

                print("* Evaluating...")
                new_pending = evaluator.start_evaluation(
                    thinking_prompts=trial_thinking_prompts or None,
                )

                # Resolve PREVIOUS trial's LLM judge (ran during this trial's GPU work).
                resolve_pending(pending)

                pending = (new_pending, current_trial, trial_index, trial_phase_times)
                current_trial = None  # Now tracked via pending.

            # Flush last trial.
            resolve_pending(pending)
            pending = None

        except KeyboardInterrupt:
            _fail_outstanding_trials()
        except Exception:
            _fail_outstanding_trials()
            raise

    _run_trial_loop()

    if count_completed_trials() == settings.n_trials:
        study.set_user_attr("finished", True)

    while True:
        # If no trials at all have been evaluated, the study must have been stopped
        # by pressing Ctrl+C while the first trial was running. In this case, we just
        # re-raise the interrupt to invoke the standard handler defined below.
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed_trials:
            raise KeyboardInterrupt

        # Generic Pareto extraction using raw trial metrics.
        # A trial is Pareto-optimal if no other trial is better on ALL metrics.
        def _get_metric_vector(trial: Trial) -> list[float]:
            """Return minimization metrics for Pareto dominance checking."""
            metrics = [
                trial.user_attrs["refusals"],
                trial.user_attrs["kl_divergence"],
            ]
            rate = trial.user_attrs.get("thinking_completion_rate")
            if rate is not None:
                # Minimize incompletion (1 - rate).
                metrics.append(1.0 - rate)
            return metrics

        def _dominates(a: list[float], b: list[float]) -> bool:
            """Return True if a dominates b (a <= b on all, a < b on at least one)."""
            at_least_one_strict = False
            for ai, bi in zip(a, b):
                if ai > bi:
                    return False
                if ai < bi:
                    at_least_one_strict = True
            return at_least_one_strict

        metric_vectors = [_get_metric_vector(t) for t in completed_trials]
        best_trials = []
        for i, trial in enumerate(completed_trials):
            dominated = False
            for j, other in enumerate(completed_trials):
                if i != j and _dominates(metric_vectors[j], metric_vectors[i]):
                    dominated = True
                    break
            if not dominated:
                best_trials.append(trial)

        # Sort for display: refusals ascending, then KL ascending.
        best_trials.sort(
            key=lambda t: (
                t.user_attrs["refusals"],
                t.user_attrs["kl_divergence"],
            )
        )

        # Post-optimization stress test: run full thinking eval on top-K Pareto trials.
        if thinking_eval_active and evaluator.thinking_prompts and best_trials:
            # Only stress-test the top-K candidates to limit GPU cost.
            top_k = min(5, len(best_trials))
            stress_candidates = best_trials[:top_k]

            # Skip trials that already have stress-test results (from prior runs).
            untested = [
                t
                for t in stress_candidates
                if "thinking_stress_completion_rate" not in t.user_attrs
            ]

            if untested:
                print()
                print(
                    f"Running thinking stress test on {len(untested)} "
                    f"Pareto-optimal trial(s)..."
                )
                for trial in untested:
                    idx = trial.user_attrs["index"]
                    print(f"  * Rebuilding trial {idx}...")
                    model.reset_model()
                    model.abliterate(
                        refusal_directions,
                        trial.user_attrs["direction_index"],
                        {
                            k: AbliterationParameters(**v)
                            for k, v in trial.user_attrs["parameters"].items()
                        },
                    )
                    rate, failures, total = evaluator.evaluate_thinking(
                        evaluator.thinking_prompts
                    )
                    # FrozenTrial.set_user_attr() only modifies the in-memory
                    # copy; persist via storage so results survive across runs.
                    trial.set_user_attr("thinking_stress_completion_rate", rate)
                    trial.set_user_attr("thinking_stress_failures", failures)
                    trial.set_user_attr("thinking_stress_samples", total)
                    storage = study._storage
                    storage.set_trial_user_attr(
                        trial._trial_id, "thinking_stress_completion_rate", rate
                    )
                    storage.set_trial_user_attr(
                        trial._trial_id, "thinking_stress_failures", failures
                    )
                    storage.set_trial_user_attr(
                        trial._trial_id, "thinking_stress_samples", total
                    )
                    completed = total - failures
                    pct = rate * 100
                    print(f"  * Trial {idx}: {completed}/{total} complete ({pct:.1f}%)")

        def _trial_label(trial: Trial) -> str:
            label = (
                f"[Trial {trial.user_attrs['index']:>3}] "
                f"Refusals: {trial.user_attrs['refusals']:>2}/{len(evaluator.bad_prompts)}, "
                f"KL divergence: {trial.user_attrs['kl_divergence']:.4f}"
            )
            rate = trial.user_attrs.get("thinking_completion_rate")
            if rate is not None:
                label += f", Thinking: {rate * 100:.0f}%"
            stress_rate = trial.user_attrs.get("thinking_stress_completion_rate")
            if stress_rate is not None:
                label += f" (stress: {stress_rate * 100:.0f}%)"
            return label

        choices = [
            Choice(title=_trial_label(trial), value=trial) for trial in best_trials
        ]

        choices.append(
            Choice(
                title="Run additional trials",
                value="continue",
            )
        )

        choices.append(
            Choice(
                title="Exit program",
                value="",
            )
        )

        print()
        print("[bold green]Optimization finished![/]")
        print()
        print(
            (
                "The following trials resulted in Pareto optimal combinations of refusals and KL divergence. "
                "After selecting a trial, you will be able to save the model, upload it to Hugging Face, "
                "or chat with it to test how well it works. You can return to this menu later to select a different trial. "
                "[yellow]Note that KL divergence values above 1 usually indicate significant damage to the original model's capabilities.[/]"
            )
        )

        # Headless mode: auto-select trial, save, and exit.
        if settings.headless:
            # Select trial.
            if settings.trial is not None:
                candidates = [
                    t for t in best_trials if t.user_attrs["index"] == settings.trial
                ]
                if not candidates:
                    print(
                        f"[red]Trial {settings.trial} not found among Pareto-optimal trials.[/]"
                    )
                    sys.exit(1)
                selected = candidates[0]
            else:
                # Rank: refusals asc, thinking_stress_completion desc (or
                # thinking_completion desc), KL asc.
                def _headless_sort_key(t: Trial) -> tuple[float, ...]:
                    stress = t.user_attrs.get("thinking_stress_completion_rate")
                    rate = t.user_attrs.get("thinking_completion_rate")
                    thinking_key = -(stress if stress is not None else (rate or 0.0))
                    return (
                        t.user_attrs["refusals"],
                        thinking_key,
                        t.user_attrs["kl_divergence"],
                    )

                selected = min(best_trials, key=_headless_sort_key)

            print()
            headless_msg = (
                f"[bold green]Headless mode:[/] auto-selected trial "
                f"[bold]{selected.user_attrs['index']}[/] "
                f"(refusals: {selected.user_attrs['refusals']}/{len(evaluator.bad_prompts)}, "
                f"KL divergence: {selected.user_attrs['kl_divergence']:.4f}"
            )
            rate = selected.user_attrs.get("thinking_completion_rate")
            if rate is not None:
                headless_msg += f", thinking: {rate * 100:.0f}%"
            headless_msg += ")"
            print(headless_msg)

            # Restore model.
            print("* Resetting model...")
            model.reset_model()
            print("* Abliterating...")
            model.abliterate(
                refusal_directions,
                selected.user_attrs["direction_index"],
                {
                    k: AbliterationParameters(**v)
                    for k, v in selected.user_attrs["parameters"].items()
                },
            )

            # Determine save path and merge strategy.
            save_directory = settings.output_dir or "./output"
            Path(save_directory).mkdir(parents=True, exist_ok=True)

            if settings.quantization == QuantizationMethod.BNB_4BIT:
                print("Saving LoRA adapter (quantized model)...")
                model.model.save_pretrained(save_directory)
                model.tokenizer.save_pretrained(save_directory)
                print(
                    "[yellow]Note: Quantized model saved as LoRA adapter only. "
                    "Merge with base model before deployment.[/]"
                )
            else:
                print("Saving merged model...")
                merged_model = model.get_merged_model()
                merged_model.save_pretrained(save_directory)
                del merged_model
                empty_cache()
                model.tokenizer.save_pretrained(save_directory)

            print(f"[bold green]Model saved to {save_directory}[/]")
            return

        while True:
            print()
            trial = prompt_select("Which trial do you want to use?", choices)

            if trial == "continue":
                while True:
                    try:
                        n_additional_trials = prompt_text(
                            "How many additional trials do you want to run?"
                        )
                        if n_additional_trials is None or n_additional_trials == "":
                            n_additional_trials = 0
                            break
                        n_additional_trials = int(n_additional_trials)
                        if n_additional_trials > 0:
                            break
                        print("[red]Please enter a number greater than 0.[/]")
                    except ValueError:
                        print("[red]Please enter a number.[/]")

                if n_additional_trials == 0:
                    continue

                settings.n_trials += n_additional_trials
                study.set_user_attr("settings", settings.model_dump_json())
                study.set_user_attr("finished", False)

                _run_trial_loop()

                if count_completed_trials() == settings.n_trials:
                    study.set_user_attr("finished", True)

                break

            elif trial is None or trial == "":
                return

            print()
            print(f"Restoring model from trial [bold]{trial.user_attrs['index']}[/]...")
            print("* Parameters:")
            for name, value in get_trial_parameters(trial).items():
                print(f"  * {name} = [bold]{value}[/]")
            print("* Resetting model...")
            model.reset_model()
            print("* Abliterating...")
            model.abliterate(
                refusal_directions,
                trial.user_attrs["direction_index"],
                {
                    k: AbliterationParameters(**v)
                    for k, v in trial.user_attrs["parameters"].items()
                },
            )

            while True:
                print()
                action = prompt_select(
                    "What do you want to do with the decensored model?",
                    [
                        "Save the model to a local folder",
                        "Upload the model to Hugging Face",
                        "Chat with the model",
                        "Benchmark the model",
                        "Return to the trial selection menu",
                    ],
                )

                if action is None or action == "Return to the trial selection menu":
                    break

                # All actions are wrapped in a try/except block so that if an error occurs,
                # another action can be tried, instead of the program crashing and losing
                # the optimized model.
                try:
                    match action:
                        case "Save the model to a local folder":
                            save_directory = prompt_path("Path to the folder:")
                            if not save_directory:
                                continue

                            strategy = obtain_merge_strategy(settings)
                            if strategy is None:
                                continue

                            if strategy == "adapter":
                                print("Saving LoRA adapter...")
                                model.model.save_pretrained(save_directory)
                            else:
                                print("Saving merged model...")
                                merged_model = model.get_merged_model()
                                merged_model.save_pretrained(save_directory)
                                del merged_model
                                empty_cache()
                                model.tokenizer.save_pretrained(save_directory)

                            print(f"Model saved to [bold]{save_directory}[/].")

                        case "Upload the model to Hugging Face":
                            # We don't use huggingface_hub.login() because that stores the token on disk,
                            # and since this program will often be run on rented or shared GPU servers,
                            # it's better to not persist credentials.
                            token = huggingface_hub.get_token()
                            if not token:
                                token = prompt_password("Hugging Face access token:")
                            if not token:
                                continue

                            user = huggingface_hub.whoami(token)
                            fullname = user.get(
                                "fullname",
                                user.get("name", "unknown user"),
                            )
                            email = user.get("email", "no email found")
                            print(f"Logged in as [bold]{fullname} ({email})[/]")

                            repo_id = prompt_text(
                                "Name of repository:",
                                default=f"{user['name']}/{Path(settings.model).name}-heretic",
                            )

                            visibility = prompt_select(
                                "Should the repository be public or private?",
                                [
                                    "Public",
                                    "Private",
                                ],
                            )
                            if visibility is None:
                                continue
                            private = visibility == "Private"

                            strategy = obtain_merge_strategy(settings)
                            if strategy is None:
                                continue

                            if strategy == "adapter":
                                print("Uploading LoRA adapter...")
                                model.model.push_to_hub(
                                    repo_id,
                                    private=private,
                                    token=token,
                                )
                            else:
                                print("Uploading merged model...")
                                merged_model = model.get_merged_model()
                                merged_model.push_to_hub(
                                    repo_id,
                                    private=private,
                                    token=token,
                                )
                                del merged_model
                                empty_cache()
                                model.tokenizer.push_to_hub(
                                    repo_id,
                                    private=private,
                                    token=token,
                                )

                            # If the model path exists locally and includes the
                            # card, use it directly. If the model path doesn't
                            # exist locally, it can be assumed to be a model
                            # hosted on the Hugging Face Hub, in which case
                            # we can retrieve the model card.
                            model_path = Path(settings.model)
                            if model_path.exists():
                                card_path = (
                                    model_path / huggingface_hub.constants.REPOCARD_NAME
                                )
                                if card_path.exists():
                                    card = ModelCard.load(card_path)
                                else:
                                    card = None
                            else:
                                card = ModelCard.load(settings.model)
                            if card is not None:
                                if card.data is None:
                                    card.data = ModelCardData()
                                if card.data.tags is None:
                                    card.data.tags = []
                                card.data.tags.append("heretic")
                                card.data.tags.append("uncensored")
                                card.data.tags.append("decensored")
                                card.data.tags.append("abliterated")
                                card.text = (
                                    get_readme_intro(
                                        settings,
                                        trial,
                                        evaluator.base_refusals,
                                        evaluator.bad_prompts,
                                    )
                                    + card.text
                                )
                                card.push_to_hub(repo_id, token=token)

                            print(f"Model uploaded to [bold]{repo_id}[/].")

                        case "Chat with the model":
                            print()
                            print(
                                "[cyan]Press Ctrl+C at any time to return to the menu.[/]"
                            )

                            chat = [
                                {"role": "system", "content": settings.system_prompt},
                            ]

                            while True:
                                try:
                                    message = prompt_text(
                                        "User:",
                                        qmark=">",
                                        unsafe=True,
                                    )
                                    if not message:
                                        break
                                    chat.append({"role": "user", "content": message})

                                    print("[bold]Assistant:[/] ", end="")
                                    response = model.stream_chat_response(chat)
                                    chat.append(
                                        {"role": "assistant", "content": response}
                                    )
                                except (KeyboardInterrupt, EOFError):
                                    # Ctrl+C/Ctrl+D
                                    break

                        case "Benchmark the model":
                            benchmarks = questionary.checkbox(
                                "Which benchmarks do you want to run?",
                                [
                                    Choice(
                                        title=f"{benchmark.name}: {benchmark.description}",
                                        value=benchmark,
                                    )
                                    for benchmark in settings.benchmarks
                                ],
                                style=Style([("highlighted", "reverse")]),
                            ).ask()
                            if not benchmarks:
                                continue

                            scope = prompt_select(
                                (
                                    "Do you want to benchmark the original model along with the decensored model? "
                                    "Benchmarking both models allows you to compare the scores, but it takes twice as much time."
                                ),
                                [
                                    "Benchmark only the decensored model",
                                    "Benchmark both models",
                                ],
                            )
                            if scope is None:
                                continue
                            benchmark_original_model = scope == "Benchmark both models"

                            hflm = HFLM(
                                pretrained=model.model,  # ty:ignore[invalid-argument-type]
                                tokenizer=model.tokenizer,  # ty:ignore[invalid-argument-type]
                            )

                            table = Table()
                            table.add_column("Benchmark")
                            table.add_column("Metric")
                            if benchmark_original_model:
                                table.add_column("This model", justify="right")
                                table.add_column("Original model", justify="right")
                            else:
                                table.add_column("Value", justify="right")

                            try:
                                first_benchmark = True

                                for benchmark in benchmarks:
                                    print(
                                        f"Running benchmark [bold]{benchmark.name}[/]..."
                                    )

                                    def get_results() -> dict[str, Any]:
                                        results = lm_eval.simple_evaluate(
                                            model=hflm,
                                            tasks=[benchmark.task],
                                            batch_size="auto",
                                        )
                                        return results["results"][benchmark.task]

                                    results = get_results()
                                    if benchmark_original_model:
                                        with model.model.disable_adapter():  # ty:ignore[call-non-callable]
                                            original_results = get_results()

                                    first_row = True

                                    for metric, value in results.items():
                                        if metric != "alias":
                                            if first_row and not first_benchmark:
                                                if benchmark_original_model:
                                                    table.add_row("", "", "", "")
                                                else:
                                                    table.add_row("", "", "")

                                            def format_value(value: Any) -> str:
                                                if isinstance(
                                                    value,
                                                    (float, np.floating),
                                                ):
                                                    return f"{value:.4f}"
                                                else:
                                                    return f"{value}"

                                            cells = [
                                                benchmark.name if first_row else "",
                                                metric,
                                                format_value(value),
                                            ]
                                            if benchmark_original_model:
                                                cells.append(
                                                    format_value(
                                                        original_results[metric]
                                                    )
                                                )
                                            table.add_row(*cells)

                                            first_row = False
                                            first_benchmark = False
                            except KeyboardInterrupt:
                                pass

                            # The benchmark run might have been cancelled by the user
                            # before any benchmark was completed, so we only print results
                            # if there actually are some.
                            if table.rows:
                                print(table)

                except Exception as error:
                    print(f"[red]Error: {error}[/]")


def main():
    # Install Rich traceback handler.
    install()

    try:
        run()
    except BaseException as error:
        # Transformers appears to handle KeyboardInterrupt (or BaseException)
        # internally in some places, which can re-raise a different error in the handler,
        # masking the root cause. We therefore check both the error itself and its context.
        if isinstance(error, KeyboardInterrupt) or isinstance(
            error.__context__, KeyboardInterrupt
        ):
            print()
            print("[red]Shutting down...[/]")
        else:
            raise
