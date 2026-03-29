# Thinking-Aware Trial Evaluation for Heretic

## Design Overview

Heretic currently suppresses thinking prefixes during evaluation, so Optuna never sees damage to the model's ability to close a thinking chain. This feature adds a low-cost secondary evaluation pass that measures thinking chain completion without changing the existing refusal-removal objective.

Key design decisions:

- Model thinking integrity as a third minimized objective when active: `thinking_incompletion_score = 1.0 - thinking_completion_rate`. Refusal count and KL divergence stay unchanged so thinking preservation does not get hidden inside a weighted penalty.
- Reuse the existing `Model.get_responses_batched()` path by running a secondary pass with thinking enabled and `skip_special_tokens=False`. The standard refusal pass keeps the current suppressed-prefix behavior.
- Separate per-trial guidance from final validation. Each trial uses only the first `thinking_eval_samples` prompts from a dedicated thinking dataset, while the full dataset is reused for a post-optimization stress test on Pareto-optimal trials.
- Detect support through a reusable `ThinkingProfile` derived from prefix detection. Unsupported or non-thinking models stay on the existing two-objective path.
- Replace the current hand-written two-dimensional Pareto logic with a generic raw-metric Pareto helper so headless and interactive flows can work with either two or three metrics.

Architecture constraints discovered during recon:

- `src/heretic/evaluator.py` currently returns only `(kld_score, refusals_score)`, so adding a third signal requires changing both `PendingScore.resolve()` and the `main.py` trial bookkeeping.
- `src/heretic/main.py` hardcodes a two-dimensional Pareto sweep and headless auto-selection order.
- `src/heretic/model.py` has a single mutable `response_prefix`, so thinking evaluation must restore state cleanly after temporarily enabling thinking.
- Study checkpoints are keyed only by model path, which means resumed runs must validate that the stored objective shape matches the active configuration.

## Proposed Data Model

### `config.py`

Add the following settings:

- `thinking_eval_enabled: bool = False`
- `thinking_eval_prompts: DatasetSpecification | None = None`
- `thinking_eval_samples: int = 10`

Validation rules:

- `thinking_eval_samples` must be greater than zero.
- `thinking_eval_prompts` is required only when `thinking_eval_enabled` is `true`.
- When `thinking_eval_enabled` is `false`, missing `thinking_eval_prompts` is valid and should not trigger warnings.

Rationale:

- The feature must be opt-in because it adds extra GPU work.
- A dedicated dataset is necessary because the refusal dataset is the wrong workload for chain-integrity measurement.
- The prompt specification should remain a normal `DatasetSpecification` so it reuses existing prompt loading and CLI/env/config layering.

### `model.py`

Introduce a small profile object instead of scattering hardcoded prefix logic across `main.py` and `evaluator.py`.

```python
@dataclass(frozen=True)
class ThinkingProfile:
    name: str
    opening_marker: str
    completion_marker: str
    suppressed_prefix: str
```

Add `model.thinking_profile: ThinkingProfile | None` and keep `model.response_prefix` as the active evaluation prefix.

Rationale:

- The existing prefix-detection code already knows which thinking syntax is present.
- The evaluator needs a stable completion marker but should not duplicate string matching rules.
- This keeps the public generation API unchanged and pulls syntax-specific complexity behind one small object.

### `evaluator.py`

Replace the loose tuple return path with an explicit evaluation payload.

```python
@dataclass(frozen=True)
class TrialEvaluation:
    objectives: tuple[float, ...]
    kl_divergence: float
    refusals: int
    thinking_completion_rate: float | None = None
    thinking_failures: int | None = None
    thinking_samples: int = 0
```

`PendingScore.resolve()` should return `TrialEvaluation` instead of raw tuples.

Rationale:

- Objective shape becomes dynamic: two objectives when thinking is inactive, three when active.
- `main.py` needs both the objective tuple for `study.tell()` and the raw metrics for `trial.user_attrs`.
- A named payload prevents tuple-position bugs when the feature is disabled or skipped.

## Prefix Detection and Thinking Support

Refactor the current prefix suppression block in `main.py` into a helper:

```python
def detect_thinking_profile(prefix: str) -> ThinkingProfile | None:
    ...
```

Supported profiles in v1:

- `<think>` ... `</think>`
- `<|channel|>analysis<|message|>` ... `<|end|><|start|>assistant<|channel|>final<|message|>`
- `<thought>` ... `</thought>`
- `[THINK]` ... `[/THINK]`

Behavior:

1. Run the existing common-prefix detection.
2. If a known thinking profile matches, store it on the model and set `model.response_prefix = profile.suppressed_prefix`.
3. If no known thinking profile matches, leave `model.thinking_profile = None` and keep current behavior.
4. If thinking evaluation was requested but no profile was detected, print a single warning and continue in two-objective mode.

Important design choice:

- The thinking pass should temporarily set `model.response_prefix = ""` instead of injecting the raw thinking prefix. This measures real end-to-end inference behavior from a clean prompt, avoids storing two synchronized prefixes, and keeps the extra logic small.

## Trial Evaluation Flow

### Standard refusal and KL path

This stays structurally identical:

1. Generate refusal-evaluation responses with the suppressed prefix.
2. Submit the LLM judge future, if enabled.
3. Compute KL divergence on good prompts.

### Secondary thinking pass

When all of the following are true, run a second GPU pass before returning `PendingScore`:

- `settings.thinking_eval_enabled`
- `settings.thinking_eval_prompts` is configured
- `model.thinking_profile is not None`
- At least one thinking prompt was loaded

The pass should:

1. Select `trial_thinking_prompts = thinking_prompts[:thinking_eval_samples]`.
2. Temporarily clear `model.response_prefix`.
3. Call `model.get_responses_batched(trial_thinking_prompts, skip_special_tokens=False)`.
4. Restore `model.response_prefix` in a `try/finally` guarded helper or context manager.
5. Compute `thinking_completion_rate = completed / len(trial_thinking_prompts)`.

Completion rule:

- Success: the generated response contains `thinking_profile.completion_marker`, and the completion marker appears after the opening marker when both are present in the decoded text.
- Failure: the completion marker is missing, appears before the opening marker, or the response is empty.

Notes:

- `skip_special_tokens=False` is required because some supported thinking syntaxes use tokenizer-level sentinel strings.
- The metric intentionally measures closure, not semantic reasoning quality. It is a structure-preservation signal, not a correctness benchmark.

### Objective mapping

Raw metrics:

- `kl_divergence`
- `refusals`
- `thinking_completion_rate`

Optimization objectives:

- Objective 1: existing `kld_score`
- Objective 2: existing `refusals_score`
- Objective 3: `thinking_incompletion_score = 1.0 - thinking_completion_rate`

Why a third objective instead of a penalty term:

- It keeps censorship removal visible and first-class instead of burying it behind a weight.
- It avoids inventing a hard-coded exchange rate between thinking damage and refusal removal.
- It fits Optuna's existing multi-objective study shape with fewer implicit assumptions.

## Pipelining and Performance

The current pipeline overlaps trial `N`'s LLM-judge wait with trial `N+1`'s GPU work. The new thinking pass keeps that architecture intact:

- The LLM judge still runs in the background thread after the refusal responses are generated.
- KL divergence and the thinking pass remain GPU-bound and execute serially inside `Evaluator.start_evaluation()`.
- Trial `N`'s judge future is still resolved only after trial `N+1` has already started its GPU work.

Expected cost profile:

- Per-trial GPU time increases by one small generation batch.
- LLM-judge overlap is preserved.
- Prompt count stays bounded by `thinking_eval_samples` to avoid meaningfully slowing optimization.

Non-goal:

- The feature does not attempt to hide the thinking pass behind a second GPU worker. That would compete with the next trial's ablation and generation steps on the same device.

## Post-Optimization Stress Test

After the optimization loop completes and before presenting Pareto-optimal trials, run a one-time stress test for each Pareto-optimal trial when thinking evaluation is active.

Stress-test plan:

1. Rebuild each Pareto-optimal trial as the code already does for save/chat/benchmark actions.
2. Run the same thinking-evaluation helper on the full `thinking_eval_prompts` dataset, not just the first `thinking_eval_samples` prompts.
3. Store:
   - `thinking_stress_completion_rate`
   - `thinking_stress_failures`
   - `thinking_stress_samples`
4. Display those values in the trial picker and headless summary.

Rationale:

- The small per-trial sample is good enough to guide Optuna.
- The full-dataset pass is better for final acceptance or rejection decisions.
- This avoids adding a second configuration knob while still giving the final selection step a stronger signal.

## Optuna and Trial Selection Changes

### Study creation

Create the study with:

- `directions=[MINIMIZE, MINIMIZE]` when thinking evaluation is inactive
- `directions=[MINIMIZE, MINIMIZE, MINIMIZE]` when thinking evaluation is active

Persist study metadata:

- `thinking_eval_active`
- `objective_names`

### Resume validation

Because checkpoints are keyed only by model path, a resumed study must validate:

- stored `thinking_eval_active` matches the active run
- stored number of objectives matches the evaluator

If not, abort continuation with a clear message instructing the user to restart the study or use a different checkpoint directory. Silent reuse would mix incomparable trials.

### Generic Pareto extraction

Replace the current sorted two-dimensional sweep with a helper that uses raw trial attributes:

- always minimize `refusals`
- always minimize `kl_divergence`
- when active, minimize `thinking_incompletion_rate = 1 - thinking_completion_rate`

This keeps the selection menu honest because the displayed Pareto front is based on the real metrics, not the shaped optimization scores.

### Headless auto-selection

When thinking stress-test data exists, rank candidates by:

1. `refusals` ascending
2. `thinking_stress_completion_rate` descending
3. `kl_divergence` ascending

Fallback when stress-test data is unavailable:

1. `refusals` ascending
2. `thinking_completion_rate` descending
3. `kl_divergence` ascending

This preserves censorship removal as the primary filter while still preferring trials that keep thinking intact.

## User-Facing Output

Add concise output lines during evaluation:

- `Thinking evaluation: 8/10 complete (80.0%)`
- `Thinking evaluation skipped: no supported thinking prefix detected`

Add trial metadata:

- `trial.user_attrs["thinking_completion_rate"]`
- `trial.user_attrs["thinking_failures"]`
- `trial.user_attrs["thinking_samples"]`
- stress-test counterparts when available

Update the Pareto menu text to include thinking metrics whenever they exist.

## Threat Model

- [ ] [Security] Bound the extra workload so user-supplied datasets cannot accidentally multiply per-trial GPU time. Use only the first `thinking_eval_samples` prompts inside the optimization loop and report the effective sample count.
  Dependencies: `Settings` validation and evaluator prompt loading must both be in place before the limit is enforceable.
  DONE WHEN: A run with a 1,000-prompt thinking dataset still evaluates only `thinking_eval_samples` prompts per trial and prints the bounded count.

- [ ] [Security] Prevent stale checkpoint reuse across objective shapes. Mixing two-objective and three-objective trials inside the same study would silently corrupt Pareto selection.
  Dependencies: Study metadata must be stored at creation time and compared before checkpoint continuation restores settings.
  DONE WHEN: Resuming a legacy two-objective checkpoint with `thinking_eval_enabled=true` exits with a clear mismatch message instead of continuing.

- [ ] [Security] Prevent response-prefix state leakage after the thinking pass. A failed restore would contaminate later refusal evaluation, chat actions, and saved models.
  Dependencies: The thinking pass helper must own prefix swapping instead of open-coding mutations in multiple places.
  DONE WHEN: An injected exception inside the thinking pass leaves `model.response_prefix` unchanged after the helper returns.

## Debate Findings

### Round 1

Adopted:

- Use a third objective instead of a weighted penalty. This keeps censorship removal measurable and avoids an arbitrary penalty coefficient that would be hard to justify. If TPE convergence proves unstable with the coarse third metric, the next step is switching to constraint/feasibility, not a penalty.
- Use a small prompt subset during optimization and the full prompt pool for post-optimization stress testing. This is the best tradeoff between optimizer guidance and runtime cost.
- Introduce `ThinkingProfile` rather than special-casing only `<think>`. The repository already recognizes multiple thinking syntaxes, and the evaluation feature should not regress that support surface. Keep it minimal (frozen dataclass, no strategy classes or plugin systems).
- Replace tuple-based evaluation results with `TrialEvaluation` (frozen dataclass). Low-cost, high-reward refactor that prevents tuple-position bugs as objective count becomes dynamic.
- Checkpoint validation must go beyond objective shape: store schema version, `objective_names`, and `thinking_eval_active` in study metadata. Legacy missing-key handling and non-interactive fail-fast are required.
- Baseline sanity check: before the optimization loop, run thinking evaluation on the unablated model. If the baseline completion rate is abnormal, skip thinking evaluation entirely and warn.
- Stress test cap: do not stress-test all Pareto-optimal trials. Only test top-K candidates (ranked by refusals, then thinking completion rate) or user-selected trials.
- Any thinking-preservation approach that risks re-introducing refusal behavior (e.g., repair adapter merge) must be opt-in and off by default. Users can always use external CoT simulation as a zero-risk fallback.

### Round 2

Adopted:

- **Sample count (A1)**: Keep 10 samples per trial for v1. The 0.1 granularity (11 buckets) is sufficient for a coarse guardrail objective. MOTPE can still rank trials because the other two objectives (KL, refusals) provide continuous signal. Escalate to 16-20 samples only if a concrete symptom appears: many Pareto-front trials cluster at 0.8/0.9/1.0 and the third objective becomes the ranking bottleneck.
- **Constraint approach rejected (A3)**: Switching to Optuna constraint/feasibility was rejected for v1. Reasons: (1) current Pareto extraction logic explicitly does not handle constraints, (2) threshold is brittle at 10 samples (4/10 vs 5/10 is one prompt's difference), (3) discards ordinal information (0.49 and 0.0 both become infeasible).
- **Per-study fixed subset (B1)**: Seed-shuffle the thinking prompt dataset once at study creation, then use the same subset for all trials. This gives TPE the most stable, comparable signal because generation is greedy/deterministic. Store the seed and prompt indices in study attrs for reproducibility. Do NOT randomize per-trial (B2), as this injects observation noise into an already coarse metric.
- **Implementation detail**: store `thinking_eval_seed` and `thinking_eval_prompt_indices` in `study.user_attrs` alongside the seed for reproducibility.

Deferred:

- A dedicated `thinking_eval_max_response_length` setting is deferred. V1 reuses `max_response_length`; the documentation should tell users to raise it if their thinking prompts need longer chains.
- Semantic reasoning-quality scoring is deferred. The feature only protects structural chain completion, not answer correctness, because correctness evaluation would materially increase cost and design scope.
- Support for unknown thinking syntaxes is deferred. V1 activates only for profiles the existing prefix-detection logic can recognize reliably.
- All research-level protection approaches (null-space projection, equality-constrained editing, SAE feature-level protection, activation patching masks, repair adapter merge) are deferred to v1.1+. The evaluation feature must land first to provide metrics for measuring whether these techniques actually help. Null-space projection is the best candidate for v1.1 as it is closest to the existing `orthogonalize_direction` mechanism.
- Token-ID level completion detection is deferred. V1 uses string matching on decoded text with `skip_special_tokens=False`. Long-term, token-ID subsequence matching would be more robust across tokenizer architectures.

## Implementation Plan

- [ ] [Main] Add `ThinkingProfile` detection and configuration plumbing across `main.py`, `model.py`, and `config.py`.
  Context: The evaluator cannot decide whether to run a thinking pass unless the prefix-detection result and the dedicated prompt dataset configuration are both available through stable interfaces.
  Dependencies: None.
  DONE WHEN: `Settings` exposes the new fields, prefix detection returns `ThinkingProfile | None`, and non-thinking models still behave exactly as before when the feature is disabled.

- [ ] [Main] Replace tuple-based evaluation results with `TrialEvaluation` and add the secondary thinking pass in `evaluator.py`.
  Context: Dynamic objective counts and raw metric reporting are awkward and fragile with positional tuples. A named result object keeps the evaluator understandable and makes the thinking pass easy to test in isolation.
  Dependencies: The configuration plumbing and thinking-profile detection must exist first.
  DONE WHEN: `Evaluator.start_evaluation()` can return a two-objective or three-objective `TrialEvaluation`, and thinking metrics are populated only when the feature is active.

- [ ] [Main] Generalize study creation, resume validation, trial bookkeeping, and Pareto extraction in `main.py`.
  Context: The optimization loop, checkpoint continuation path, and headless selection currently assume exactly two objectives and exactly two raw metrics.
  Dependencies: `TrialEvaluation` must exist so `main.py` can store raw metrics without tuple unpacking ambiguity.
  DONE WHEN: Two-objective studies still run unchanged, three-objective studies persist thinking metrics, and objective-shape mismatches abort resume safely.

- [ ] [Main] Add the post-optimization thinking stress test and surface the results in interactive and headless flows.
  Context: The per-trial sample is intentionally small, so final trial acceptance needs a stronger signal before save, upload, or deployment decisions.
  Dependencies: Generic Pareto extraction and the thinking evaluation helper must already work.
  DONE WHEN: Every Pareto-optimal trial receives stress-test attributes when thinking evaluation is active, and the selection UI displays them.

- [ ] [Main] Document the feature in `config.default.toml` and `docs/thinking-model-compatibility.md`.
  Context: Users need to know that the feature is opt-in, requires a dedicated reasoning prompt dataset, and may need a higher `max_response_length` or trial count for good results.
  Dependencies: Final field names and activation behavior must be stable.
  DONE WHEN: The default config includes the new fields with explanatory comments, and the compatibility doc no longer describes thinking-aware evaluation as only a planned idea.

## Acceptance Checks

1. With `thinking_eval_enabled = false`, the evaluator, study directions, checkpoint resume path, and final trial picker behave exactly like the current two-objective implementation.
2. With `thinking_eval_enabled = true` and a supported thinking profile, each completed trial stores `thinking_completion_rate`, `thinking_failures`, and `thinking_samples`.
3. With `thinking_eval_enabled = true` and no supported thinking profile, the run logs a skip message and stays on the two-objective path without crashing.
4. Post-optimization stress testing runs on top-K Pareto-optimal trials (not all) and stores `thinking_stress_completion_rate`, `thinking_stress_failures`, and `thinking_stress_samples`.
5. Resuming a checkpoint created under a different objective shape or schema version fails fast with a clear restart instruction.
6. The thinking evaluation prompt subset is seed-shuffled once per study, and the seed and prompt indices are stored in `study.user_attrs` for reproducibility.
7. A baseline sanity check on the unablated model runs before the optimization loop. If the baseline thinking completion rate is abnormal, thinking evaluation is skipped with a warning.
8. The thinking pass temporarily clears `model.response_prefix` and restores it in a `try/finally` guard. An exception during the thinking pass must not leak prefix state.
