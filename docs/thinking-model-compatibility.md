# Thinking Model Compatibility

Directional ablation can degrade a model's ability to properly close `<think>...</think>` blocks during inference. This document covers the known behavior, ablation-phase mitigations, and inference-phase strategies.

## Background

Heretic removes censorship via directional ablation: it identifies weight directions associated with refusal behavior and removes them. However, some of these directions overlap with the directions responsible for generating structural tokens like `</think>`. This manifests as:

- The model can still *enter* thinking mode (generate `<think>`)
- The model loses the ability to *exit* thinking mode (generate `</think>`) under certain conditions
- The effect is stochastic and prompt-dependent: the same trial may loop on one prompt and complete normally on another

### Measured impact (Qwen3.5-2B)

| Model | Think-loop rate | Source |
|-------|----------------|--------|
| Original (unmodified) | 0% (0/5) | A/B test |
| Heretic-ablated (trial 139/150) | 40% (2/5) | A/B test |
| Heretic + SimPO DPO | 20% (1/5) | A/B test |

Methodology: 5 runs per model, same Chinese roleplay prompt (~578 chars), 60s timeout, temperature=0.7 via Ollama. See [issue #7](https://github.com/RyderFreeman4Logos/heretic/issues/7) for full reproduction data.

### Why Heretic's optimizer is blind to this

During evaluation, Heretic suppresses CoT output by forcing the response prefix to `<think></think>` (see `main.py` prefix detection). All scoring (KL divergence, refusal count) operates on thinking-suppressed output. The optimizer therefore has **zero visibility** into thinking chain integrity and cannot penalize trials that damage `</think>` generation.

## Ablation Phase: What You Can Do

The mitigations below reduce think-loop probability **without weakening censorship removal**. They are ordered from safest/simplest to most experimental. None can guarantee zero think-loops, but they can meaningfully reduce the rate.

> **Note on conservative ablation parameters**: Raising `kl_divergence_target` or lowering `winsorization_quantile` will reduce thinking damage, but at the cost of weaker censorship removal. This is a direct tradeoff, not a free improvement — users who need strong censorship removal are better served by disabling native thinking and using external CoT orchestration instead. These parameters are therefore not listed as recommended mitigations.

### 1. Thinking-aware trial evaluation (recommended, planned)

Currently, Heretic suppresses CoT during evaluation (`main.py` forces `<think></think>` prefix), so the optimizer is completely blind to thinking damage. Adding thinking chain integrity as an evaluation signal lets Optuna actively avoid parameter regions that break `</think>` generation — without reducing ablation strength.

**How it works**:

- During each trial's evaluation, generate a small batch (5-10 prompts) of responses **with thinking enabled**
- Compute `thinking_completion_rate = (responses with proper </think> closure) / total`
- Use this as a penalty term or third optimization objective

**Effect**: Not all parameter combinations damage thinking equally. This guides Optuna toward trials that remove censorship effectively while preserving thinking chain integrity. Expected reduction: 40% → 10-15% loop rate.

**Current status**: Requires modifying `evaluator.py` to run a secondary evaluation pass with thinking enabled. Planned for implementation.

### 2. Increased trial count (safe, immediate)

With thinking-aware evaluation enabled, more trials give the optimizer a better chance of finding the sweet spot where censorship is removed and thinking is preserved. This is a simple amplifier for mitigation #1.

**Recommendation**: For thinking models, use 300+ trials (vs the default 200). Set `n_trials = 300` or higher in `config.toml`. See the comment in `config.default.toml` for details.

### 3. Selective layer/component ablation (experimental, needs research)

Thinking ability may be concentrated in specific transformer layers (typically middle layers handle reasoning/planning, while shallow/deep layers handle token generation). If we can identify which layers are critical for `</think>` generation, those layers can be excluded or attenuated during ablation.

**How to investigate**:

1. Per-layer ablation analysis: ablate one layer at a time and measure thinking completion rate
2. Identify "dangerous layers" where ablation disproportionately damages `</think>` generation
3. Reduce or skip ablation on those layers while maintaining full ablation elsewhere

**Effect**: Unknown until investigated, but if thinking is concentrated in a few layers, the benefit could be large.

### 4. Direction analysis (research, diagnostic only)

Compute the cosine similarity between refusal directions and `</think>`-generation directions. This tells you how much overlap exists:

- **Low overlap** (cosine < 0.3): Protection is feasible — you can project out the structural component before ablating
- **High overlap** (cosine > 0.7): Fundamental limitation — censorship removal and thinking integrity are in tension
- **Medium overlap**: Partial mitigation possible, results will vary

This is purely diagnostic and does not change any model weights. It helps set realistic expectations for what ablation-level fixes can achieve.

### 5. Post-ablation thinking stress test (safe, recommended)

After optimization, before accepting a trial for production use, run a dedicated thinking stress test:

1. Select 20-50 prompts known to trigger extended thinking chains (complex reasoning, roleplay, multi-step analysis)
2. Generate responses **with thinking enabled** (not suppressed)
3. Check each response for `<think>` without matching `</think>`
4. Record the think-loop rate for the trial

This does not change the ablation itself, but lets you **reject bad trials** before deployment. Trials with loop rate > N% (choose your threshold) should be discarded even if their KL/refusal scores look good.

**Implementation note**: This is currently manual. A future Heretic feature could automate this as a post-optimization validation step.

## Inference Phase: Deployment Strategies

These strategies operate at the inference/serving layer, independent of Heretic's ablation process.

### Strategy 1: Disable native thinking (safest)

Disable the model's native `<think>` mode entirely and simulate chain-of-thought through system-level orchestration:

```
# Instead of native thinking:
# <think>reasoning...</think>answer

# Use external orchestration:
# Request 1: "Analyze this problem step by step: {prompt}"
# → Get reasoning (discard or store as hidden context)
# Request 2: "Based on the analysis, provide your final answer"
# → Get clean response
```

**Pros**: Zero think-loop risk; reasoning quality controllable via prompt engineering
**Cons**: Higher latency (2x requests); higher token usage; reasoning quality may differ from native thinking

This is the only strategy that **guarantees** zero think-loops.

### Strategy 2: Streaming repetition detection + retry (recommended for native thinking)

Monitor the token stream for repetition patterns and kill generation when a loop is detected:

```
Detection algorithm (pseudocode):
  window = rolling buffer of last N tokens (e.g., N=200)
  for each new token:
    if window contains 3+ repetitions of any substring > 20 tokens:
      kill generation
      retry or fallback
```

The think-loop pattern is highly regular (same correction phrase repeated), making it easy to detect with simple substring matching.

| Metric | Value |
|--------|-------|
| Detection latency | ~3-5s after loop onset (2-3 repetitions needed) |
| Wasted tokens per detection | ~500-800 |
| False positive rate | Very low (legitimate reasoning rarely repeats 20+ token blocks 3 times) |

**On detection, the user/application can choose**:

1. **Retry with thinking**: Re-send the same prompt, hoping for a non-looping path (stochastic — may work)
2. **Fallback to non-thinking**: Re-send with thinking disabled (guaranteed success, lower reasoning quality)
3. **Retry with modified prompt**: Add slight perturbation to break the loop pattern

### Strategy 3: Token budget limit (simple fallback)

Set `num_predict` (Ollama) or equivalent `max_tokens` to a reasonable upper bound:

```
# Ollama Modelfile example
PARAMETER num_predict 4096
```

**Pros**: Simple, prevents infinite generation and runaway costs
**Cons**: Produces truncated, incomplete responses when a loop occurs — the user sees garbage

This should be used as a **safety net** alongside Strategy 2, not as the primary mitigation.

### Strategy 4: Combined (recommended for production)

```
Layer 1: Streaming repetition detection (primary)
  → On detect: retry (up to 2 attempts) or fallback to non-thinking
Layer 2: Token budget limit (safety net)
  → Catches any loop that evades repetition detection
Layer 3: Application-level timeout (last resort)
  → Catches any other degenerate generation behavior
```

### Retry budget and latency expectations

With local inference (user's own hardware), latency tolerance is typically high (minutes, not seconds). This changes the retry calculus significantly:

| Think-loop rate | Retries needed (99.9%) | Expected worst-case latency* |
|-----------------|------------------------|------------------------------|
| 5% | 2 | detect(5s) + retry(30s) = 35s |
| 10% | 3 | 2×detect(5s) + retry(30s) = 40s |
| 15% | 3-4 | 3×detect(5s) + retry(30s) = 45s |
| 20% | 4 | 3×detect(5s) + retry(30s) = 45s |
| 40% | 6+ | 5×detect(5s) + retry(30s) = 55s+ |

*Assumes streaming repetition detection (~5s per loop detection) and normal response time of ~30s. Actual times depend on hardware and prompt complexity.

## Recommendations by Use Case

| Use case | Recommended strategy |
|----------|---------------------|
| Production API serving | Disable native thinking + external CoT orchestration |
| Local interactive chat | Repetition detection + retry/fallback + token budget |
| Research / experimentation | Token budget only (accept occasional truncation) |
| Batch processing | Repetition detection + unlimited retries (latency irrelevant) |

## Current Status

- **Heretic core**: No thinking-aware evaluation yet. CoT is suppressed during optimization.
- **This fork**: Investigating thinking-aware trial evaluation and post-ablation stress testing.
- **Upstream (p-e-w/heretic)**: Classified as out-of-scope / fundamental limitation ([#253](https://github.com/p-e-w/heretic/issues/253)).

## References

- [Issue #7: Directional ablation damages thinking chain termination](https://github.com/RyderFreeman4Logos/heretic/issues/7)
- [Upstream issue #253](https://github.com/p-e-w/heretic/issues/253) (closed as out-of-scope)
