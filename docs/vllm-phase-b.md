# Experimental vLLM Phase B generation

HF remains the default and continues to compute residuals and KL scores. An
explicit `[vllm]` table routes only `Model.get_responses_batched` to an existing,
dedicated vLLM server. This prototype supports unquantized PEFT trials, not
`evaluate_model`, merged models, or the current bnb-4bit production workload.
There is no automatic engine launch or restart.

Use a trusted isolated server with the **same base weights, revision and
unquantized representation** as HF. Its architecture and every adapter target
module must support runtime LoRA. Prepare its shared mount and enable runtime
LoRA using `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` and `--enable-lora`, with
`--max-lora-rank` large enough for Heretic's configured rank. Prefer
`--generation-config vllm` so server generation defaults do not add unrelated
sampling constraints. Restrict the administrative endpoints to trusted callers.
Do not apply these settings to an existing shared production model service.

Create a private snapshot directory readable by the same user inside the engine
and add to the local `config.toml`:

```toml
[vllm]
endpoint = "http://127.0.0.1:8081/v1"
local_adapter_dir = "/absolute/private/shared/adapters"
server_adapter_dir = "/adapters"
timeout = 300
batch_size = 128
# api_key = "your dedicated endpoint credential"
```

The whole table is excluded from saved settings because it contains private
paths, network information and possibly a credential. Reconfigure it explicitly
when resuming a checkpoint. Omitting the table restores ordinary HF generation.

Every generation call exports the **current default PEFT adapter** to a new
owned directory, loads it under a fresh UUID model name, and sends pre-tokenized
prompts with a fresh prefix-cache salt. All batches in that call use the same
immutable snapshot. The local HF tokenizer applies the chat/thinking template
and response prefix, and decodes the returned generated token IDs using the
requested `skip_special_tokens` setting. Missing token IDs, response-index
mismatches, unsupported adapter layouts and HTTP errors fail the call; they do
not fall back to base or previously loaded weights.

After completion or generation failure, only the owned adapter is unloaded.
Files are removed only after confirmed unload. An uncertain unload retains the
snapshot and poisons that backend object so later trials cannot silently
continue. Reconcile the reported UUID/path on the dedicated engine before
restarting the run. No foreign adapter is unloaded and no base model is changed.

This is an experimental integration, **not a demonstrated speedup**. Loopback
contracts and a CPU random-model/real-PEFT caller test cannot prove vLLM numerical
parity, Qwen3.5 hybrid/quantization compatibility or GB10 throughput. Before a real
optimization, compare baseline and at least two changed adapters against HF on
a small fixed prompt set, including refusal/thinking decisions, then measure
export/load/generation/unload time under an explicit memory budget. Different
kernels and numeric representations can change greedy outputs. Issue #68 stays
open until the target performance acceptance is measured.

Contracts: [runtime LoRA](https://docs.vllm.ai/en/stable/features/lora/) and
[completion protocol](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/openai/completion/protocol/).
