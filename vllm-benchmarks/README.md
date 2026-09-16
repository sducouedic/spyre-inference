# Spyre vLLM benchmarks

Benchmark configs for the `vLLM Benchmark` CI workflow and for local runs on
Spyre hardware. Each config file under `benchmarks/spyre/` is a YAML list of
test entries; one entry per `(model, shape)`:

- `latency-tests.yaml` → `vllm bench latency`
- `throughput-tests.yaml` → `vllm bench throughput`
- `serve-tests.yaml` → `vllm bench serve` (starts a server, waits for health,
  then benchmarks against it)

Serve entries come in two flavours. `*_in64_out64` sends random prompts at a fixed 64-in/64-out shape — cheap, and the right smoke test. `*_4k` replays a recorded agentic trace instead, served with `max-model-len: 4096`: real router prompts, each request keeping the output length it actually produced, replayed in recorded order so prefix-cache behaviour is reproducible. Use the latter for performance numbers; the fixed shape cannot show prefill chunking, prefix reuse, or KV-block pressure.

The trace path is the `SPYRE_AIOPS_DATASET` environment variable, so each host can point it at its own copy. Unset, it falls back to the location on the Spyre benchmark hosts. Where the file is not present, the runner skips the `*_4k` entries with a warning instead of failing.

## Running locally

Benchmarks run through the `perf-tests` Make target. Two optional filters:

- `MODELS` — comma-separated model names (matched case-insensitively). Empty =
  all models.
- `BENCH_TYPES` — comma-separated subset of `latency,throughput,serve`. Empty =
  all types.

```bash
# Everything (all models, all bench types)
make perf-tests RESULTS_DIR=benchmark-results

# Just the serve benchmark for one model
make perf-tests RESULTS_DIR=benchmark-results \
  MODELS=ibm-granite/granite-3.3-8b-instruct \
  BENCH_TYPES=serve
```
