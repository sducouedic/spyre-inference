---
name: trace-and-document-shapes
description: Generate a self-contained HTML "cheat sheet" documenting the concrete shapes, dtypes, and value patterns of variables in a scope of code — a function, a file, or a set of selected lines — by instrumenting the real code with prints, running a real workload, and capturing actual runtime values (never invented/synthetic examples). Use when a user is losing track of tensor shapes/dtypes/padding across a debugging session and wants a durable visual reference instead of re-deriving it from prints/breakpoints each time. Works for any module in this repo (attention backend, model runner, custom ops) — the scope is whatever the user points at.
user-invocable: true
argument-hint: "<file>[:<line-range>] [--scenario <description>] [--run-config <path or args>]"
---

# Shape / value cheat sheet

Build a hierarchical, color-coded HTML reference for the concrete shapes, dtypes, and value patterns of the variables in a user-specified scope — a whole file, one function, or a selected line range. The point is to replace "print it, understand it, forget it, print it again next week" with a durable artifact the user re-opens instead of re-deriving.

**Everything in the cheat sheet must come from a real, instrumented run.** Never invent example shapes/values, even plausible-looking ones — this is the one rule that can't be relaxed. If you can't run the code (no hardware, no fixture, missing input), say so and stop; do not fill the gap with a guess.

## Inputs

- **Scope** (required): a file path, optionally with a line range (`spyre_inference/v1/attention/backends/spyre_attn.py:792-971`), a function name, or the user's current IDE selection. If the user gives a bare file, cover its main entry points (the functions a caller actually calls), not every private helper.
- **Scenario(s)** (optional): what workload/config exercises the interesting paths. If omitted, ask what the user normally runs, or find an example script under `examples/` and a matching launch config under `.vscode/launch.json`. Prefer whatever the user already uses for this code — don't invent a new harness.
- **Number of scenarios**: default to capturing **every distinct code path** the scope can take (e.g. prefill vs decode, bucketed vs per-seq loop, sliding-window vs not) rather than just one run. Ask the user only if the scope's branches aren't obvious from a first read.

## Workflow

### 1. Read the scope first, plan the instrumentation

Read the target file(s) in full before touching anything. Identify:

- **Level-0 variables**: constants / config that hold for the whole scope — dims like `num_heads`, alignment constants, dtype, device. These almost never change between calls and anchor the top of the cheat sheet.
- **Branch points**: `if`/dispatch logic that picks between code paths (e.g. bucketed vs per-seq, prefill vs decode, sliding-window vs full). Each reachable branch is a candidate scenario.
- **Representative capture points**: for each function in scope, the places whose local variables actually explain how shapes relate to each other — usually right after inputs are received, right before/after a reshape/gather/scatter, and right before dispatching into a compiled kernel. That's often 1-3 points for a simple function, but a longer or branchier one may need more — the target is full understanding of the scope, not a fixed count. Still don't instrument every line: skip variables that are trivially derived from ones you've already captured, or that don't change between the points you're already printing. A print flood is as useless to future-you as too few prints.
- **Interesting, not just representative, inputs**: a capture point is only as good as the batch that hits it. Drive the workload with multiple concurrent requests of *different* prompt/generation lengths, not a single request or a batch of identical lengths — padding, bucketing, and ragged-batch logic are invisible when every sequence in the batch is the same size. If the scenario is user-supplied and under-specifies this, ask for or construct a mixed-length batch rather than capturing a degenerate uniform one.

Do not start editing/running yet — form this plan mentally (or jot it in your own scratch notes), then move to instrumentation.

### 2. Instrument with a throwaway, env-gated debug helper

Add a small helper near the top of the file under test (after imports/logger setup) and call it at the capture points identified above. It must be:

- **Env-var gated** so it's silent by default and never bothers anyone re-running the same file without the flag, e.g. `SPYRE_DEBUG_DUMP=1`.
- **Step-ranged** so a multi-step workload (e.g. 5 prompts × N decode steps) doesn't flood stdout — gate with a counter and an `SPYRE_DEBUG_STEP_START=<n>`/`SPYRE_DEBUG_STEP_END=<n>` window, incremented once per top-level call into the scope. Default the start to `0` so the common case is unchanged, but when the scope of interest only becomes active after some warmup (e.g. decode behavior after N prefill-only calls), push the start past the uninteresting steps instead of burning the window on them.
- **Tensor-aware**: print shape, dtype, device, numel, and a bounded preview of values (e.g. first ~24 elements or `.tolist()` for small 1-D tensors) — never a full dump of a large tensor.

A minimal helper (adapt names to the file's existing logging style — check for an existing `logger = init_logger(__name__)` and match its conventions rather than importing `logging` fresh):

```python
import os
_DBG_ON = os.environ.get("SPYRE_DEBUG_DUMP") == "1"
_DBG_STEP_START = int(os.environ.get("SPYRE_DEBUG_STEP_START", "0"))
_DBG_STEP_END = int(os.environ.get("SPYRE_DEBUG_STEP_END", "6"))
_DBG_STEP = 0

def _dbg(tag, **kv):
    if not _DBG_ON or not (_DBG_STEP_START <= _DBG_STEP <= _DBG_STEP_END):
        return
    print(f"### DBG[{tag}]")
    for k, v in kv.items():
        if torch.is_tensor(v):
            prev = v.flatten()[:24].tolist()
            print(f"  {k}: Tensor shape={tuple(v.shape)} dtype={v.dtype} device={v.device} "
                  f"numel={v.numel()} preview={prev}{' ...' if v.numel() > 24 else ''}")
        else:
            print(f"  {k}: {type(v).__name__} = {v!r}")
```

Increment `_DBG_STEP` once per call into the scope's top-level entry point (e.g. once per `build()` call), not once per print — otherwise a single request can exhaust the step budget on its first line.

### 3. Run a real workload and capture output

Use whatever the user actually runs for this code — an example script under `examples/`, the matching `.vscode/launch.json` config if one exists, or a targeted pytest invocation if the scope is better exercised by a unit test than an end-to-end script. Don't build a new harness if one already fits; don't over-debug the environment either — if the user says "it works for me, just run it," trust that and run it directly rather than pre-flighting imports/devices in isolation.

```bash
SPYRE_DEBUG_DUMP=1 SPYRE_DEBUG_STEP_START=0 SPYRE_DEBUG_STEP_END=6 <same env the user's config uses> \
  uv run --no-sync python <example script> <same args the user's config uses> \
  2>&1 | tee .claude/skills/trace-and-document-shapes/logs/<slug>-run.log
```

Raise `SPYRE_DEBUG_STEP_START` past the warmup calls when the scope's interesting behavior only kicks in later (e.g. decode-only prints need the start pushed past the prefill steps) — don't spend the whole window capturing steps you already understand.

Respect the single-accelerator constraint from `CLAUDE.md` — never run this concurrently with another Spyre-backed command.

Grep the log for `### DBG[` blocks and confirm you actually captured every scenario planned in step 1 (e.g. both the bucketed and per-seq paths). If a scenario didn't fire, adjust the run args (batch size, prompt count, env vars) and re-run — don't ship a cheat sheet missing a branch just because the first run didn't exercise it.

### 4. Revert the instrumentation

Before building the HTML, remove the debug helper and all `_dbg(...)` calls so the source file is back to its original state:

```bash
git diff --stat <instrumented file>   # confirm what you're about to revert
git checkout -- <instrumented file>   # or restore from a saved copy if the file predates this session's git history
```

Keep the captured log — it's the source data for step 5. Never ship a cheat sheet whose source file still carries the debug prints.

### 5. Build the HTML

Load the `artifact-design` skill before writing the file (required for any HTML artifact) — this is a **utilitarian/reference treatment**, not editorial: the user re-opens this repeatedly mid-debugging, so information density and scannability outrank visual flourish. Then write one self-contained HTML file and publish it with `Artifact` (or, if artifact publishing isn't available in the session, save it under `.claude/skills/trace-and-document-shapes/logs/<slug>.html` and tell the user the path).

Structure, matching the pattern validated on the attention-backend cheat sheet:

1. **Masthead** — scope covered, the exact command/config used to capture it, model/dtype if relevant.
2. **Legend** — the color key, stated once, small.
3. **Level 0 strip** — a compact grid of the global/config values that hold for the whole scope (dims, alignment constants, special values like a mask fill constant). One glance should re-anchor the whole mental model.
4. **One `<section>` per scenario/code path**, each holding one collapsible `<details>` block per function, each holding a `<div class="vrow">` per variable. Every variable row leads with **shape → dtype → device**, then a value preview — notation before prose, matching the user's explicit ask. Add a one-line `<div class="note">` only when the pattern isn't obvious from the shape/values alone (e.g. "padding clamps to the last valid row, not zero").
5. **Footer** — how the data was captured (env-gated prints, now reverted) and that line numbers refer to the clean file.

Design conventions to reuse (don't re-derive from scratch each time, but do adapt the palette/type pairing to the specific subject rather than copy-pasting verbatim):

- **Color coding is semantic and consistent across the whole document**: pick one color per *concept* (e.g. head/query dims, KV/block dims, sequence/batch dims) and reuse it everywhere that concept appears, in both prose chips and inline numbers. Padding gets its own neutral color (grey) used consistently for every padded value/row in every scenario — the user should recognize "that's padding" on sight without reading a label. If the domain has a masked/sentinel value (e.g. fp16 min for an additive attention mask), give it a fixed color too.
- **Notation over prose**: render each tensor as `name` / shape chip / dtype chip / device chip / a monospace value preview with per-element coloring (real values in the concept's color, padding in grey, sentinel values in their own color). A short `<div class="note">` is for the one fact the shape alone doesn't convey — not a restatement of the shape.
- **Real vs. padding ratio bars**: for bucketed/padded tensors, a tiny horizontal two-segment bar (solid = real, hatched/muted = padding) next to the value preview makes the ratio legible without counting elements.
- **Collapsible `<details>` per function** so the page stays scannable at a glance and expandable on demand — default the first scenario open, collapse the rest, or default them all open if the total content is short enough to not need collapsing.
- **Theme-aware**: define light-mode tokens on bare `:root`, redefine under `@media (prefers-color-scheme: dark)` guarded by `:root:not([data-theme="light"])`, and again under `:root[data-theme="dark"]` — see the `artifact-design` skill for the exact pattern. Never define a color only inside a media/attribute block.
- **Fonts**: a real monospace face for all tensor/shape/value notation (e.g. JetBrains Mono) since digits and brackets need to line up and read cleanly; a plain sans (e.g. Inter) for labels/prose so it stays out of the way.

## What NOT to do

- Don't invent or extrapolate example values "because they're plausible" — every number in the sheet must trace back to a captured `### DBG[...]` block.
- Don't instrument every line of the scope — pick the capture points per function that actually explain the shape relationships, per step 1, skipping variables that are trivial or unchanged since the last capture.
- Don't leave debug prints in the source file after capture — always revert (step 4) before considering the task done.
- Don't build a new run harness when an existing example script + launch config already exercises the scope — reuse what's there.
- Don't skip a reachable branch/scenario to save time; a cheat sheet missing the bucketed-vs-per-seq (or equivalent) distinction defeats the purpose.
