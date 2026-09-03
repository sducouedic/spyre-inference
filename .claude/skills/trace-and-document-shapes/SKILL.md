---
name: trace-and-document-shapes
description: Generate a self-contained HTML "cheat sheet" documenting the concrete shapes, dtypes, and value patterns of variables in a scope of code — a function, a file, or a set of selected lines — by instrumenting the real code with prints, running a real workload, and capturing actual runtime values (never invented/synthetic examples). Use when a user is losing track of tensor shapes/dtypes/padding across a debugging session and wants a durable visual reference instead of re-deriving it from prints/breakpoints each time. Works for any module in this repo (attention backend, model runner, custom ops) — the scope is whatever the user points at.
user-invocable: true
argument-hint: "<file>[:<line-range>] [--scenario <description>] [--run-config <path or args>]"
---

# Shape / value cheat sheet

Build a hierarchical, color-coded HTML reference for the concrete shapes, dtypes, and value patterns of the variables in a user-specified scope — a whole file, one function, or a selected line range. The point is to replace "print it, understand it, forget it, print it again next week" with a durable artifact the user re-opens instead of re-deriving.

**Everything in the cheat sheet must come from a real, instrumented run.** Never invent example shapes/values, even plausible-looking ones — this is the one rule that can't be relaxed. If you can't run the code (no hardware, no fixture, missing input), say so and stop; do not fill the gap with a guess.

**Terseness is a hard requirement.** This is a notation reference, not documentation. A variable is described by its shape, dtype, device and a value preview — nothing else. Where words are unavoidable, spend at most **a few** on the inputs, the few local variables that actually carry the logic, and the final output. No sentences explaining what a reshape does; the before/after shapes already say it. See "Word budget" below — violating it is the most common way this skill produces something the user won't re-open.

## Two kinds of variables

Separate them explicitly; they are captured differently and rendered differently.

**Base configs** — the fixed values that hold for a whole run: `num_kv_heads`, `head_size`, `max_model_len`, `max_num_batched_tokens`, `block_size`, dtype, device, alignment constants. Tracing their "path" through the scope is not interesting. What matters is that they're *real*, because otherwise you'd be picking them arbitrarily and every downstream shape would be fiction. So: get them from **one end-to-end run** of the user's own basic scenario (an `examples/` script, the matching `.vscode/launch.json` config), print them once, and treat them as fixed inputs from then on.

**Traced variables** — the tensors and locals whose shapes actually move: query/key/value, block tables, slot mappings, masks, seq-len arrays, intermediate reshapes, the output. These are what the cheat sheet is *about*, and they're what you vary across scenarios.

## Workflow

### 1. Read the scope, split base configs from traced variables

Read the target file(s) in full before touching anything. Identify:

- **Base configs** (per above) — list them; they anchor the top of the sheet as a compact strip.
- **Branch points**: `if`/dispatch logic that picks between code paths (bucketed vs per-seq, prefill vs decode, sliding-window vs not). Each reachable branch is one scenario/tab.
- **Capture points** for the traced variables: for each function in scope, the places whose locals actually explain how shapes relate — usually right after inputs arrive, around a reshape/gather/scatter, and right before dispatching into a compiled kernel. 1-3 per simple function; a branchier one needs more. Skip variables trivially derived from ones already captured, or unchanged since the last point. A print flood is as useless to future-you as too few prints.

### 2. One end-to-end run to pin the base configs

Run the user's own basic scenario once, end to end, with the debug flag on, purely to harvest the base configs (and to confirm the scope is reached at all).

```bash
SPYRE_DEBUG_DUMP=1 <same env the user's config uses> \
  uv run --no-sync python <example script> <same args the user's config uses> \
  2>&1 | tee .claude/skills/trace-and-document-shapes/logs/<slug>-base.log
```

**Don't pre-flight the environment.** No checking whether a Spyre card is present, whether imports resolve, whether the device is configured — if the user says their scenario works, assume it works and run it directly. If it's actually broken, the run fails loudly — report the failure to the user and stop; don't debug it yourself. Probing devices and imports in isolation first only burns time and invents failure modes that aren't there, and so does chasing an environment problem that's the user's to resolve.

Record the harvested configs verbatim — they become the fixed inputs for step 3 and the "Base config" strip in the HTML. Respect the single-accelerator constraint from `CLAUDE.md`: never run this concurrently with another Spyre-backed command.

### 3. Patch and drive the scope locally, once per scenario

Do **not** try to reach every branch by rerunning end-to-end with different flags — that's slow (~3 min of vLLM startup each) and often can't reach a branch at all. Instead, with the base configs now known, drive the scope **as locally as possible**, test-style: call the function(s) directly from a throwaway script (or a targeted pytest invocation if a test already sets the scope up), constructing inputs from the real base configs, and patching/stubbing only what's between you and the branch you want.

```bash
uv run --no-sync python /tmp/<slug>_drive.py 2>&1 \
  | tee .claude/skills/trace-and-document-shapes/logs/<slug>-<scenario>.log
```

Vary across scenarios what actually changes the shapes: prefill vs decode, bucketed vs per-seq, and above all **ragged batches** — several concurrent requests of *different* prompt/generation lengths. Padding and bucketing logic is invisible when every sequence is the same size, so a uniform batch is a degenerate capture; construct a mixed-length one.

Keep the throwaway driver out of the repo (`/tmp`, or delete it at the end). Grep each log for `### DBG[` to see which branches actually fired.

**Don't chase unreachable branches.** A branch that won't fire may be dead code, gated by config you don't have, or plain buggy — that's a finding, not a task. Give it one or two honest attempts, then move on: build the sheet from the branches you did capture, and tell the user which ones didn't fire and what you saw. A sheet covering three of four paths, with the fourth named as unreached, is worth far more than one delayed by an afternoon of driver archaeology.

### 4. Instrumentation helper

Add a small helper near the top of the file under test (after imports/logger setup) and call it at the capture points. It must be:

- **Env-var gated** so it's silent by default: `SPYRE_DEBUG_DUMP=1`.
- **Step-ranged** so a multi-step workload doesn't flood stdout — a counter plus `SPYRE_DEBUG_STEP_START`/`SPYRE_DEBUG_STEP_END`, incremented once per top-level call into the scope (e.g. once per `build()`), not once per print. Push the start past warmup when the interesting behavior only begins later.
- **Tensor-aware**: shape, dtype, device, numel, and a bounded value preview — never a full dump.

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

Match the file's existing logging conventions (check for `logger = init_logger(__name__)`) rather than importing `logging` fresh.

### 5. Revert the instrumentation

Before building the HTML, remove the helper and all `_dbg(...)` calls so the source file is back to its original state:

```bash
git diff --stat <instrumented file>
git checkout -- <instrumented file>
```

Keep the logs — they're the source data for step 6. Never ship a cheat sheet whose source file still carries debug prints.

### 6. Build the HTML

Write one self-contained HTML file to `.claude/skills/trace-and-document-shapes/logs/<slug>.html` and give the user the path. Self-contained means no external CSS/JS/font requests: inline everything so the file opens straight from disk. This is a **utilitarian/reference treatment**, not editorial — the user re-opens it repeatedly mid-debugging, so density and scannability outrank flourish.

Structure:

1. **Masthead** — scope covered, the exact command/config used, model/dtype.
2. **Legend** — the color key, once, small.
3. **Base config strip** — a compact grid of the fixed values from step 2. One glance re-anchors the mental model. Values only, no prose.
4. **One tab per scenario/code path** (see below), each holding one collapsible `<details>` per function, each holding a `<div class="vrow">` per variable. Every row leads with **shape → dtype → device**, then a value preview.
5. **Footer** — one line: env-gated prints, since reverted; line numbers refer to the clean file.

**Scenarios go in tabs, not stacked sections.** One tab bar across the top, one panel per scenario, first panel active on load — so switching between prefill and decode is a click rather than a scroll, and the same variable sits in the same screen position across paths. The markup and script, which have to agree on the `tab-<name>` ids:

```html
<div class="tabs-bar">
  <button class="tab-btn active" onclick="switchTab('prefill', this)">Prefill</button>
  <button class="tab-btn"        onclick="switchTab('decode', this)">Decode</button>
</div>
<div id="tab-prefill" class="tab-panel active">…</div>
<div id="tab-decode"  class="tab-panel">…</div>
<script>
function switchTab(name, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}
</script>
```

Style it however suits the sheet, with one catch: give every `.tab-btn` a transparent `border-bottom` of the same width as the active one, so the row doesn't shift when the underline appears.

#### Word budget

- Variable rows: **no prose at all** by default — `name` / shape / dtype / device / value preview.
- A `<div class="note">` is allowed only for a fact the shape and values cannot convey, and is capped at **one short line** (e.g. "padding clamps to last valid row, not zero"). Never a restatement of the shape.
- Per function: at most a handful of words naming what comes in and what goes out. No description of the steps between.
- If you're writing a second sentence anywhere, delete it.

#### Design conventions

- **Color coding is semantic and consistent document-wide**: one color per *concept* (head/query dims, KV/block dims, sequence/batch dims), reused everywhere that concept appears, in chips and inline numbers alike. Padding gets its own neutral grey, used for every padded value in every tab, so "that's padding" reads on sight. A masked/sentinel value (e.g. fp16 min in an additive mask) gets its own fixed color.
- **Notation over prose**: each tensor as `name` / shape chip / dtype chip / device chip / monospace value preview with per-element coloring.
- **Real vs. padding ratio bars**: for bucketed/padded tensors, a tiny two-segment bar (solid = real, hatched = padding) next to the preview makes the ratio legible without counting.
- **Collapsible `<details>` per function**, all open.
- **Theme-aware**: light tokens on bare `:root`, redefined under `@media (prefers-color-scheme: dark)` guarded by `:root:not([data-theme="light"])`, and again under `:root[data-theme="dark"]`. Never define a color only inside a media/attribute block.
- **Fonts**: system stacks only, since the file must open offline from disk — `ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace` for all notation (digits and brackets need to line up), `system-ui, sans-serif` for labels. Don't link Google Fonts.

## What NOT to do

- Don't write prose. Shapes and values are the content; a paragraph explaining a reshape is noise the user has to skip past every time they open the sheet.
- Don't invent or extrapolate values "because they're plausible" — every number traces back to a captured `### DBG[...]` block.
- Don't pick base configs yourself — harvest them from the user's own end-to-end run (step 2).
- Don't pre-flight hardware/imports/device setup before the first run — trust the user's scenario and run it. If it fails, report the failure and stop; don't debug the environment.
- Don't rerun end-to-end once per branch; patch and drive the scope locally instead (step 3).
- Don't instrument every line — capture the points that explain the shape relationships, skipping trivial or unchanged variables.
- Don't leave debug prints in the source file after capture — always revert (step 5).
- Don't stack scenarios as long scrolling sections — use the tab bar.
- Don't skip an *easily* reachable branch to save time — the bucketed-vs-per-seq (or equivalent) distinction is usually the point of the sheet. But don't sink time into one that resists either: report it as unreached and ship the rest.
