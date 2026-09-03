---
name: trace-and-document-variables
description: Generate a self-contained HTML "cheat sheet" documenting the concrete structure, types and value patterns of the variables in a scope of code — a function, a file, or a set of selected lines — by instrumenting the real code with prints, running a real workload, and capturing actual runtime values (never invented/synthetic examples). Covers any variable worth tracing: tensors (shape/dtype/device — usually the hardest to hold in your head, so they get the most notation) but equally dicts, dataclasses/configs, lists, index arrays and scalars whose structure is doing real work. Use when a user is losing track of shapes, keys, padding or nesting across a debugging session and wants a durable visual reference instead of re-deriving it from prints/breakpoints each time. Works for any module in this repo (attention backend, model runner, custom ops) — the scope is whatever the user points at.
user-invocable: true
argument-hint: "<file>[:<line-range>] [--scenario <description>] [--run-config <path or args>]"
---

# Structure / value cheat sheet

Build a hierarchical, color-coded HTML reference for the concrete structure, types and value patterns of the variables in a user-specified scope — a whole file, one function, or a selected line range. "Structure" means whatever is hard to hold in your head for that variable's kind: shape/dtype/device for a tensor, keys and per-key value kinds for a dict, field values for a config/dataclass, length and element kind for a list, the value itself for a scalar. Tensors are usually the worst offenders, so they get the densest notation — but any variable whose structure carries logic belongs in the sheet. The point is to replace "print it, understand it, forget it, print it again next week" with a durable artifact the user re-opens instead of re-deriving.

**Everything in the cheat sheet must come from a real, instrumented run.** Never invent example shapes, keys or values, even plausible-looking ones — this is the one rule that can't be relaxed. If you can't run the code (no hardware, no fixture, missing input), say so and stop; do not fill the gap with a guess.

**Terseness is a hard requirement, but not silence.** This is a notation reference, not documentation. A variable is described by its structure (shape/dtype/device, keys, length, fields), a value preview, and — where it helps — a **few-word reminder** of what it holds, what a shape dimension counts, or what the non-padded prefix of a padded tensor represents. Assume the reader already knows the code: you are jogging their memory, not teaching them. "flat token → physical KV slot", "dim1 = kv heads (GQA)", "5 real seqs, rest bucket padding" is the right size; a sentence explaining *why* a reshape happens is not. See "Word budget" below — over-explaining is the most common way this skill produces something the user won't re-open.

## Two kinds of variables

Separate them explicitly; they are captured differently and rendered differently.

**Base configs** — the fixed values that hold for a whole run: `num_kv_heads`, `head_size`, `max_model_len`, `max_num_batched_tokens`, `block_size`, dtype, device, alignment constants. Tracing their "path" through the scope is not interesting. What matters is that they're *real*, because otherwise you'd be picking them arbitrarily and every downstream shape would be fiction. So: get them from **one end-to-end run** of the user's own basic scenario (an `examples/` script, the matching `.vscode/launch.json` config), print them once, and treat them as fixed inputs from then on.

**Traced variables** — anything whose structure actually moves across the scope, of any type:

- **Tensors** — query/key/value, block tables, slot mappings, masks, seq-len arrays, intermediate reshapes, the output. Usually the bulk of the sheet and the reason it exists.
- **Containers** — dicts (which keys are present, what each maps to, whether the key set is dynamic), lists/tuples (length, element kind, raggedness), nested combinations. A dict of per-layer tensors or a kwargs bag is often as confusing as any tensor.
- **Objects** — dataclasses, metadata/attn-metadata objects, config-ish structures constructed inside the scope: which fields are set, which are `None` on this path.
- **Scalars and index values** — token counts, offsets, bucket sizes, flags that select a branch. Cheap to capture and often the missing link between two shapes.

These are what the cheat sheet is *about*, and they're what you vary across scenarios. When capturing a container or object, capture its *structure* (keys/fields/length + the kind of each value), recursing into tensors it holds — not a full dump.

## Workflow

### 1. Read the scope, split base configs from traced variables

Read the target file(s) in full before touching anything. Identify:

- **Base configs** (per above) — list them; they anchor the top of the sheet as a compact strip.
- **Branch points**: `if`/dispatch logic that picks between code paths (bucketed vs per-seq, prefill vs decode, sliding-window vs not). Each reachable branch is one scenario/tab.
- **Capture points** for the traced variables: for each function in scope, the places whose locals actually explain how the data is structured and how those structures relate — usually right after inputs arrive, around a reshape/gather/scatter or a dict/object being assembled or unpacked, and right before dispatching into a compiled kernel. 1-3 per simple function; a branchier one needs more. Skip variables trivially derived from ones already captured, or unchanged since the last point. A print flood is as useless to future-you as too few prints.

### 2. One end-to-end run to pin the base configs

Run the user's own basic scenario once, end to end, with the debug flag on, purely to harvest the base configs (and to confirm the scope is reached at all).

```bash
SPYRE_DEBUG_DUMP=1 <same env the user's config uses> \
  uv run --no-sync python <example script> <same args the user's config uses> \
  2>&1 | tee .claude/skills/trace-and-document-variables/logs/<slug>-base.log
```

**Don't pre-flight the environment.** No checking whether a Spyre card is present, whether imports resolve, whether the device is configured — if the user says their scenario works, assume it works and run it directly. If it's actually broken, the run fails loudly — report the failure to the user and stop; don't debug it yourself. Probing devices and imports in isolation first only burns time and invents failure modes that aren't there, and so does chasing an environment problem that's the user's to resolve.

Record the harvested configs verbatim — they become the fixed inputs for step 3 and the "Base config" strip in the HTML. Respect the single-accelerator constraint from `CLAUDE.md`: never run this concurrently with another Spyre-backed command.

### 3. Patch and drive the scope locally, once per scenario

Do **not** try to reach every branch by rerunning end-to-end with different flags — that's slow (~3 min of vLLM startup each) and often can't reach a branch at all. Instead, with the base configs now known, drive the scope **as locally as possible**, test-style: call the function(s) directly from a throwaway script (or a targeted pytest invocation if a test already sets the scope up), constructing inputs from the real base configs, and patching/stubbing only what's between you and the branch you want.

```bash
uv run --no-sync python /tmp/<slug>_drive.py 2>&1 \
  | tee .claude/skills/trace-and-document-variables/logs/<slug>-<scenario>.log
```

Vary across scenarios what actually changes the structures: prefill vs decode, bucketed vs per-seq, which optional keys/fields are populated, and above all **ragged batches** — several concurrent requests of *different* prompt/generation lengths. Padding and bucketing logic is invisible when every sequence is the same size, so a uniform batch is a degenerate capture; construct a mixed-length one.

Keep the throwaway driver out of the repo (`/tmp`, or delete it at the end). Grep each log for `### DBG[` to see which branches actually fired.

**Don't chase unreachable branches.** A branch that won't fire may be dead code, gated by config you don't have, or plain buggy — that's a finding, not a task. Give it one or two honest attempts, then move on: build the sheet from the branches you did capture, and tell the user which ones didn't fire and what you saw. A sheet covering three of four paths, with the fourth named as unreached, is worth far more than one delayed by an afternoon of driver archaeology.

### 4. Instrumentation helper

Add a small helper near the top of the file under test (after imports/logger setup) and call it at the capture points. It must be:

- **Env-var gated** so it's silent by default: `SPYRE_DEBUG_DUMP=1`.
- **Step-ranged** so a multi-step workload doesn't flood stdout — a counter plus `SPYRE_DEBUG_STEP_START`/`SPYRE_DEBUG_STEP_END`, incremented once per top-level call into the scope (e.g. once per `build()`), not once per print. Push the start past warmup when the interesting behavior only begins later.
- **Type-aware**, not tensor-only: tensors print shape/dtype/device/numel plus a bounded preview; dicts print their key set with a one-line summary per value; lists/tuples print length and element kinds (flagging ragged ones); dataclasses/objects print their fields the same way; scalars print as-is. Everything bounded — never a full dump.

```python
import dataclasses
import os
_DBG_ON = os.environ.get("SPYRE_DEBUG_DUMP") == "1"
_DBG_STEP_START = int(os.environ.get("SPYRE_DEBUG_STEP_START", "0"))
_DBG_STEP_END = int(os.environ.get("SPYRE_DEBUG_STEP_END", "6"))
_DBG_STEP = 0

def _fmt(v, depth=0):
    if torch.is_tensor(v):
        prev = v.flatten()[:24].tolist()
        return (f"Tensor shape={tuple(v.shape)} dtype={v.dtype} device={v.device} "
                f"numel={v.numel()} preview={prev}{' ...' if v.numel() > 24 else ''}")
    if isinstance(v, dict):
        head = f"dict len={len(v)} keys={list(v)[:12]}{' ...' if len(v) > 12 else ''}"
        if depth >= 2:
            return head
        return head + "".join(f"\n{'  ' * (depth + 2)}[{k!r}] {_fmt(x, depth + 1)}"
                              for k, x in list(v.items())[:12])
    if isinstance(v, (list, tuple)):
        kinds = {type(x).__name__ for x in v}
        head = f"{type(v).__name__} len={len(v)} of {sorted(kinds)}"
        if depth >= 2 or not v:
            return head
        return head + f"\n{'  ' * (depth + 2)}[0] {_fmt(v[0], depth + 1)}"
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        fields = [f.name for f in dataclasses.fields(v)]
        head = f"{type(v).__name__} fields={fields}"
        if depth >= 2:
            return head
        return head + "".join(f"\n{'  ' * (depth + 2)}.{f} {_fmt(getattr(v, f), depth + 1)}"
                              for f in fields)
    return f"{type(v).__name__} = {v!r}"

def _dbg(tag, **kv):
    if not _DBG_ON or not (_DBG_STEP_START <= _DBG_STEP <= _DBG_STEP_END):
        return
    print(f"### DBG[{tag}]")
    for k, v in kv.items():
        print(f"  {k}: {_fmt(v)}")
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

Write one self-contained HTML file to `.claude/skills/trace-and-document-variables/logs/<slug>.html` and give the user the path. Self-contained means no external CSS/JS/font requests: inline everything so the file opens straight from disk. This is a **utilitarian/reference treatment**, not editorial — the user re-opens it repeatedly mid-debugging, so density and scannability outrank flourish.

**Start from the worked example.** `reference/example-cheatsheet.html` next to this file is a complete, correct sheet — treat it as the template and adapt it, rather than designing a new one. Carry over:

- Its **color palette** — the `:root` token block, with the light/dark/`[data-theme]` triple already in place.
- Its **row anatomy** — `.vrow` → `.name` + inline `<span class="loc">` gloss / `.sig` structure chips / `.data` monospace preview, with per-element `.n.pad` / `.n.neg` / `.n.hd` / `.n.kv` / `.n.sq` coloring.
- Its **chrome** — `.legend` strip, `.dims` base-config grid, real-vs-padding `.bar`, `<details class="fn">` per function, `.tabs-bar` + `.tab-panel` with the `switchTab` script.
- Its **gloss density** — the `loc` after each variable name, the one-line `.note` under a preview, the scenario `.sub` line, and the bulleted `.callout` are exactly the word budget described above, in situ.

Structure:

1. **Masthead** — scope covered, the exact command/config used, model/dtype.
2. **Legend** — the color key, once, small.
3. **Base config strip** — a compact grid of the fixed values from step 2. One glance re-anchors the mental model. Values only; a derived one may carry a few words (`num_queries_per_kv = 32/8 — GQA group size`).
4. **One tab per scenario/code path** (see below) — each tab headed by its few-word description and run facts — each holding one collapsible `<details>` per function, each holding a `<div class="vrow">` per variable. Every row is **name + few-word gloss**, then structure chips, then a value preview. Which chips depends on the kind — one row layout, kind-appropriate chips:
   - tensor: shape → dtype → device → value preview
   - dict: `len` → key list (keys as chips), one indented sub-row per interesting value
   - list/tuple: `len` → element kind → preview (mark ragged lengths explicitly)
   - dataclass/object: type name → one indented sub-row per field, `None` fields greyed like padding
   - scalar/flag: type → value
   Nest sub-rows at most two levels deep; below that, summarize.
5. **Footer** — one line: env-gated prints, since reverted; line numbers refer to the clean file.

**Scenarios go in tabs, not stacked sections** — one tab bar, one panel per scenario, first active on load, so switching between prefill and decode is a click rather than a scroll and the same variable sits in the same screen position across paths. Copy the `.tabs-bar` / `.tab-panel` CSS, the `<button onclick="switchTab('<name>', this)">` markup and the `switchTab` function from the example; the button names and the `tab-<name>` panel ids have to agree. The base-config strip stays **above** the bar, since it holds for every scenario.

#### Word budget

- **Variable rows**: an inline gloss of **a few words** next to the name — what the variable holds, in the reader's own vocabulary (`slot_mapping · flat token → physical KV slot`). Not a definition, a label.
- **Shape dimensions**: name what a dimension counts when it isn't obvious from the chip (`dim0 = flat tokens across batch`, `64 = b_seqs × num_kv_heads`). Do this once per novel shape, not on every row that repeats it.
- **Padded / bucketed tensors**: say what the real prefix is and what the padding is (`5 real decode rows, 3 bucket-padding rows`). This is the single most valuable gloss in the sheet — never skip it.
- A `<div class="note">` is for a fact the structure and values cannot convey, capped at **one short line** (e.g. "padding clamps to last valid row, not zero", "key absent on the decode path"). Never a restatement of the structure.
- **Per function**: a few words naming what comes in and what goes out. No description of the steps between.
- Everywhere: **one line, no second sentence.** If a gloss needs a second sentence to land, it's explaining rather than reminding — cut it.

#### Scenario descriptions

Every scenario/tab carries a **tiny high-level description** of what the case *is*, next to its title — the thing that stops the user having to reverse-engineer the setup from the shapes each time they open the tab:

- Shape: a handful of words. `single sequence, full prefill` · `per-seq loop, ragged batch` · `4D full-batch decode matmul` · `3 decode + 1 mid-prefill + 1 full-prefill`.
- Plus a one-line sub-label with the concrete run facts: step number, `num_reqs`, per-seq query/kv lengths, the env var that selected the path.
- If the scenario is genuinely complex (an unusual dispatch, a non-obvious precondition, a bucket lattice that has to be known to read the numbers), a slightly longer note is fine — a sentence or two of context, once, at the top of the tab. Complexity earns words; a straightforward prefill does not.

#### Design conventions

- **Color coding is semantic and consistent document-wide**: one color per *concept* (head/query dims, KV/block dims, sequence/batch dims), reused everywhere that concept appears — in shape chips, dict keys, field names and inline numbers alike, so a dict keyed by layer and a tensor sized by layer read as the same concept. Padding gets its own neutral grey, used for every padded value in every tab, so "that's padding" reads on sight. A masked/sentinel value (e.g. fp16 min in an additive mask) gets its own fixed color.
- **Notation over prose, glossed**: each variable as `name` + few-word gloss / structure chips / monospace value preview with per-element coloring. The gloss rides inline with the name (the example's `<span class="loc">`), so it reads as part of the notation rather than a paragraph beside it. Tensors get the densest treatment (shape/dtype/device + colored elements) because they are the hardest to read; containers get keys/lengths as chips with nested rows for the values that matter.
- **Real vs. padding ratio bars**: for bucketed/padded tensors, a tiny two-segment bar (solid = real, hatched = padding) next to the preview makes the ratio legible without counting.
- **Collapsible `<details>` per function**, all open.
- **Theme-aware**: light tokens on bare `:root`, redefined under `@media (prefers-color-scheme: dark)` guarded by `:root:not([data-theme="light"])`, and again under `:root[data-theme="dark"]`. Never define a color only inside a media/attribute block.
- **Fonts**: system stacks only, since the file must open offline from disk — `ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace` for all notation (digits and brackets need to line up), `system-ui, sans-serif` for labels. Don't link Google Fonts.

## What NOT to do

- Don't write prose. Structures and values are the content, glossed with a few words each; a paragraph explaining a reshape or a dict's layout is noise the user has to skip past every time they open the sheet.
- Don't strip the sheet down to bare notation either — an unlabelled `(64, 4, 1, 128)` costs the user the same re-derivation the sheet exists to prevent. Gloss it.
- Don't leave a scenario tab unlabelled — every tab says in a few words what the case is (see "Scenario descriptions").
- Don't invent or extrapolate structures or values "because they're plausible" — every number traces back to a captured `### DBG[...]` block.
- Don't pick base configs yourself — harvest them from the user's own end-to-end run (step 2).
- Don't pre-flight hardware/imports/device setup before the first run — trust the user's scenario and run it. If it fails, report the failure and stop; don't debug the environment.
- Don't rerun end-to-end once per branch; patch and drive the scope locally instead (step 3).
- Don't instrument every line — capture the points that explain how the structures relate, skipping trivial or unchanged variables.
- Don't restrict the sheet to tensors. A dict whose key set changes per branch, a metadata object with half its fields `None`, or a bucket-size scalar all belong in it — tensors get the most notation, not exclusive coverage.
- Don't dump a container in full. Keys/fields/length plus a bounded per-value summary, two levels deep at most.
- Don't leave debug prints in the source file after capture — always revert (step 5).
- Don't stack scenarios as long scrolling sections — use the tab bar.
- Don't paste multi-fact callouts as one paragraph — one `<li>` per fact.
- Don't link web fonts; the file must render identically opened offline from disk.
- Don't skip an *easily* reachable branch to save time — the bucketed-vs-per-seq (or equivalent) distinction is usually the point of the sheet. But don't sink time into one that resists either: report it as unreached and ship the rest.
