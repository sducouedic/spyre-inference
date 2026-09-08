---
name: trace-and-document-variables
description: Generate a self-contained HTML "cheat sheet" documenting the concrete structure, types and value patterns of the variables in a scope of code — a function, a file, or a set of selected lines — by instrumenting the real code with prints, running a real workload, and capturing actual runtime values (never invented/synthetic examples). Covers any variable worth tracing: tensors (shape/dtype/device — usually the hardest to hold in your head, so they get the most notation) but equally dicts, dataclasses/configs, lists, index arrays and scalars whose structure is doing real work. Use when a user is losing track of shapes, keys, padding or nesting across a debugging session and wants a durable visual reference instead of re-deriving it from prints/breakpoints each time. Works for any module in this repo (attention backend, model runner, custom ops) — the scope is whatever the user points at.
user-invocable: true
argument-hint: "<file>[:<line-range>] [--scenario <description>] [--run-config <path or args>]"
---

# Structure / value cheat sheet

Build a hierarchical, color-coded HTML reference for the concrete structure, types and value patterns of the variables in a user-specified scope — a whole file, one function, or a line range. "Structure" means whatever is hard to hold in your head for that variable's kind: shape/dtype/device for a tensor, keys and per-key value kinds for a dict, field values for a config/dataclass, length and element kind for a list, the value itself for a scalar. Tensors get the densest notation, but any variable whose structure carries logic belongs in the sheet. The point is to replace "print it, understand it, forget it, print it again next week" with a durable artifact.

**Everything must come from a real, instrumented run.** Never invent example shapes, keys or values, even plausible-looking ones — this is the one rule that can't be relaxed. If you can't run the code (no hardware, no fixture, missing input), say so and stop.

**Terseness is a hard requirement, but not silence.** This is a notation reference, not documentation. A variable is described by its structure, a value preview, and — where it helps — a **few-word reminder** of what it holds, what a shape dimension counts, or what the non-padded prefix of a padded tensor represents. Assume the reader knows the code: jog their memory, don't teach them. "flat token → physical KV slot", "dim1 = kv heads (GQA)", "5 real seqs, rest bucket padding" is the right size; a sentence explaining *why* a reshape happens is not. Over-explaining is the most common way this skill produces something the user won't re-open.

## Two kinds of variables

**Base configs** — fixed values holding for a whole run: `num_kv_heads`, `head_size`, `max_model_len`, `max_num_batched_tokens`, `block_size`, dtype, device, alignment constants. Tracing their path is not interesting; what matters is that they're *real*, since otherwise every downstream shape is fiction. Harvest them from **one end-to-end run** of the user's own basic scenario (an `examples/` script, the matching `.vscode/launch.json` config), then treat them as fixed inputs.

**Traced variables** — anything whose structure moves across the scope, of any type: **tensors** (q/k/v, block tables, slot mappings, masks, seq-len arrays, intermediate reshapes, output — usually the bulk); **containers** (dicts — which keys are present, what each maps to, whether the key set is dynamic; lists/tuples — length, element kind, raggedness; nested combinations); **objects** (dataclasses, attn-metadata, configs built inside the scope — which fields are set, which are `None` on this path); **scalars and index values** (token counts, offsets, bucket sizes, branch-selecting flags — cheap to capture, often the missing link between two shapes).

These are what the sheet is *about* and what you vary across scenarios. For a container or object, capture its *structure* (keys/fields/length + kind of each value), recursing into tensors it holds — not a full dump.

## Workflow

### 1. Read the scope, split base configs from traced variables

Read the target file(s) in full before touching anything. Identify base configs (they anchor the top of the sheet as a compact strip); **branch points** (`if`/dispatch logic choosing between code paths — bucketed vs per-seq, prefill vs decode, sliding-window vs not; each reachable branch is one scenario/tab); and capture points.

**Capture broadly — err heavily on the side of more.** Capturing a variable is nearly free (an env-gated print, reverted in step 5); a missing one is expensive — you only find out after the run, and fixing it means re-instrumenting and re-driving. Sweep every function in scope: every local holding a tensor, container, object or structurally meaningful scalar, at every point its structure changes — inputs as they arrive, each reshape/gather/scatter, each dict/object assembled or unpacked, each branch taken, the state right before dispatching into a compiled kernel. Prefer re-printing over assuming unchanged. The only genuine limits are stdout volume across *steps* (handled by the step range, step 4) and truncating giant previews — not the number of variables. Selectivity belongs in the **rendering** (step 6, "Ranking a wide capture"): capture wide, render ranked.

**Flaws, surprises and sharp edges are a first-class output.** While reading logs, note anything wrong or merely surprising: a padded region holding live values instead of zeros/sentinels, an off-by-one in a length or offset, a dtype/device disagreeing with neighbours, a field `None` where the code expects a value, a shape that only works because two constants happen to be equal, a branch firing when it shouldn't, a `FallbackWarning`. Do **not** fix and do **not** chase these — capture the evidence and record it (step 6, "Flagging discovered flaws"). A wide capture surfaces these for free, which is much of why it's worth doing.

### 2. One end-to-end run to pin the base configs

Run the user's own basic scenario once, end to end, with the debug flag on, purely to harvest base configs (and confirm the scope is reached at all).

```bash
SPYRE_DEBUG_DUMP=1 <same env the user's config uses> \
  uv run --no-sync python <example script> <same args the user's config uses> \
  2>&1 | tee .claude/skills/trace-and-document-variables/logs/<slug>-base.log
```

**Don't pre-flight the environment.** No checking for a Spyre card, resolvable imports, or device config — if the user says their scenario works, run it directly. If it's broken the run fails loudly: report the failure and stop, don't debug it yourself. Probing in isolation only burns time and invents failure modes that aren't there.

Record the harvested configs verbatim — they are the fixed inputs for step 3 and the "Base config" strip. Respect the single-accelerator constraint from `CLAUDE.md`: never run this concurrently with another Spyre-backed command.

### 3. Patch and drive the scope locally, once per scenario

Do **not** reach branches by rerunning end-to-end with different flags — that's slow (~3 min of vLLM startup each) and often can't reach a branch at all. With base configs known, drive the scope **as locally as possible**, test-style: call the function(s) directly from a throwaway script (or a targeted pytest invocation if a test already sets the scope up), building inputs from the real base configs and patching/stubbing only what's between you and the branch you want.

```bash
uv run --no-sync python /tmp/<slug>_drive.py 2>&1 \
  | tee .claude/skills/trace-and-document-variables/logs/<slug>-<scenario>.log
```

Vary across scenarios what actually changes the structures: prefill vs decode, bucketed vs per-seq, which optional keys/fields are populated, and above all **ragged batches** — several concurrent requests of *different* prompt/generation lengths. Padding and bucketing logic is invisible when every sequence is the same size, so a uniform batch is a degenerate capture.

Keep the driver out of the repo (`/tmp`, or delete it at the end). Grep each log for `### DBG[` to see which branches fired.

**Don't chase unreachable branches.** A branch that won't fire may be dead code, gated by config you don't have, or buggy — that's a finding, not a task. Give it one or two honest attempts, then build the sheet from what you captured and tell the user which branches didn't fire and what you saw. Three of four paths with the fourth named as unreached beats an afternoon of driver archaeology.

### 4. Instrumentation helper

Add a small helper near the top of the file under test (after imports/logger setup) and call it at the capture points. It must be **env-gated** (`SPYRE_DEBUG_DUMP=1`, silent by default); **step-ranged** so a multi-step workload doesn't flood stdout — a counter plus `SPYRE_DEBUG_STEP_START`/`SPYRE_DEBUG_STEP_END`, incremented once per top-level call into the scope (e.g. once per `build()`), not once per print, pushing the start past warmup when the interesting behavior begins later; and **type-aware**, not tensor-only — tensors print shape/dtype/device/numel plus a bounded preview; dicts their key set with a one-line summary per value; lists/tuples length and element kinds (flagging ragged); dataclasses/objects their fields the same way; scalars as-is. Everything bounded — never a full dump.

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

Before building the HTML, remove the helper and all `_dbg(...)` calls so the source file is back to its original state. Never ship a cheat sheet whose source file still carries debug prints. Keep the logs — they're the source data for step 6.

```bash
git diff --stat <instrumented file>
git checkout -- <instrumented file>
```

### 6. Build the HTML

Write one self-contained HTML file to `.claude/skills/trace-and-document-variables/logs/<slug>.html` and give the user the path. Self-contained means no external CSS/JS/font requests — inline everything so it opens straight from disk. This is a **utilitarian/reference treatment**, not editorial: density and scannability outrank flourish.

**Start from the worked example.** `reference/example-cheatsheet.html` next to this file is a complete, correct sheet — adapt it rather than designing a new one. Carry over its **color palette** (the `:root` token block with the light/dark/`[data-theme]` triple), its **row anatomy** (`.vrow` → `.name` + inline `<span class="loc">` gloss / `.sig` structure chips / `.data` monospace preview, with per-element `.n.pad` / `.n.neg` / `.n.hd` / `.n.kv` / `.n.sq` coloring), its **chrome** (`.legend` strip, `.dims` base-config grid, real-vs-padding `.bar`, `<details class="fn">` per function, `.tabs-bar` + `.tab-panel` with `switchTab`), and its **gloss density** (the `loc` after each name, the one-line `.note`, the scenario `.sub`, the bulleted `.callout` — the word budget below, in situ).

Structure:

1. **Masthead** — scope covered, exact command/config used, model/dtype.
2. **Legend** — the color key, once, small.
3. **Base config strip** — compact grid of step 2's fixed values; one glance re-anchors the mental model. Values only; a derived one may carry a few words (`num_queries_per_kv = 32/8 — GQA group size`).
4. **Discovered-flaw band** — only if the run surfaced something: a red/orange rectangle per finding, directly above the tab bar for run-wide findings. Omit entirely on a clean run.
5. **One tab per scenario/code path**, each headed by its few-word description and run facts, each holding one collapsible `<details>` per function, each holding a `<div class="vrow">` per variable. Every row is **name + few-word gloss**, then structure chips, then a value preview — one row layout, kind-appropriate chips:
   - tensor: shape → dtype → device → value preview
   - dict: `len` → key list (keys as chips), one indented sub-row per interesting value
   - list/tuple: `len` → element kind → preview (mark ragged lengths explicitly)
   - dataclass/object: type name → one indented sub-row per field, `None` fields greyed like padding
   - scalar/flag: type → value

   Nest sub-rows at most two levels deep; below that, summarize.
6. **Footer** — one line: env-gated prints, since reverted; line numbers refer to the clean file.

**Scenarios go in tabs, not stacked sections** — one tab bar, one panel per scenario, first active on load, so switching prefill↔decode is a click and the same variable sits in the same screen position across paths. Copy the `.tabs-bar` / `.tab-panel` CSS, the `<button onclick="switchTab('<name>', this)">` markup and `switchTab` from the example; button names and `tab-<name>` panel ids must agree. The base-config strip stays **above** the bar, since it holds for every scenario.

#### Flagging discovered flaws

When a run surfaces a genuine bug, caveat or danger, the sheet says so **loudly** — a bordered red or orange rectangle, not a grey `.note`. The user opens this file mid-debugging, and a flaw the run already proved is the highest-value thing on the page; burying it wastes the discovery. Two tiers:

- **`.flag` (red, badge `BUG`)** — the values show something is actually wrong: padding holding live data, an off-by-one, a dtype/device mismatch, a `None` the code dereferences, a branch firing when it shouldn't, a `FallbackWarning` on a hot path.
- **`.flag.caveat` (orange, badge `CAVEAT`)** — a sharp edge that will bite: an invariant holding only by coincidence, one tensor unpadded while neighbours are bucketed, a silent clamp, a shape that works only for this config.

Each rectangle carries, in order: the **badge**, a **one-line title** naming the flaw, the **location** (`file:line · tab`) right-aligned, one or two lines of what goes wrong and why it matters, and — the part that makes it trustworthy — a **`.ev` evidence block quoting the actual `### DBG[...]` lines**. Copy the `.flag` markup and the `--flaw` / `--caveat` token triples from the example; both tiers are worked there.

Placement follows scope: run-wide findings go **above the tab bar**, path-specific ones at the **top of that tab**; when a single variable's value *is* the evidence, also mark its row `.vrow.flagged` (or a `.flagmark` ⚠ by the name) to link rectangle and row.

Rules that keep the signal meaningful: **only from captured evidence** — if the run didn't demonstrate it, it's a hypothesis and belongs in chat, not a rectangle. **Omit the band on a clean run** — no "no issues found" placeholder; an empty flag area trains the user to ignore the color. **Don't fix what you find** — this skill documents; a fix would also invalidate the capture the sheet is built from. **Keep it scarce** — more than a handful of red rectangles and the color stops meaning anything; demote the rest to caveats or `.note` lines. **Unreached branches are not flaws** — report them in the masthead or chat, unless the capture positively shows why they can't fire.

Also tell the user, in chat, about every flag in the sheet — they may want to act now, and the sheet is a reference, not a notification.

#### Ranking a wide capture

A log holds more variables than a scannable sheet can give equal weight to. **Render all of them — drop nothing captured — but rank them** in three tiers:

- **Featured** — the variables the scope exists to explain: padded/bucketed tensors, ones whose shape changes across the branch, the dict or metadata object whose key set moves. Full treatment: inline gloss, all chips, colored preview, ratio bar, note where earned.
- **Compact** — real but supporting: intermediates, unchanged re-prints, scalars confirming a count. One line, chips only, gloss only where the name isn't self-evident; a tighter row class (smaller type, no preview or a truncated one).
- **Folded** — bulk that belongs in the record but not on screen: long unchanging lists, deep nesting, a variable repeated identically. Behind a nested `<details>` inside the function block, summarized by count (`+11 unchanged intermediates`).

Rank per tab, not globally — the same tensor may be featured on prefill and compact on decode. If a function's rows all look equally important, the ranking hasn't been done.

#### Word budget

- **Variable rows**: an inline gloss of **a few words** next to the name, in the reader's own vocabulary (`slot_mapping · flat token → physical KV slot`). A label, not a definition.
- **Shape dimensions**: name what a dimension counts when the chip doesn't make it obvious (`dim0 = flat tokens across batch`, `64 = b_seqs × num_kv_heads`) — once per novel shape, not on every repeat.
- **Padded / bucketed tensors**: say what the real prefix is and what the padding is (`5 real decode rows, 3 bucket-padding rows`). The single most valuable gloss in the sheet — never skip it.
- A `<div class="note">` is for a fact the structure and values cannot convey, capped at **one short line** ("padding clamps to last valid row, not zero", "key absent on the decode path"). Never a restatement of the structure.
- **Per function**: a few words naming what comes in and what goes out. No description of the steps between.
- Everywhere: **one line, no second sentence.** A gloss needing a second sentence is explaining rather than reminding — cut it.

#### Scenario descriptions

Every tab carries a **tiny high-level description** of what the case *is*, next to its title, so the user needn't reverse-engineer the setup from the shapes: a handful of words (`single sequence, full prefill` · `per-seq loop, ragged batch` · `4D full-batch decode matmul` · `3 decode + 1 mid-prefill + 1 full-prefill`), plus a one-line sub-label with concrete run facts (step number, `num_reqs`, per-seq query/kv lengths, the env var that selected the path). A genuinely complex scenario (unusual dispatch, non-obvious precondition, a bucket lattice you must know to read the numbers) may take a sentence or two of context, once, at the top of the tab. Complexity earns words; a straightforward prefill does not.

#### Design conventions

- **Color coding is semantic and consistent document-wide**: one color per *concept* (head/query dims, KV/block dims, sequence/batch dims), reused everywhere that concept appears — shape chips, dict keys, field names, inline numbers alike — so a dict keyed by layer and a tensor sized by layer read as the same concept. Padding gets its own neutral grey, used for every padded value in every tab, so "that's padding" reads on sight. A masked/sentinel value (e.g. fp16 min in an additive mask) gets its own fixed color.
- **Notation over prose, glossed**: `name` + few-word gloss / structure chips / monospace preview with per-element coloring. The gloss rides inline with the name (the example's `<span class="loc">`) so it reads as notation, not a paragraph beside it. Tensors get the densest treatment; containers get keys/lengths as chips with nested rows for the values that matter. Don't strip rows to bare notation either — an unlabelled `(64, 4, 1, 128)` costs the same re-derivation the sheet exists to prevent.
- **Real vs. padding ratio bars**: for bucketed/padded tensors, a tiny two-segment bar (solid = real, hatched = padding) makes the ratio legible without counting.
- **Collapsible `<details>` per function**, all open.
- **Theme-aware**: light tokens on bare `:root`, redefined under `@media (prefers-color-scheme: dark)` guarded by `:root:not([data-theme="light"])`, and again under `:root[data-theme="dark"]`. Never define a color only inside a media/attribute block.
- **Fonts**: system stacks only, since the file must open offline from disk — `ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace` for all notation (digits and brackets must line up), `system-ui, sans-serif` for labels. Never link web fonts.

## What NOT to do

- Don't write prose — structures and values are the content, glossed with a few words each. A paragraph explaining a reshape or a dict layout is noise the user skips past every time.
- Don't invent or extrapolate structures, values or flaws "because they're plausible" — every number traces back to a captured `### DBG[...]` block, and every flag quotes its evidence.
- Don't fix a bug you discover mid-capture; flag it, tell the user, leave the code alone.
- Don't pick base configs yourself, and don't pre-flight hardware/imports/device setup before the first run (step 2). Don't rerun end-to-end once per branch — patch and drive locally (step 3). But don't skip an *easily* reachable branch either: the bucketed-vs-per-seq (or equivalent) distinction is usually the point of the sheet.
- Don't be stingy with capture points — trim in the HTML (rank rows), not in the instrumentation — and don't render the result as a flat wall of equal-weight rows.
- Don't restrict the sheet to tensors, and don't dump a container in full: keys/fields/length plus a bounded per-value summary, two levels deep at most.
- Don't leave debug prints in the source file after capture (step 5), leave a tab unlabelled, stack scenarios as scrolling sections instead of tabs, or paste multi-fact callouts as one paragraph (one `<li>` per fact).
