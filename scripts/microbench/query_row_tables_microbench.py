#!/usr/bin/env python3
# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wall-clock micro-benchmark for `_build_query_row_tables`.

Host-side metadata prep, called once per forward on the varlen path. It is a Python
loop over sequences doing small CPU tensor fills plus one `convert()` per sequence, so
the cost scales with `num_seqs` and (weakly) with the aligned query widths -- not with
KV length. Wall clock is the signal here, not device time: there is no kernel to
attribute, and on `--device spyre` the per-sequence H2D transfer is the dominant term.

    .venv/bin/python3 scripts/microbench/query_row_tables_microbench.py
    .venv/bin/python3 scripts/microbench/query_row_tables_microbench.py --device spyre
    .venv/bin/python3 scripts/microbench/query_row_tables_microbench.py \
        --shape decode:64 --shape prefill:8x512 --iterations 200 --csv out.csv

Shape syntax (repeatable `--shape`, `name:spec`):

    decode:64          64 sequences, query_len 1 each (batched decode)
    prefill:8x512      8 sequences, query_len 512 each
    mixed:4x512+60     4 prefills of 512 plus 60 decodes

Omit `--shape` for the built-in sweep.
"""

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_PLUGINS", "spyre_inference")
for _k, _v in (
    ("RANK", "0"),
    ("LOCAL_RANK", "0"),
    ("WORLD_SIZE", "1"),
    ("LOCAL_WORLD_SIZE", "1"),
    ("MASTER_ADDR", "127.0.0.1"),
    ("MASTER_PORT", "29500"),
):
    os.environ.setdefault(_k, _v)

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spyre_inference.v1.attention.backends.spyre_attn import (  # noqa: E402
    _build_query_row_tables,
    _stick_aligned_len,
)

# build() derives aligned_query_lens from the bucketer's query buckets, whose default
# is {1} plus multiples of 512 up to max_num_batched_tokens (_DEFAULT_QUERY_BUCKET_STEP).
# Widths drive the per-row fill, so the benchmark reproduces that rule rather than
# inventing its own; --query-buckets / SPYRE_ATTN_QUERY_BUCKETS override it exactly as
# the engine's env var does.
DEFAULT_QUERY_BUCKET_STEP = 512
DEFAULT_MAX_BATCHED_TOKENS = 2048

DEFAULT_SWEEP = [
    "decode:1",
    "decode:8",
    "decode:32",
    "decode:64",
    "decode:128",
    "decode:256",
    "prefill:1x512",
    "prefill:1x2048",
    "prefill:4x512",
    "prefill:8x512",
    "mixed:1x512+63",
    "mixed:4x512+60",
]


def default_query_buckets(max_batched: int = DEFAULT_MAX_BATCHED_TOKENS) -> list[int]:
    """SpyreAttnBucketer's default query buckets for a given max_num_batched_tokens."""
    step = min(DEFAULT_QUERY_BUCKET_STEP, max_batched)
    return sorted({1, *range(step, max_batched + 1, step), max_batched})


class FakeMetadata:
    """The three fields `_build_query_row_tables` reads.

    Standing in for SpyreAttentionMetadata keeps the benchmark off the mask-tile build
    and the vLLM config context, neither of which this function touches. The
    aligned_query_lens rule is build()'s: query_len <= 1 stays 1 (decode is exempt from
    query padding), anything longer rounds up to a query bucket.
    """

    def __init__(self, query_lens: list[int], query_buckets: list[int]):
        self.num_seqs = len(query_lens)
        self.query_start_loc = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
            dim=0, dtype=torch.int32
        )
        self.aligned_query_lens = [_align(q, query_buckets) for q in query_lens]


def _align(query_len: int, query_buckets: list[int]) -> int:
    if query_len <= 1:
        return 1
    for bucket in query_buckets:
        if query_len <= bucket:
            return bucket
    raise ValueError(
        f"query_len {query_len} exceeds the largest query bucket {query_buckets[-1]}; "
        f"raise --max-batched-tokens or pass --query-buckets."
    )


def parse_shape(spec: str) -> tuple[str, list[int]]:
    """`name:4x512+60` -> ("name", [512, 512, 512, 512, 1, 1, ...])."""
    name, _, body = spec.partition(":")
    if not body:
        raise ValueError(f"shape {spec!r} needs a name: prefix")
    query_lens: list[int] = []
    for term in body.split("+"):
        count, _, length = term.partition("x")
        if length:
            query_lens += [int(length)] * int(count)
        else:
            query_lens += [1] * int(count)
    if not query_lens:
        raise ValueError(f"shape {spec!r} describes no sequences")
    return name, query_lens


def bench_one(
    metadata: FakeMetadata, device: torch.device, iterations: int, warmup: int
) -> list[float]:
    for _ in range(warmup):
        _build_query_row_tables(metadata, device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        tables = _build_query_row_tables(metadata, device)
        samples.append((time.perf_counter() - start) * 1e3)
        del tables
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--shape",
        action="append",
        default=None,
        help="name:spec, repeatable (e.g. mixed:4x512+60). Default: built-in sweep.",
    )
    parser.add_argument("--device", default="cpu", help="cpu (default) or spyre")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10, help="Unmeasured calls per shape.")
    parser.add_argument("--csv", type=Path, default=None, help="Also write rows here.")
    parser.add_argument(
        "--query-buckets",
        default=os.getenv("SPYRE_ATTN_QUERY_BUCKETS"),
        help="Comma-separated query buckets, as SPYRE_ATTN_QUERY_BUCKETS. Default: derived.",
    )
    parser.add_argument(
        "--max-batched-tokens",
        type=int,
        default=DEFAULT_MAX_BATCHED_TOKENS,
        help="max_num_batched_tokens the default buckets are derived from.",
    )
    args = parser.parse_args()

    if args.query_buckets:
        query_buckets = sorted({int(b) for b in args.query_buckets.split(",") if b.strip()})
    else:
        query_buckets = default_query_buckets(args.max_batched_tokens)

    device = torch.device(args.device)
    if device.type == "spyre":
        import torch_spyre  # noqa: F401

        from spyre_inference.custom_ops import register_all

        register_all()
    torch.set_default_device("cpu")

    shapes = [parse_shape(s) for s in (args.shape or DEFAULT_SWEEP)]
    header = ["shape", "num_seqs", "total_rows", "median_ms", "min_ms", "max_ms", "us_per_seq"]
    print(
        f"device={device}  iterations={args.iterations}  warmup={args.warmup}  "
        f"query_buckets={query_buckets}"
    )
    print("\t".join(header))

    rows = []
    for name, query_lens in shapes:
        metadata = FakeMetadata(query_lens, query_buckets)
        total_rows = sum(_stick_aligned_len(a) for a in metadata.aligned_query_lens)
        samples = bench_one(metadata, device, args.iterations, args.warmup)
        median = statistics.median(samples)
        row = [
            name,
            metadata.num_seqs,
            total_rows,
            f"{median:.4f}",
            f"{min(samples):.4f}",
            f"{max(samples):.4f}",
            f"{median * 1e3 / metadata.num_seqs:.2f}",
        ]
        rows.append(row)
        print("\t".join(str(c) for c in row))

    if args.csv:
        with args.csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
