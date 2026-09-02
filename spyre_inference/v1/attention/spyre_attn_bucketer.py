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

"""Bucketer for the attention kernel's compile cache.

``SpyreAttentionImpl`` compiles one kernel per
``(num_blocks, padded_query_len, store_mode, needs_gather)`` key, lazily on
first use, which puts a full Inductor compile in the serving path. This module
enumerates the keys a run can reach so warmup can record them all up front.

Separate from ``SpyreShapeBucketer``: that one dispatches a single
``num_tokens`` int for the model graph, whereas an attention variant is 2-D
(a kv_len bucket and a query_len bucket) plus two discrete flags.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.logger import init_logger

from spyre_inference import envs

logger = init_logger(__name__)

# Store modes the kernel factory accepts, in the order forward() prefers them.
STORE_MODES = ("none", "copy", "index")


@dataclass(frozen=True)
class SpyreAttnBucket:
    """One recordable attention kernel variant.

    Fields mirror ``SpyreAttentionImpl._get_attn_fn``'s cache key exactly, so a
    recorded bucket and a runtime dispatch are the same tuple.
    """

    num_blocks: int
    padded_query_len: int
    store_mode: str
    needs_gather: bool

    @property
    def key(self) -> tuple[int, int, str, bool]:
        return (self.num_blocks, self.padded_query_len, self.store_mode, self.needs_gather)


def _parse_ladder(raw: str | None) -> list[int] | None:
    """Parse a comma-separated env-var ladder, or None when unset/empty."""
    if not raw:
        return None
    values = sorted({int(part) for part in raw.split(",") if part.strip()})
    if not values or values[0] < 1:
        raise ValueError(f"ladder entries must be >= 1, got {raw!r}")
    return values


def _ladder(step: int, limit: int, dense_steps: int = 1) -> list[int]:
    """Bucket ladder: ``dense_steps`` multiples of ``step``, then doubling, capped at ``limit``.

    Doubling above the dense range is what keeps warmup affordable: the recorded
    set is a product of both axes, so a ladder of every multiple of ``step`` up
    to a 32k context is tens of thousands of variants. The dense head bounds the
    round-up waste on the axis that carries real compute; above it each bucket is
    at most 2x the one below, which the mask absorbs as ordinary padding.
    ``SPYRE_ATTN_KV_BUCKETS`` / ``SPYRE_ATTN_QUERY_BUCKETS`` override it.
    """
    if limit < step:
        return [step]
    out = list(range(step, min(step * dense_steps, limit) + 1, step))
    v = out[-1] * 2
    while v < limit:
        out.append(v)
        v *= 2
    if out[-1] < limit:
        out.append(limit)
    return out


class SpyreAttnBucketer:
    """Enumerates attention variants to record, and dispatches to them.

    Both ladders round *up*: a runtime length lands on the smallest recorded
    bucket that fits it, matching ``SpyreShapeBucketer.find_bucket``. Over-max
    returns None, and the caller falls back to compiling on demand.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        block_size = vllm_config.cache_config.block_size
        self.block_size = block_size
        max_model_len = vllm_config.model_config.max_model_len
        max_batched = vllm_config.scheduler_config.max_num_batched_tokens
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs

        # Imported at call time, not module scope: spyre_attn imports this
        # module, so a top-level import back into it would be circular.
        from spyre_inference.v1.attention.backends.spyre_attn import (
            KV_LENGTH_ALIGNMENT,
            QUERY_CHUNK_SIZE,
            _powers_of_two_up_to,
        )

        kv = _parse_ladder(envs.SPYRE_ATTN_KV_BUCKETS)
        if kv is None:
            kv = _ladder(KV_LENGTH_ALIGNMENT, max_model_len)
        self._kv_buckets: list[int] = kv

        query = _parse_ladder(envs.SPYRE_ATTN_QUERY_BUCKETS)
        if query is None:
            # The leading 1 is the decode-only batch, which build() exempts from
            # query padding entirely. Denser than the kv ladder on purpose: the
            # query axis carries the matmul cost, so a pure doubling ladder would
            # round a 513-token prefill up to 1024 and compute ~2x the FLOPs.
            query = [1] + _ladder(QUERY_CHUNK_SIZE, max_batched, dense_steps=8)
        self._query_buckets: list[int] = query

        # num_blocks is what the kernel specializes on. Deriving it from the kv
        # ladder rather than enumerating every integer up to max_model_len /
        # block_size is what keeps the recorded set small: one block count per
        # kv bucket, not one per possible block count.
        self._num_blocks_buckets: list[int] = sorted(
            {(kv + block_size - 1) // block_size for kv in self._kv_buckets}
        )

        # Separate lattice for the bucketed multi-seq decode kernel
        # (_get_bucketed_decode_kernel): SpyreAttentionMetadataBuilder derives
        # these the same way, from max_num_seqs and max_model_len / block_size.
        # Kept distinct from the varlen ladders above — the decode kernel
        # specializes on (num_seqs, num_blocks) directly, not on a kv_len bucket.
        max_num_blocks_per_seq = (max_model_len + block_size - 1) // block_size
        self._decode_num_seqs_buckets: tuple[int, ...] = _powers_of_two_up_to(max_num_seqs)
        self._decode_num_blocks_buckets: tuple[int, ...] = _powers_of_two_up_to(
            max_num_blocks_per_seq
        )

        self._is_warmed_up = False

        logger.info(
            "SpyreAttnBucketer: %d kv buckets [%d..%d], %d query buckets [%d..%d], "
            "max num_blocks=%d",
            len(self._kv_buckets),
            self._kv_buckets[0],
            self._kv_buckets[-1],
            len(self._query_buckets),
            self._query_buckets[0],
            self._query_buckets[-1],
            self._num_blocks_buckets[-1],
        )

    @property
    def kv_buckets(self) -> list[int]:
        return self._kv_buckets

    @property
    def query_buckets(self) -> list[int]:
        return self._query_buckets

    @property
    def num_blocks_buckets(self) -> list[int]:
        return self._num_blocks_buckets

    @property
    def decode_num_seqs_buckets(self) -> tuple[int, ...]:
        return self._decode_num_seqs_buckets

    @property
    def decode_num_blocks_buckets(self) -> tuple[int, ...]:
        return self._decode_num_blocks_buckets

    @property
    def is_warmed_up(self) -> bool:
        return self._is_warmed_up

    def mark_warmed_up(self) -> None:
        self._is_warmed_up = True

    def find_kv_bucket(self, kv_len: int) -> int | None:
        return self._round_up(kv_len, self._kv_buckets)

    def find_query_bucket(self, query_len: int) -> int | None:
        return self._round_up(query_len, self._query_buckets)

    @staticmethod
    def _round_up(n: int, buckets: list[int]) -> int | None:
        idx = bisect.bisect_left(buckets, n)
        return buckets[idx] if idx < len(buckets) else None

    def dispatch(self, kv_len: int, query_len: int) -> SpyreAttnBucket | None:
        """Map a concrete (kv_len, query_len) onto a recorded variant.

        ``store_mode``/``needs_gather`` are decided per call by ``forward``
        against buffers this class cannot see, so the returned bucket carries
        the reachable-worst-case combination: an indexed store behind a gather.
        """
        padded_query_len = self.find_query_bucket(query_len)
        kv_bucket = self.find_kv_bucket(kv_len)
        if padded_query_len is None or kv_bucket is None:
            return None
        num_blocks = (kv_bucket + self.block_size - 1) // self.block_size
        return SpyreAttnBucket(
            num_blocks=num_blocks,
            padded_query_len=padded_query_len,
            store_mode="index",
            needs_gather=True,
        )

    def variants(self) -> list[SpyreAttnBucket]:
        """Every variant worth recording, largest first.

        The two size axes are not independent: a sequence holding
        ``query_len`` new tokens has ``kv_len >= query_len``, so a query bucket
        can only appear with block counts that can hold it. Taking the full
        cross product instead would record hundreds of thousands of unreachable
        variants at a 32k context.

        The flag axes are pruned against ``forward``'s dispatch logic:
        - ``needs_gather=False`` requires the sequence to own the whole query
          buffer from row 0, which is exactly the single-sequence batch. That
          also makes ``output.shape[0] == query_len``, so the store is a
          ``copy_``; ``"index"`` is unreachable there.
        - ``needs_gather=True`` means the sequence owns a strict slice, so
          ``output.shape[0] != query_len`` and ``"copy"`` is unreachable.
        - ``store_mode="none"`` is the un-fused fallback (``fused_store_ok``
          false, e.g. a misaligned per-layer output buffer) and pairs with
          either gather setting.
        """
        flag_pairs = (("index", True), ("none", True), ("copy", False), ("none", False))
        out: list[SpyreAttnBucket] = []
        for num_blocks in sorted(self._num_blocks_buckets, reverse=True):
            max_query_here = num_blocks * self.block_size
            for padded_query_len in sorted(self._query_buckets, reverse=True):
                if padded_query_len > max_query_here:
                    continue
                for store_mode, needs_gather in flag_pairs:
                    out.append(
                        SpyreAttnBucket(
                            num_blocks=num_blocks,
                            padded_query_len=padded_query_len,
                            store_mode=store_mode,
                            needs_gather=needs_gather,
                        )
                    )
        return out

    def decode_variants(self) -> list[tuple[int, int]]:
        """Every ``(bucket_num_seqs, bucket_num_blocks)`` pair the bucketed
        decode kernel (``_get_bucketed_decode_kernel``) can dispatch to,
        largest first. Below ``_MIN_SEQS_BUCKET`` the per-seq loop is always
        used instead, so those pairs are never recorded.
        """
        from spyre_inference.v1.attention.backends.spyre_attn import _MIN_SEQS_BUCKET

        return [
            (num_seqs, num_blocks)
            for num_seqs in sorted(self._decode_num_seqs_buckets, reverse=True)
            if num_seqs >= _MIN_SEQS_BUCKET
            for num_blocks in sorted(self._decode_num_blocks_buckets, reverse=True)
        ]
