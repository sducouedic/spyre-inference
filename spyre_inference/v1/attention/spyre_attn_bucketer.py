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

Vocabulary: a *bucket* is one padded size a runtime length rounds up onto; the
sorted list of them for one axis is that axis's *buckets*; the spacing between
consecutive buckets is the *bucket step*.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from weakref import ReferenceType, ref

from vllm.config import VllmConfig
from vllm.logger import init_logger

from spyre_inference import envs

logger = init_logger(__name__)

# Store modes the kernel factory accepts, in the order forward() prefers them.
STORE_MODES = ("none", "copy", "index")

# Spacing of the default query buckets above the decode bucket, capped against
# max_num_batched_tokens. Every non-decode batch pads its query length up to a
# multiple of this.
_DEFAULT_QUERY_BUCKET_STEP = 512


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


def _parse_buckets(raw: str | None) -> list[int] | None:
    """Parse a comma-separated env-var bucket list, or None when unset/empty."""
    if not raw:
        return None
    values = sorted({int(part) for part in raw.split(",") if part.strip()})
    if not values or values[0] < 1:
        raise ValueError(f"bucket entries must be >= 1, got {raw!r}")
    return values


class SpyreAttnBucketer:
    """Enumerates attention variants to record, and dispatches to them.

    Both axes round *up*: a runtime length lands on the smallest recorded
    bucket that fits it, matching ``SpyreShapeBucketer.find_bucket``. Over-max
    returns None, and the caller falls back to compiling on demand.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        block_size = vllm_config.cache_config.block_size
        self.block_size = block_size
        max_model_len = vllm_config.model_config.max_model_len
        max_batched = vllm_config.scheduler_config.max_num_batched_tokens

        # Imported at call time, not module scope: spyre_attn imports this
        # module, so a top-level import back into it would be circular.
        from spyre_inference.v1.attention.backends.spyre_attn import _powers_of_two_up_to

        if block_size & (block_size - 1):
            # Not fatal: _powers_of_two_up_to rounds the start up to a power of
            # two, so the buckets are still correct -- just coarser at the bottom
            # than a power-of-two block_size would give. Worth saying out loud
            # because the platform only forces a multiple of 64 (see
            # SpyrePlatform.check_and_update_config), so this is reachable.
            logger.warning(
                "block_size=%d is not a power of two; the smallest KV bucket is the next "
                "power of two instead, making it larger than one block. Prefer a "
                "power-of-two block_size.",
                block_size,
            )

        kv = _parse_buckets(envs.SPYRE_ATTN_KV_BUCKETS)
        if kv is None:
            # Powers of two from block_size up to (and including) max_model_len.
            # Doubling keeps warmup affordable: the recorded set is a product of
            # both axes, so a bucket per KV token up to a 32k context would be tens
            # of thousands of variants. Each bucket is at most 2x the one below,
            # which the mask absorbs as ordinary padding.
            #
            # Starting at block_size rather than 1 drops buckets that buy nothing:
            # num_blocks (below) is ceil(kv / block_size), so every kv <= block_size
            # collapses to num_blocks == 1 and would be deduped away anyway.
            kv = list(_powers_of_two_up_to(max_model_len, start=block_size))
        self._kv_buckets: list[int] = kv

        query = _parse_buckets(envs.SPYRE_ATTN_QUERY_BUCKETS)
        if query is None:
            # [1] then multiples of a fixed step up to (and including)
            # max_num_batched_tokens. Coarse bucketing for now: a prefill pays
            # padding up to the next bucket, which the mask discards.
            #
            # 1 is the decode-only batch, which build() exempts from query padding
            # entirely. The step is capped at 512 so a large
            # max_num_batched_tokens does not make the single non-decode bucket
            # enormous; the multiples then carry the buckets up to the top, so a
            # query_len above the step still finds a bucket rather than falling
            # off the end into a serving-path compile.
            step = min(_DEFAULT_QUERY_BUCKET_STEP, max_batched)
            query = sorted({1, *range(step, max_batched + 1, step), max_batched})
        self._query_buckets: list[int] = query

        # num_blocks is what the kernel specializes on. Deriving it from the kv
        # buckets rather than enumerating every integer up to max_model_len /
        # block_size is what keeps the recorded set small: one block count per
        # kv bucket, not one per possible block count.
        self._num_blocks_buckets: list[int] = sorted(
            {(kv + block_size - 1) // block_size for kv in self._kv_buckets}
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
        cross product instead would record many unreachable variants at a long
        context.

        The bound is on the *smallest real* query_len that reaches a bucket, not
        on the bucket itself: a bucket is a padded value, so a 2-token query on a
        1-block sequence legitimately dispatches to the 512 bucket. Bounding by
        the bucket would prune exactly that variant and put a compile back in the
        serving path.

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
        # Smallest real query_len that rounds up to each bucket: one past the
        # bucket below it (and 1 for the smallest bucket).
        ascending = sorted(self._query_buckets)
        min_real_query = {
            bucket: (ascending[i - 1] + 1 if i else 1) for i, bucket in enumerate(ascending)
        }
        out: list[SpyreAttnBucket] = []
        for num_blocks in sorted(self._num_blocks_buckets, reverse=True):
            max_query_here = num_blocks * self.block_size
            for padded_query_len in sorted(self._query_buckets, reverse=True):
                if min_real_query[padded_query_len] > max_query_here:
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


# One bucketer per VllmConfig, shared by every consumer. The metadata builder
# rounds each sequence's num_blocks onto these buckets and the warmup recorder
# compiles exactly those buckets, so the two must agree; sharing one instance
# makes that hold by construction rather than by an after-the-fact assertion.
#
# Keyed by config identity because VllmConfig is an eq=True dataclass and so
# unhashable, and because two configs that compare equal are still separate
# engines. A weakref callback drops the entry when the config goes away, so the
# many short-lived configs a test session builds are not pinned here.
_bucketers: dict[int, SpyreAttnBucketer] = {}
_bucketer_refs: dict[int, ReferenceType[VllmConfig]] = {}
# Configs that cannot be weak-referenced, kept alive so their id() stays theirs.
_bucketer_pins: dict[int, VllmConfig] = {}


def get_attn_bucketer(vllm_config: VllmConfig) -> SpyreAttnBucketer:
    """Return the one ``SpyreAttnBucketer`` belonging to ``vllm_config``.

    Built on first call and memoized after: the buckets are a pure function of
    the config, so every caller shares the same object -- and the same
    ``is_warmed_up`` flag, which is what lets a reader downstream of warmup tell
    whether the recorder has run.
    """
    key = id(vllm_config)
    bucketer = _bucketers.get(key)
    if bucketer is not None:
        return bucketer
    bucketer = SpyreAttnBucketer(vllm_config)
    _bucketers[key] = bucketer
    try:
        _bucketer_refs[key] = ref(vllm_config, lambda _r, key=key: _forget(key))
    except TypeError:
        # Not weak-referenceable (some mocks and __slots__ stand-ins in tests).
        # Pin the config instead: without a strong reference its id() could be
        # handed to a later, unrelated config, which would then be served this
        # bucketer. Leaks the entry for the process, which only tests hit.
        _bucketer_pins[key] = vllm_config
    return bucketer


def _forget(key: int) -> None:
    _bucketers.pop(key, None)
    _bucketer_refs.pop(key, None)
