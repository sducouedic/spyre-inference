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

"""Tests for the attention graph recorder.

CPU-only: these check that recording populates the kernel cache with exactly
the keys the bucketer enumerates, and that a subsequent dispatch reuses them
instead of growing the cache. The kernels run eagerly here (no Spyre), which
is enough to exercise dummy-arg construction and cache bookkeeping.
"""

from unittest.mock import MagicMock

import pytest
import torch

from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionImpl,
    SpyrePagedKVCache,
)
from spyre_inference.v1.attention.spyre_attn_bucketer import SpyreAttnBucketer

pytestmark = pytest.mark.attention

NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_SIZE = 64
BLOCK_SIZE = 64
NUM_PAGES = 8


@pytest.fixture()
def impl(default_vllm_config):
    impl = SpyreAttentionImpl(
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        scale=1.0 / (HEAD_SIZE**0.5),
        num_kv_heads=NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
    )
    # The fixture's bare CompilationConfig leaves mode unset, which resolves to
    # eager. Recording is a no-op there by design, so force the compiled path;
    # _maybe_compile is what actually decides whether Inductor is invoked.
    impl._compile_attn = True
    return impl


@pytest.fixture()
def kv_cache():
    shape = (NUM_PAGES, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    return SpyrePagedKVCache(
        k_pages=torch.zeros(shape, dtype=torch.float16),
        v_pages=torch.zeros(shape, dtype=torch.float16),
    )


def make_bucketer(max_model_len=256, max_num_batched_tokens=64, max_num_seqs=32):
    config = MagicMock()
    config.model_config.max_model_len = max_model_len
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    config.scheduler_config.max_num_seqs = max_num_seqs
    return SpyreAttnBucketer(config, BLOCK_SIZE)


class TestRecordGraphs:
    def test_populates_cache_with_enumerated_keys(self, impl, kv_cache):
        bucketer = make_bucketer()
        assert impl._attn_fns == {}

        recorded = impl.record_graphs(kv_cache, torch.device("cpu"), bucketer)

        assert recorded > 0
        expected = {v.key for v in bucketer.variants() if v.num_blocks <= NUM_PAGES}
        assert set(impl._attn_fns) == expected

    def test_dispatch_after_recording_does_not_grow_the_cache(self, impl, kv_cache):
        """The acceptance criterion: no request compiles a new variant."""
        bucketer = make_bucketer()
        impl.record_graphs(kv_cache, torch.device("cpu"), bucketer)
        snapshot = len(impl._attn_fns)

        for kv_len in (1, 60, 64, 200, 256):
            for query_len in (1, 5, 32, 64):
                if query_len > kv_len:
                    continue
                bucket = bucketer.dispatch(kv_len, query_len)
                assert bucket is not None
                if bucket.num_blocks > NUM_PAGES:
                    continue
                impl._get_attn_fn(
                    bucket.num_blocks,
                    bucket.padded_query_len,
                    store_mode=bucket.store_mode,
                    needs_gather=bucket.needs_gather,
                )

        assert len(impl._attn_fns) == snapshot

    def test_is_idempotent(self, impl, kv_cache):
        bucketer = make_bucketer()
        impl.record_graphs(kv_cache, torch.device("cpu"), bucketer)
        after_first = len(impl._attn_fns)

        assert impl.record_graphs(kv_cache, torch.device("cpu"), bucketer) == 0
        assert len(impl._attn_fns) == after_first

    def test_skips_variants_exceeding_the_page_allocation(self, impl, kv_cache):
        """A ladder sized from max_model_len can outrun a small KV cache."""
        bucketer = make_bucketer(max_model_len=4096)
        impl.record_graphs(kv_cache, torch.device("cpu"), bucketer)

        assert impl._attn_fns
        assert all(key[0] <= NUM_PAGES for key in impl._attn_fns)

    def test_eager_records_nothing(self, impl, kv_cache):
        impl._compile_attn = False
        assert impl.record_graphs(kv_cache, torch.device("cpu"), make_bucketer()) == 0
        assert impl._attn_fns == {}

    def test_a_failing_variant_does_not_abort_the_pass(self, impl, kv_cache, monkeypatch):
        """One bad variant must not take down engine startup."""
        bucketer = make_bucketer()
        calls = {"n": 0}
        real = impl._record_one

        def flaky(bucket, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("synthetic lowering failure")
            return real(bucket, *args, **kwargs)

        monkeypatch.setattr(impl, "_record_one", flaky)
        recorded = impl.record_graphs(kv_cache, torch.device("cpu"), bucketer)

        assert recorded == calls["n"] - 1
        # The failed key is left uncached, so it can still compile on first use.
        assert bucketer.variants()[0].key not in impl._attn_fns


class TestRecordKvUpdateGraphs:
    def test_records_each_token_count(self, impl, kv_cache):
        recorded = impl.record_kv_update_graphs(kv_cache, torch.device("cpu"), [4, 8, 8, 16])
        assert recorded == 3

    def test_skips_counts_beyond_the_slot_capacity(self, impl, kv_cache):
        num_slots = NUM_PAGES * BLOCK_SIZE
        assert impl.record_kv_update_graphs(kv_cache, torch.device("cpu"), [num_slots + 1]) == 0

    def test_eager_records_nothing(self, impl, kv_cache):
        impl._compile_attn = False
        assert impl.record_kv_update_graphs(kv_cache, torch.device("cpu"), [8]) == 0


class TestRecompileLimit:
    def test_limit_is_raised_during_recording_and_restored(self, impl, kv_cache):
        """Dynamo's accumulated limit is global, so a ladder wider than it would
        otherwise stop compiling partway through and fall back to eager."""
        bucketer = make_bucketer()
        before = torch._dynamo.config.accumulated_recompile_limit
        seen = []

        real = impl._record_one

        def spy(*args, **kwargs):
            seen.append(torch._dynamo.config.accumulated_recompile_limit)
            return real(*args, **kwargs)

        impl._record_one = spy
        impl.record_graphs(kv_cache, torch.device("cpu"), bucketer)

        assert seen and min(seen) >= len(bucketer.variants())
        assert torch._dynamo.config.accumulated_recompile_limit == before

    def test_limit_is_restored_even_when_recording_raises(self, impl, kv_cache, monkeypatch):
        before = torch._dynamo.config.accumulated_recompile_limit
        monkeypatch.setattr(
            impl, "_record_all", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        with pytest.raises(RuntimeError):
            impl.record_graphs(kv_cache, torch.device("cpu"), make_bucketer())
        assert torch._dynamo.config.accumulated_recompile_limit == before


class TestRecordBucketedDecodeGraphs:
    """``_decode_fns`` is keyed by (bucket_num_seqs, bucket_num_blocks), a
    contract independent of ``_attn_fns``: no real KV cache is needed to
    record it, only dummy tensors matching the kernel's shapes."""

    @pytest.fixture(autouse=True)
    def _enable_bucketed_decode(self, monkeypatch):
        monkeypatch.setenv("SPYRE_BUCKETED_DECODE", "1")

    def test_populates_cache_with_enumerated_keys(self, impl):
        bucketer = make_bucketer()
        assert impl._decode_fns == {}

        recorded = impl.record_bucketed_decode_graphs(bucketer, torch.device("cpu"))

        assert recorded > 0
        assert set(impl._decode_fns) == set(bucketer.decode_variants())

    def test_dispatch_after_recording_does_not_grow_the_cache(self, impl):
        """The acceptance criterion: no decode batch compiles a new variant."""
        bucketer = make_bucketer()
        impl.record_bucketed_decode_graphs(bucketer, torch.device("cpu"))
        snapshot = len(impl._decode_fns)

        for num_seqs in (4, 5, 8, 16, 32):
            for num_blocks in (1, 2, 3, 4, 8):
                b_seqs = bucketer._round_up(num_seqs, list(bucketer.decode_num_seqs_buckets))
                b_blocks = bucketer._round_up(num_blocks, list(bucketer.decode_num_blocks_buckets))
                if b_seqs is None or b_blocks is None:
                    continue
                impl._get_bucketed_decode_kernel(b_seqs, b_blocks)

        assert len(impl._decode_fns) == snapshot

    def test_is_idempotent(self, impl):
        bucketer = make_bucketer()
        impl.record_bucketed_decode_graphs(bucketer, torch.device("cpu"))
        after_first = len(impl._decode_fns)

        assert impl.record_bucketed_decode_graphs(bucketer, torch.device("cpu")) == 0
        assert len(impl._decode_fns) == after_first

    def test_eager_records_nothing(self, impl):
        impl._compile_attn = False
        assert impl.record_bucketed_decode_graphs(make_bucketer(), torch.device("cpu")) == 0
        assert impl._decode_fns == {}

    def test_disabled_flag_records_nothing(self, monkeypatch, impl):
        monkeypatch.setenv("SPYRE_BUCKETED_DECODE", "0")
        assert impl.record_bucketed_decode_graphs(make_bucketer(), torch.device("cpu")) == 0
        assert impl._decode_fns == {}

    def test_skips_when_alibi_slopes_set(self, impl, monkeypatch):
        impl.alibi_slopes = torch.zeros(NUM_KV_HEADS, impl.num_queries_per_kv, 1, 1)
        bucketer = make_bucketer()
        spy = MagicMock(wraps=bucketer.decode_variants)
        monkeypatch.setattr(bucketer, "decode_variants", spy)

        assert impl.record_bucketed_decode_graphs(bucketer, torch.device("cpu")) == 0
        assert impl._decode_fns == {}
        spy.assert_not_called()

    def test_skips_when_soft_cap_set(self, impl, monkeypatch):
        impl.logits_soft_cap = 30.0
        bucketer = make_bucketer()
        spy = MagicMock(wraps=bucketer.decode_variants)
        monkeypatch.setattr(bucketer, "decode_variants", spy)

        assert impl.record_bucketed_decode_graphs(bucketer, torch.device("cpu")) == 0
        assert impl._decode_fns == {}
        spy.assert_not_called()

    def test_a_failing_variant_does_not_abort_the_pass(self, impl, monkeypatch):
        """One bad variant must not take down engine startup."""
        bucketer = make_bucketer()
        calls = {"n": 0}
        real = impl._record_one_bucketed_decode

        def flaky(b_seqs, b_blocks, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("synthetic lowering failure")
            return real(b_seqs, b_blocks, *args, **kwargs)

        monkeypatch.setattr(impl, "_record_one_bucketed_decode", flaky)
        recorded = impl.record_bucketed_decode_graphs(bucketer, torch.device("cpu"))

        assert recorded == calls["n"] - 1
        # The failed key is left uncached, so it can still compile on first use.
        assert bucketer.decode_variants()[0] not in impl._decode_fns

    def test_dummy_args_match_kernel_shape_contract(self, impl):
        """Guards the dummy shapes in ``_record_one_bucketed_decode`` against
        drift from ``specialized_bucketed_decode_kernel``'s real contract: the
        cached kernel must also accept real (non-dummy) tensors of the same
        shapes without error."""
        b_seqs, b_blocks = 4, 2
        impl._record_one_bucketed_decode(b_seqs, b_blocks, BLOCK_SIZE, torch.device("cpu"))
        kernel = impl._get_bucketed_decode_kernel(b_seqs, b_blocks)

        num_kv_heads = impl.num_kv_heads
        num_queries_per_kv = impl.num_queries_per_kv
        head_size = impl.head_size
        q = torch.randn(
            b_seqs * num_kv_heads, num_queries_per_kv, 1, head_size, dtype=impl.model_dtype
        )
        k_list = [
            torch.randn(b_seqs * num_kv_heads, 1, BLOCK_SIZE, head_size, dtype=impl.model_dtype)
            for _ in range(b_blocks)
        ]
        v_list = [
            torch.randn(b_seqs * num_kv_heads, 1, BLOCK_SIZE, head_size, dtype=impl.model_dtype)
            for _ in range(b_blocks)
        ]
        mask_list = [
            torch.zeros(b_seqs * num_kv_heads, 1, 1, BLOCK_SIZE, dtype=impl.model_dtype)
            for _ in range(b_blocks)
        ]

        result = kernel(q, k_list, v_list, mask_list, impl.scale)

        assert result.shape == (b_seqs * num_kv_heads, num_queries_per_kv, head_size)


class TestBucketedDecodeRecompileLimit:
    def test_limit_is_raised_during_recording_and_restored(self, impl, monkeypatch):
        """Same global-limit caveat as record_graphs: a lattice wider than the
        accumulated limit would otherwise stop compiling partway through."""
        monkeypatch.setenv("SPYRE_BUCKETED_DECODE", "1")
        bucketer = make_bucketer()
        before = torch._dynamo.config.accumulated_recompile_limit
        seen = []

        real = impl._record_one_bucketed_decode

        def spy(*args, **kwargs):
            seen.append(torch._dynamo.config.accumulated_recompile_limit)
            return real(*args, **kwargs)

        impl._record_one_bucketed_decode = spy
        impl.record_bucketed_decode_graphs(bucketer, torch.device("cpu"))

        assert seen and min(seen) >= len(bucketer.decode_variants())
        assert torch._dynamo.config.accumulated_recompile_limit == before

    def test_limit_is_restored_even_when_recording_raises(self, impl, monkeypatch):
        monkeypatch.setenv("SPYRE_BUCKETED_DECODE", "1")
        before = torch._dynamo.config.accumulated_recompile_limit
        monkeypatch.setattr(
            impl,
            "_record_all_bucketed_decode",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with pytest.raises(RuntimeError):
            impl.record_bucketed_decode_graphs(make_bucketer(), torch.device("cpu"))
        assert torch._dynamo.config.accumulated_recompile_limit == before
