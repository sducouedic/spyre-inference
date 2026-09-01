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

"""Unit tests for SpyreAttnBucketer. No hardware required."""

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from spyre_inference import envs
from spyre_inference.v1.attention.spyre_attn_bucketer import (
    SpyreAttnBucket,
    SpyreAttnBucketer,
    _ladder,
    _parse_ladder,
)

BLOCK_SIZE = 64


def make_config(max_model_len=2048, max_num_batched_tokens=512, max_num_seqs=32):
    config = MagicMock()
    config.model_config.max_model_len = max_model_len
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    config.scheduler_config.max_num_seqs = max_num_seqs
    return config


@pytest.fixture()
def bucketer():
    return SpyreAttnBucketer(make_config(), BLOCK_SIZE)


@pytest.fixture(autouse=True)
def _clear_env_cache(monkeypatch):
    """envs caches on first read, so each test must start from a clean slate."""
    envs.clear_env_cache()
    yield
    envs.clear_env_cache()


class TestLadders:
    def test_kv_ladder_is_geometric_from_alignment(self, bucketer):
        assert bucketer.kv_buckets == [256, 512, 1024, 2048]

    def test_query_ladder_leads_with_decode_case(self, bucketer):
        assert bucketer.query_buckets[0] == 1
        # Dense to 8*32, then doubling: the query axis carries the matmul cost.
        assert bucketer.query_buckets == [1, 32, 64, 96, 128, 160, 192, 224, 256, 512]

    def test_ladder_doubles_by_default(self):
        assert _ladder(256, 2048) == [256, 512, 1024, 2048]

    def test_ladder_includes_non_power_of_two_limit(self):
        assert _ladder(256, 3000) == [256, 512, 1024, 2048, 3000]

    def test_ladder_dense_head_then_doubling(self):
        assert _ladder(32, 2048, dense_steps=8) == [
            32,
            64,
            96,
            128,
            160,
            192,
            224,
            256,
            512,
            1024,
            2048,
        ]

    def test_dense_head_bounds_round_up_waste(self):
        """A pure doubling ladder would round a 129-token prefill up to 256."""
        assert 160 in _ladder(32, 2048, dense_steps=8)

    def test_ladder_stops_at_limit(self):
        assert _ladder(32, 64, dense_steps=8) == [32, 64]
        assert _ladder(32, 100, dense_steps=8) == [32, 64, 96, 100]

    def test_ladder_limit_below_step(self):
        assert _ladder(256, 128) == [256]
        assert _ladder(32, 16, dense_steps=8) == [32]


class TestFindBucket:
    def test_exact_match(self, bucketer):
        assert bucketer.find_kv_bucket(512) == 512
        assert bucketer.find_query_bucket(32) == 32

    def test_rounds_up(self, bucketer):
        assert bucketer.find_kv_bucket(257) == 512
        assert bucketer.find_query_bucket(33) == 64

    def test_query_len_one_maps_to_decode_bucket(self, bucketer):
        assert bucketer.find_query_bucket(1) == 1

    def test_query_round_up_stays_within_one_chunk_in_dense_range(self, bucketer):
        assert bucketer.find_query_bucket(129) == 160

    def test_exceeds_max_returns_none(self, bucketer):
        assert bucketer.find_kv_bucket(2049) is None
        assert bucketer.find_query_bucket(513) is None


class TestDispatch:
    def test_num_blocks_derived_from_kv_tier(self, bucketer):
        b = bucketer.dispatch(kv_len=300, query_len=8)
        assert b is not None
        # 300 rounds to the 512 tier, which is 8 blocks of 64.
        assert b.num_blocks == 512 // BLOCK_SIZE
        assert b.padded_query_len == 32

    def test_decode_dispatch_keeps_query_len_one(self, bucketer):
        b = bucketer.dispatch(kv_len=1000, query_len=1)
        assert b is not None
        assert b.padded_query_len == 1

    def test_over_max_on_either_axis_returns_none(self, bucketer):
        assert bucketer.dispatch(kv_len=99999, query_len=32) is None
        assert bucketer.dispatch(kv_len=256, query_len=99999) is None

    def test_descriptor_is_frozen(self, bucketer):
        b = bucketer.dispatch(kv_len=256, query_len=1)
        assert b is not None
        with pytest.raises(FrozenInstanceError):
            b.num_blocks = 10

    @pytest.mark.parametrize("kv_len", [1, 64, 255, 256, 257, 1024, 2048])
    @pytest.mark.parametrize("query_len", [1, 2, 31, 32, 33, 200, 512])
    def test_dispatch_always_lands_on_a_recorded_variant(self, bucketer, kv_len, query_len):
        """The whole point of recording: no runtime batch may miss the cache."""
        if query_len > kv_len:
            pytest.skip("a sequence cannot have more new tokens than total KV")
        b = bucketer.dispatch(kv_len, query_len)
        assert b is not None
        assert b.key in {v.key for v in bucketer.variants()}


class TestVariants:
    def test_no_duplicates(self, bucketer):
        variants = bucketer.variants()
        assert len(variants) == len({v.key for v in variants})

    def test_stable_across_calls(self, bucketer):
        assert [v.key for v in bucketer.variants()] == [v.key for v in bucketer.variants()]

    def test_largest_first(self, bucketer):
        variants = bucketer.variants()
        assert variants[0].num_blocks == max(v.num_blocks for v in variants)

    def test_prunes_unreachable_flag_combinations(self, bucketer):
        keys = {(v.store_mode, v.needs_gather) for v in bucketer.variants()}
        # "copy" implies the sequence owns every output row, which implies no gather.
        assert ("copy", True) not in keys
        # "index" is only needed when the destination is a strict superset of rows.
        assert ("index", False) not in keys

    def test_prunes_query_chunks_larger_than_the_block_span(self, bucketer):
        for v in bucketer.variants():
            assert v.padded_query_len <= v.num_blocks * BLOCK_SIZE

    def test_count_stays_tractable_at_long_context(self):
        """A dense ladder here would be tens of thousands of Inductor compiles."""
        b = SpyreAttnBucketer(make_config(32768, 2048), BLOCK_SIZE)
        assert len(b.variants()) < 500


class TestEnvOverride:
    def test_kv_ladder_override(self, monkeypatch):
        monkeypatch.setenv("SPYRE_ATTN_KV_BUCKETS", "128,512,4096")
        envs.clear_env_cache()
        b = SpyreAttnBucketer(make_config(), BLOCK_SIZE)
        assert b.kv_buckets == [128, 512, 4096]

    def test_query_ladder_override_is_sorted_and_deduped(self, monkeypatch):
        monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "64,1,16,64")
        envs.clear_env_cache()
        b = SpyreAttnBucketer(make_config(), BLOCK_SIZE)
        assert b.query_buckets == [1, 16, 64]

    def test_parse_ladder_rejects_non_positive(self):
        with pytest.raises(ValueError):
            _parse_ladder("0,32")

    def test_parse_ladder_empty_is_none(self):
        assert _parse_ladder("") is None
        assert _parse_ladder(None) is None


class TestBucketerState:
    def test_initial_state_not_warmed_up(self, bucketer):
        assert not bucketer.is_warmed_up

    def test_mark_warmed_up(self, bucketer):
        bucketer.mark_warmed_up()
        assert bucketer.is_warmed_up


class TestBucketKey:
    def test_key_matches_attn_fn_cache_tuple(self):
        b = SpyreAttnBucket(
            num_blocks=4, padded_query_len=32, store_mode="index", needs_gather=True
        )
        assert b.key == (4, 32, "index", True)
