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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from spyre_inference import envs
from spyre_inference.v1.attention.backends.spyre_attn import _powers_of_two_up_to
from spyre_inference.v1.attention.spyre_attn_bucketer import (
    SpyreAttnBucket,
    SpyreAttnBucketer,
    _parse_buckets,
)

BLOCK_SIZE = 64


def make_config(max_model_len=2048, max_num_batched_tokens=512, block_size=BLOCK_SIZE):
    config = MagicMock()
    config.cache_config.block_size = block_size
    config.model_config.max_model_len = max_model_len
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    return config


def _list_pow2(limit: int, start: int = 1) -> list[int]:
    """[start, 2*start, ..., limit], the buckets the kv axis defaults to."""
    return list(_powers_of_two_up_to(limit, start=start))


@pytest.fixture()
def bucketer():
    return SpyreAttnBucketer(make_config())


@pytest.fixture(autouse=True)
def _clear_env_cache(monkeypatch):
    """envs caches on first read, so each test must start from a clean slate."""
    envs.clear_env_cache()
    yield
    envs.clear_env_cache()


class TestBuckets:
    def test_kv_buckets_are_powers_of_two_to_max_model_len(self, bucketer):
        assert bucketer.kv_buckets == _list_pow2(2048, start=BLOCK_SIZE)
        assert bucketer.kv_buckets[-1] == 2048

    def test_kv_buckets_start_at_block_size(self, bucketer):
        """Buckets below block_size all collapse to num_blocks == 1, so the
        smallest bucket is block_size rather than 1."""
        assert bucketer.kv_buckets[0] == BLOCK_SIZE

    @pytest.mark.parametrize("block_size", [64, 128, 256])
    def test_kv_buckets_start_tracks_block_size(self, block_size):
        b = SpyreAttnBucketer(make_config(max_model_len=4096, block_size=block_size))
        assert b.kv_buckets == _list_pow2(4096, start=block_size)

    def test_kv_buckets_round_non_power_of_two_block_size_up(self):
        """block_size is only forced to a multiple of 64 by the platform, so a
        non-power-of-two value is reachable; the buckets stay a clean doubling
        sequence by starting at the next power of two."""
        b = SpyreAttnBucketer(make_config(max_model_len=4096, block_size=192))
        assert b.kv_buckets == [256, 512, 1024, 2048, 4096]

    def test_query_buckets_lead_with_decode_case(self, bucketer):
        assert bucketer.query_buckets[0] == 1
        assert bucketer.query_buckets == [1, 512]

    def test_query_buckets_are_multiples_of_the_step(self):
        b = SpyreAttnBucketer(make_config(max_num_batched_tokens=2048))
        assert b.query_buckets == [1, 512, 1024, 1536, 2048]

    def test_query_bucket_step_capped_by_max_num_batched_tokens(self):
        b = SpyreAttnBucketer(make_config(max_num_batched_tokens=300))
        assert b.query_buckets == [1, 300]

    def test_buckets_include_non_power_of_two_limit(self):
        b = SpyreAttnBucketer(make_config(max_model_len=3000, max_num_batched_tokens=100))
        assert b.kv_buckets == _list_pow2(2048, start=BLOCK_SIZE) + [3000]
        assert b.query_buckets == [1, 100]

    def test_largest_bucket_is_always_the_limit(self):
        for limit in (1, 2, 3, 64, 100, 4096, 32768):
            b = SpyreAttnBucketer(make_config(max_model_len=limit, max_num_batched_tokens=limit))
            assert b.kv_buckets[-1] == limit
            assert b.query_buckets[-1] == limit

    def test_buckets_have_no_duplicates(self):
        for limit in (1, 2, 3, 64, 100, 512, 513, 4096, 32768):
            b = SpyreAttnBucketer(make_config(max_model_len=limit, max_num_batched_tokens=limit))
            assert b.kv_buckets == sorted(set(b.kv_buckets))
            assert b.query_buckets == sorted(set(b.query_buckets))


class TestFindBucket:
    def test_exact_match(self, bucketer):
        assert bucketer.find_kv_bucket(512) == 512
        assert bucketer.find_query_bucket(512) == 512

    def test_rounds_up(self, bucketer):
        assert bucketer.find_kv_bucket(257) == 512
        assert bucketer.find_query_bucket(33) == 512

    def test_query_len_one_maps_to_decode_bucket(self, bucketer):
        assert bucketer.find_query_bucket(1) == 1

    def test_query_above_decode_rounds_to_the_step(self, bucketer):
        """Only two buckets by default, so every non-decode query pads to the step."""
        for query_len in (2, 33, 129, 511, 512):
            assert bucketer.find_query_bucket(query_len) == 512

    def test_kv_below_block_size_rounds_to_block_size(self, bucketer):
        assert bucketer.find_kv_bucket(1) == BLOCK_SIZE
        assert bucketer.find_kv_bucket(BLOCK_SIZE) == BLOCK_SIZE

    def test_exceeds_max_returns_none(self, bucketer):
        assert bucketer.find_kv_bucket(2049) is None
        assert bucketer.find_query_bucket(513) is None


class TestDispatch:
    def test_num_blocks_derived_from_kv_tier(self, bucketer):
        b = bucketer.dispatch(kv_len=300, query_len=8)
        assert b is not None
        # 300 rounds to the 512 bucket, which is 8 blocks of 64.
        assert b.num_blocks == 512 // BLOCK_SIZE
        assert b.padded_query_len == 512

    def test_decode_dispatch_keeps_query_len_one(self, bucketer):
        b = bucketer.dispatch(kv_len=1000, query_len=1)
        assert b is not None
        assert b.padded_query_len == 1

    def test_over_max_on_either_axis_raises(self, bucketer):
        """Both axes are built to cover their limit, so an over-max length is a
        contract violation by the caller, not a bucket set to fall back from."""
        with pytest.raises(AssertionError, match="no attention bucket"):
            bucketer.dispatch(kv_len=99999, query_len=32)
        with pytest.raises(AssertionError, match="no attention bucket"):
            bucketer.dispatch(kv_len=256, query_len=99999)

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

    def test_prunes_query_buckets_no_real_query_len_can_reach(self, bucketer):
        """Pruning bounds the *smallest real* query_len that reaches a bucket, not
        the bucket itself: a padded bucket may exceed the sequence's block span (a
        2-token query on a 1-block sequence still pads to 512), but a bucket whose
        whole input range lies past the span is unreachable."""
        ascending = sorted(bucketer.query_buckets)
        min_real = {b: (ascending[i - 1] + 1 if i else 1) for i, b in enumerate(ascending)}
        for v in bucketer.variants():
            assert min_real[v.padded_query_len] <= v.num_blocks * BLOCK_SIZE

    def test_count_stays_tractable_at_long_context(self):
        """Dense buckets here would be tens of thousands of Inductor compiles."""
        b = SpyreAttnBucketer(make_config(32768, 2048))
        assert len(b.variants()) < 500


class TestEnvOverride:
    def test_kv_buckets_override(self, monkeypatch):
        """Kept verbatim: the top entry already covers max_model_len=2048."""
        monkeypatch.setenv("SPYRE_ATTN_KV_BUCKETS", "128,512,4096")
        envs.clear_env_cache()
        b = SpyreAttnBucketer(make_config())
        assert b.kv_buckets == [128, 512, 4096]

    def test_query_buckets_override_is_sorted_and_deduped(self, monkeypatch):
        monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "64,1,16,64")
        envs.clear_env_cache()
        b = SpyreAttnBucketer(make_config(max_num_batched_tokens=64))
        assert b.query_buckets == [1, 16, 64]

    def test_truncated_kv_override_is_topped_up_to_max_model_len(self, monkeypatch):
        """A short override would otherwise leave (512, 2048] with no bucket."""
        monkeypatch.setenv("SPYRE_ATTN_KV_BUCKETS", "128,512")
        envs.clear_env_cache()
        b = SpyreAttnBucketer(make_config(max_model_len=2048))
        assert b.kv_buckets == [128, 512, 2048]
        assert b.find_kv_bucket(2048) == 2048

    def test_truncated_query_override_is_topped_up_to_max_batched(self, monkeypatch):
        monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "1,16")
        envs.clear_env_cache()
        b = SpyreAttnBucketer(make_config(max_num_batched_tokens=512))
        assert b.query_buckets == [1, 16, 512]
        assert b.find_query_bucket(512) == 512

    def test_override_above_the_limit_is_left_alone(self, monkeypatch):
        """Entries past the limit are unreachable, not wrong; don't prune them."""
        monkeypatch.setenv("SPYRE_ATTN_KV_BUCKETS", "128,8192")
        envs.clear_env_cache()
        b = SpyreAttnBucketer(make_config(max_model_len=2048))
        assert b.kv_buckets == [128, 8192]

    def test_dispatch_covers_every_in_contract_length_under_a_short_override(self, monkeypatch):
        """The point of the top-up: no in-contract batch falls off either axis."""
        monkeypatch.setenv("SPYRE_ATTN_KV_BUCKETS", "128")
        monkeypatch.setenv("SPYRE_ATTN_QUERY_BUCKETS", "1")
        envs.clear_env_cache()
        max_model_len, max_batched = 1024, 256
        b = SpyreAttnBucketer(make_config(max_model_len, max_batched))
        for kv_len in (1, 129, 500, max_model_len):
            for query_len in (1, 2, 200, max_batched):
                if query_len > kv_len:
                    continue
                assert b.dispatch(kv_len, query_len) is not None

    def test_dispatch_rejects_a_length_outside_the_contract(self, monkeypatch):
        """Past max_model_len there is no bucket by design -- and no silent None."""
        b = SpyreAttnBucketer(make_config(max_model_len=2048))
        with pytest.raises(AssertionError, match="no attention bucket"):
            b.dispatch(2049, 1)

    def test_parse_buckets_rejects_non_positive(self):
        with pytest.raises(ValueError):
            _parse_buckets("0,32")

    def test_parse_buckets_empty_is_none(self):
        assert _parse_buckets("") is None
        assert _parse_buckets(None) is None


class TestBucketKey:
    def test_key_matches_attn_fn_cache_tuple(self):
        b = SpyreAttnBucket(
            num_blocks=4, padded_query_len=32, store_mode="index", needs_gather=True
        )
        assert b.key == (4, 32, "index", True)


class TestBuilderAttnBucketer:
    """The recorder takes the builders' bucketer instead of building its own."""

    @staticmethod
    def _runner(*group_bucketers):
        """A bare runner whose attention groups hold the given bucketers.

        One argument per group, each a list of per-ubatch bucketers (or Nones for
        a builder that exposes none).
        """
        from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

        runner = TorchSpyreModelRunner.__new__(TorchSpyreModelRunner)
        runner.attn_groups = [
            [
                SimpleNamespace(
                    metadata_builders=[
                        SimpleNamespace(_attn_bucketer=b) if b is not None else SimpleNamespace()
                        for b in builders
                    ]
                )
                for builders in group_bucketers
            ]
        ]
        return runner

    def test_returns_the_builders_instance(self):
        bucketer = SpyreAttnBucketer(make_config())
        runner = self._runner([bucketer])
        assert runner._resolve_builder_attn_bucketer() is bucketer

    def test_none_when_no_builder_exposes_one(self):
        assert self._runner([None])._resolve_builder_attn_bucketer() is None
        assert self._runner()._resolve_builder_attn_bucketer() is None

    def test_agreeing_builders_are_accepted(self):
        """Two groups, separately constructed from the same config: same buckets."""
        first = SpyreAttnBucketer(make_config())
        second = SpyreAttnBucketer(make_config())
        runner = self._runner([first], [second])
        assert runner._resolve_builder_attn_bucketer() is first

    def test_diverging_builders_raise(self):
        first = SpyreAttnBucketer(make_config(max_model_len=2048))
        second = SpyreAttnBucketer(make_config(max_model_len=8192))
        runner = self._runner([first], [second])
        with pytest.raises(AssertionError, match="diverge between metadata builders"):
            runner._resolve_builder_attn_bucketer()

    def test_skips_builders_without_a_bucketer(self):
        bucketer = SpyreAttnBucketer(make_config())
        runner = self._runner([None, bucketer])
        assert runner._resolve_builder_attn_bucketer() is bucketer
