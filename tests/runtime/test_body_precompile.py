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

"""Tests for the parallel body precompile in warmup (no device needed)."""

import contextlib
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

from spyre_inference.v1.worker import body_precompile


class _Block(nn.Module):
    def forward(self, positions, hidden_states, residual, **kwargs):
        raise AssertionError("the compiled call stands in for forward")


class _OtherBlock(nn.Module):
    def forward(self, x):
        raise AssertionError("never precompiled")


def _run(monkeypatch, cold):
    """Three blocks over two buckets; returns what the precompile did."""

    class CompileOnlyAbort(Exception):
        pass

    pool = SimpleNamespace(ok=4, failed=0)
    forks = []

    @contextlib.contextmanager
    def fork_inner_compiles(workers, timeout_s=300.0, should_fork=None):
        forks.append(should_fork)
        yield pool

    parallel_precompile = SimpleNamespace(
        CompileOnlyAbort=CompileOnlyAbort,
        SecondPassCheck=lambda expected: SimpleNamespace(expected=expected),
        fork_inner_compiles=fork_inner_compiles,
    )
    monkeypatch.setitem(
        sys.modules, "torch_spyre._inductor.parallel_precompile", parallel_precompile
    )
    resets = []
    monkeypatch.setattr(torch._dynamo, "reset_code", resets.append)

    calls = []

    def compiled(i):
        def call(positions, hidden_states, residual, **kwargs):
            calls.append((i, hidden_states.shape[0], residual is None, forks[-1]()))
            if cold:
                raise CompileOnlyAbort()
            return hidden_states + 1, hidden_states

        return call

    blocks = nn.ModuleList([_Block(), _Block(), _OtherBlock()])
    impls = [compiled(i) for i in range(3)]
    for block, impl in zip(blocks, impls):
        block._compiled_call_impl = impl
    monkeypatch.setattr(body_precompile, "repeated_block_lists", lambda model: [(blocks, False)])

    outputs = []

    def dummy_run(size):
        assert not forks[-1](), "compiles outside a block stay in this process"
        hidden, residual = torch.zeros(size, 4), None
        for block in blocks[:2]:
            hidden, residual = block(None, hidden, residual)
        outputs.append((hidden, residual))

    check = body_precompile.precompile_body_buckets(nn.Module(), dummy_run, [1, 8], workers=4)
    assert [b._compiled_call_impl for b in blocks] == impls, "the compiled calls are restored"
    return calls, outputs, resets, check


def test_cold_start_traces_every_bucket_and_passes_hidden_states_through(monkeypatch):
    calls, outputs, resets, check = _run(monkeypatch, cold=True)
    # Largest bucket first; the first block sees no residual, the next its stand-in's.
    assert calls == [
        (0, 8, True, True),
        (1, 8, False, True),
        (0, 1, True, True),
        (1, 1, False, True),
    ]
    for hidden, residual in outputs:
        # Made in inference mode, as a compiled block's output is.
        assert hidden.is_inference() and residual.is_inference()
    assert _Block.forward.__code__ in resets
    assert nn.Module._call_impl.__code__ in resets
    assert check.expected == 4


def test_restart_with_warm_caches_skips_the_other_buckets(monkeypatch):
    calls, outputs, resets, check = _run(monkeypatch, cold=False)
    assert [c[:2] for c in calls] == [(0, 8), (1, 8)]
    assert check is None
    assert _Block.forward.__code__ in resets


def test_only_decoder_signatures_get_a_stand_in():
    assert body_precompile._passthrough(_Block()) is not None
    assert body_precompile._passthrough(_OtherBlock()) is None
