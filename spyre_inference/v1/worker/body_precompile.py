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

"""Compile the decoder blocks of every body bucket in forked processes side by side.

With ``SPYRE_PARALLEL_BODY_PRECOMPILE=N``, the body warmup first runs one dummy forward
per bucket in which each decoder block is traced here and compiled in one of N forked
processes (torch-spyre's ``fork_inner_compiles``). A block whose compile went to a
child does not run: it returns its input hidden states, so the forward goes on to the
next block, and to the next bucket. Every other graph (embedding, norm, MoE regions)
compiles in this process as usual. The warmup that follows then finds the blocks in
this start's caches.

Only the block graphs are forked because they are most of the body warmup (two per
bucket for Gemma-4, one per layer kind) and their output has the shape of their input.
"""

import contextlib
import functools
import inspect
import time
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
from vllm.logger import init_logger

from spyre_inference.v1.worker.compile_granularity import repeated_block_lists

logger = init_logger(__name__)


def _compiled_blocks(model: nn.Module) -> list[nn.Module]:
    """The decoder blocks that ``compile_blocks`` compiled in place, once each."""
    seen: dict[int, nn.Module] = {}
    for blocks, marked in repeated_block_lists(model):
        if marked:  # a vision tower: compiled at image budgets, not body buckets
            continue
        for block in blocks:
            if getattr(block, "_compiled_call_impl", None) is not None:
                seen.setdefault(id(block), block)
    return list(seen.values())


def _passthrough(block: nn.Module) -> Callable | None:
    """What a block whose compile went to a child returns instead of running.

    vLLM decoder blocks take ``(positions, hidden_states, residual, ...)`` and return
    ``(hidden_states, residual)``. The stand-in returns a new tensor like its input
    hidden states, and it again as the residual when none came in, as a block that adds
    its own residual does. New and made in inference mode, as a compiled block's output
    is: the next block's guards check that, and the first block's input can differ (a
    multimodal model's merged embeddings are not inference tensors). None for a block
    with another signature: it is not precompiled.
    """
    try:
        params = list(inspect.signature(block.forward).parameters)
    except (TypeError, ValueError):
        return None
    if params[:3] != ["positions", "hidden_states", "residual"]:
        return None

    def output(*args, **kwargs):
        bound = dict(zip(params, args))
        bound.update(kwargs)
        with torch.inference_mode():
            hidden = torch.empty_like(bound["hidden_states"])
        residual = bound.get("residual")
        return hidden, hidden if residual is None else residual

    return output


def precompile_body_buckets(
    model: nn.Module,
    dummy_run: Callable[[int], object],
    bucket_sizes: Iterable[int],
    workers: int,
):
    """Trace each bucket's blocks, compiling them in up to ``workers`` children.

    ``dummy_run(size)`` runs one dummy forward of ``size`` tokens. Returns a started
    ``SecondPassCheck`` for the warmup that follows when children compiled something,
    else None (warm caches, nothing to fork, or a torch-spyre without the helper).
    """
    try:
        from torch_spyre._inductor.parallel_precompile import (  # ty: ignore[unresolved-import]
            CompileOnlyAbort,
            SecondPassCheck,
            fork_inner_compiles,
        )
    except ImportError:
        logger.warning(
            "SPYRE_PARALLEL_BODY_PRECOMPILE needs a torch-spyre with fork_inner_compiles "
            "(torch-spyre/torch-spyre#6262); compiling the body buckets one by one."
        )
        return None
    params = inspect.signature(fork_inner_compiles).parameters
    if "should_fork" not in params:
        logger.warning(
            "SPYRE_PARALLEL_BODY_PRECOMPILE needs fork_inner_compiles(should_fork=...); "
            "compiling the body buckets one by one."
        )
        return None

    blocks = [(b, out) for b in _compiled_blocks(model) if (out := _passthrough(b)) is not None]
    if not blocks:
        logger.info("Parallel body precompile: no decoder block to precompile.")
        return None

    in_block = [False]
    aborted = [0]

    def wrap(impl: Callable, output: Callable) -> Callable:
        @functools.wraps(impl)
        def call(*args, **kwargs):
            in_block[0] = True
            try:
                return impl(*args, **kwargs)
            except CompileOnlyAbort:
                aborted[0] += 1
                return output(*args, **kwargs)
            finally:
                in_block[0] = False

        return call

    originals = [(b, b._compiled_call_impl) for b, _ in blocks]
    t0 = time.time()
    sizes = sorted(bucket_sizes, reverse=True)
    warm = False
    # At TP > 1 every rank precompiles, so stay within this rank's CPU share, which
    # vLLM's threading setup gives torch.
    workers = max(1, min(workers, torch.get_num_threads()))
    try:
        for block, output in blocks:
            block._compiled_call_impl = wrap(block._compiled_call_impl, output)
        # A compile can take minutes in a child that shares the CPUs with the others.
        # A cache hit loads here, so with warm caches the first bucket's blocks run and
        # the precompile stops there.
        extra = {"load_cache_hits": True} if "load_cache_hits" in params else {}
        with fork_inner_compiles(
            workers, timeout_s=3600.0, should_fork=lambda: in_block[0], **extra
        ) as pool:
            for i, size in enumerate(sizes):
                # A stand-in outside a block stops this bucket: the warmup compiles the rest.
                with contextlib.suppress(CompileOnlyAbort):
                    dummy_run(size)
                if i == 0 and aborted[0] == 0:
                    warm = True  # the first bucket's blocks were cache hits: a restart
                    break
    finally:
        for block, impl in originals:
            block._compiled_call_impl = impl
        # Forget the stand-in graphs, so the warmup traces the blocks again.
        codes = {type(b).forward.__code__ for b, _ in blocks}
        codes.add(nn.Module._call_impl.__code__)
        for code in codes:
            torch._dynamo.reset_code(code)
    logger.info(
        "Parallel body precompile: %s in %.1fs.",
        "skipped, the caches are warm"
        if warm
        else f"{pool.ok} block graphs compiled in {workers} workers ({pool.failed} failed) "
        f"over {len(sizes)} buckets",
        time.time() - t0,
    )
    if warm or not pool.ok:
        return None
    return SecondPassCheck(expected=pool.ok)
