# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register-resident top-k index selection for the sm89 DSA indexer.

Borrows SGLang's topk-v2 register-resident idea (scores read once, both passes
stay on-chip) via the pack-and-sort argsort already used in gemma4 routing:
each program loads one logits row into a single on-chip tile, packs
``(sortable_key, index)`` into int64, sorts once, and emits the first ``topk``
indices. Exact (a full sort, not a threshold approximation); -inf/padding rows
recover to ``-1``.

Standalone leaf function. Wire it in place of ``persistent_topk`` at the DSA
indexer call site only after profiling shows selection is worth it.
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

_MAX_N = 16384  # tile is next_pow2(N) int64; beyond this fall back to persistent_topk


@triton.jit
def _topk_row_kernel(
    logits_ptr, out_ptr,
    N, TOPK,
    stride_lm, stride_om,
    PADDED_N: tl.constexpr,
):
    # Ascending-sortable float32 bijection (identical to gemma4 routing): larger
    # logit -> smaller key, so an ascending sort puts the largest logits first.
    MIN32 = -2147483648
    row = tl.program_id(0)
    offs = tl.arange(0, PADDED_N)
    valid = offs < N

    x = tl.load(logits_ptr + row * stride_lm + offs, mask=valid, other=-float("inf"))
    x = x.to(tl.float32)

    bits = x.to(tl.int32, bitcast=True)
    sign = bits >> 31
    key = tl.where(sign == 0, bits ^ -1, bits ^ MIN32)
    key = tl.where(valid, key, 0x7FFFFFFF)  # out-of-range -> sorts last
    packed = ((key.to(tl.int64) & 0x00000000FFFFFFFF) << 32) | offs.to(tl.int64)

    sorted_p = tl.sort(packed, descending=False)  # smallest key (largest logit) first
    all_keys = ((sorted_p >> 32) & 0x00000000FFFFFFFF).to(tl.int32)
    all_ids = (sorted_p & 0x00000000FFFFFFFF).to(tl.int32)

    # Inverse bijection to detect -inf (fewer finite scores than topk -> pad -1).
    sk = all_keys >> 31
    rbits = tl.where(sk < 0, all_keys ^ -1, all_keys ^ MIN32)
    rlog = rbits.to(tl.float32, bitcast=True)
    finite = rlog > -float("inf")

    out = tl.where(finite, all_ids, -1)
    tl.store(out_ptr + row * stride_om + offs, out, mask=offs < TOPK)


def can_use_triton_topk(n: int) -> bool:
    return n <= _MAX_N


def triton_topk_indices(logits: torch.Tensor, topk: int) -> torch.Tensor:
    """Top-``topk`` indices per row of ``logits`` [M, N] (largest first).

    Masked positions must already be -inf in ``logits``. Returns [M, topk]
    int32; rows with fewer than ``topk`` finite scores are -1 padded.
    """
    M, N = logits.shape
    padded_n = triton.next_power_of_2(N)
    out = torch.empty((M, topk), dtype=torch.int32, device=logits.device)
    num_warps = 8 if padded_n <= 4096 else 16
    _topk_row_kernel[(M,)](
        logits, out,
        N, topk,
        logits.stride(0), out.stride(0),
        PADDED_N=padded_n,
        num_warps=num_warps,
    )
    return out
