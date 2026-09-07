# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernel for the sm89 DSA prefill MQA logits.

Replaces the ``[H, M, N]`` fp32 score materialization (~2 GiB at M=N=4096,
H=32) with a single fused pass: each program computes a ``[BLOCK_M, BLOCK_N]``
logits tile, looping over heads in registers so no intermediate score tile ever
touches HBM. Dequant stays in the torch wrapper — only the hot
``qk -> relu -> weight -> reduce_h -> mask`` reduction runs here.

fp32 matmul (``input_precision="ieee"``) by default to match the torch
reference exactly. The 4090 has no fp32 tensor cores, so that win is memory
traffic, not flops. Set ``VLLM_SM89_DSA_LOGITS_BF16=1`` to cast the dot inputs
to bf16 and hit the Ada bf16 tensor cores (fp32 accumulate) — several times
faster, at the cost of bf16 mantissa error. DSA logits only feed the topk
selection, so the ranking is what matters, not the exact values.
"""
from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

_MAX_BLOCK_D = 256  # single-tile contraction; larger head dims fall back to torch


def _use_bf16_dot() -> bool:
    return os.environ.get("VLLM_SM89_DSA_LOGITS_BF16") == "1"


@triton.jit
def _mqa_logits_kernel(
    q_ptr, k_ptr, w_ptr, ks_ptr, ke_ptr, out_ptr,
    M, N, H,
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kd,
    stride_wm, stride_wh,
    stride_om, stride_on,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DOT_BF16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < M
    n_mask = offs_n < N
    d_mask = offs_d < D

    # K tile [BLOCK_D, BLOCK_N] loaded once, reused across all heads.
    k_ptrs = k_ptr + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn
    k_tile = tl.load(k_ptrs, mask=d_mask[:, None] & n_mask[None, :], other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h in range(H):
        q_ptrs = (
            q_ptr
            + offs_m[:, None] * stride_qm
            + h * stride_qh
            + offs_d[None, :] * stride_qd
        )
        q_tile = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        if DOT_BF16:
            score = tl.dot(q_tile, k_tile)  # bf16 inputs -> fp32 accumulate
        else:
            score = tl.dot(q_tile, k_tile, input_precision="ieee")
        score = tl.maximum(score, 0.0)
        w = tl.load(w_ptr + offs_m * stride_wm + h * stride_wh, mask=m_mask, other=0.0)
        acc += score * w[:, None]

    ks = tl.load(ks_ptr + offs_m, mask=m_mask, other=0)
    ke = tl.load(ke_ptr + offs_m, mask=m_mask, other=0)
    keep = (offs_n[None, :] >= ks[:, None]) & (offs_n[None, :] < ke[:, None])
    acc = tl.where(keep, acc, float("-inf"))

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def can_use_triton_mqa_logits(q_f: torch.Tensor) -> bool:
    D = q_f.shape[-1]
    return q_f.is_cuda and D <= _MAX_BLOCK_D


def triton_mqa_logits(
    q_f: torch.Tensor,      # [M, H, D] fp32
    k_f: torch.Tensor,      # [N, D] fp32
    weights: torch.Tensor,  # [M, H] fp32
    cu_seqlen_ks: torch.Tensor,  # [M] int32
    cu_seqlen_ke: torch.Tensor,  # [M] int32
    bf16_dot: bool | None = None,
) -> torch.Tensor:
    """Fused DSA prefill logits. See module docstring. Returns [M, N] fp32."""
    M, H, D = q_f.shape
    N = k_f.shape[0]
    use_bf16 = _use_bf16_dot() if bf16_dot is None else bf16_dot
    dot_dtype = torch.bfloat16 if use_bf16 else torch.float32
    q_in = q_f.to(dot_dtype).contiguous()
    k_in = k_f.to(dot_dtype).contiguous()
    weights = weights.float().contiguous()
    ks = cu_seqlen_ks.to(torch.int32)
    ke = cu_seqlen_ke.to(torch.int32)
    out = torch.empty((M, N), dtype=torch.float32, device=q_f.device)

    # BLOCK_N=64 (not 128): sm89 has only ~99 KiB shared memory; the k_tile
    # [BLOCK_D, BLOCK_N] fp32 plus q_tile and acc must fit. 128 needs 128 KiB
    # and silently falls back to torch.
    BLOCK_M, BLOCK_N = 64, 64
    BLOCK_D = triton.next_power_of_2(D)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _mqa_logits_kernel[grid](
        q_in, k_in, weights, ks, ke, out,
        M, N, H,
        q_in.stride(0), q_in.stride(1), q_in.stride(2),
        k_in.stride(0), k_in.stride(1),
        weights.stride(0), weights.stride(1),
        out.stride(0), out.stride(1),
        D=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        DOT_BF16=use_bf16,
    )
    return out
