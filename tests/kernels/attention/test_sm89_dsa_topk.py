# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register-resident DSA topk (Triton) must match torch.topk on distinct data.

CUDA-only. On ties torch.topk and the sort-based kernel may pick different
indices, so the test uses distinct random logits and compares the selected
index *set* per row, plus -1 padding when finite < topk.
"""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton needs GPU")
def test_topk_matches_torch():
    from vllm.utils.sm89_dsa_topk_triton import triton_topk_indices

    torch.manual_seed(0)
    M, N, TOPK = 7, 3000, 512
    # randperm-based distinct values per row -> no ties
    logits = torch.stack([torch.randperm(N, device="cuda").float() for _ in range(M)])

    got = triton_topk_indices(logits, TOPK)
    want = logits.topk(TOPK, dim=-1).indices.to(torch.int32)

    for r in range(M):
        assert set(got[r].tolist()) == set(want[r].tolist())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton needs GPU")
def test_topk_pads_when_fewer_finite():
    from vllm.utils.sm89_dsa_topk_triton import triton_topk_indices

    N, TOPK = 100, 32
    logits = torch.full((1, N), -float("inf"), device="cuda")
    logits[0, :10] = torch.arange(10, device="cuda").float()  # only 10 finite

    got = triton_topk_indices(logits, TOPK)[0]
    assert set(got[:10].tolist()) == set(range(10))
    assert (got[10:] == -1).all()
