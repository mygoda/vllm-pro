# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm89 DSA prefill logits: tiled path must equal the naive full einsum.

CPU-only, no GPU needed. Guards the M-chunk loop transformation in
vllm.utils.deep_gemm._sm89_torch_ref_mqa_logits.
"""
import pytest
import torch

from vllm.utils.deep_gemm import _sm89_torch_ref_mqa_logits


def _naive(q_f, k_f, weights, ks, ke):
    N = k_f.shape[0]
    score = torch.einsum("mhd,nd->hmn", q_f, k_f)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(0)
    idx = torch.arange(N)
    mask = (idx[None, :] >= ks[:, None]) & (idx[None, :] < ke[:, None])
    return logits.masked_fill(~mask, float("-inf"))


def test_tiled_matches_naive():
    torch.manual_seed(0)
    M, H, D, N = 1300, 8, 64, 900  # M spans several 512-chunks
    q = torch.randn(M, H, D)
    k = torch.randn(N, D)
    kscale = torch.rand(N) + 0.5
    weights = torch.rand(M, H)
    ks = torch.randint(0, N // 2, (M,), dtype=torch.int32)
    ke = torch.randint(N // 2, N, (M,), dtype=torch.int32)

    got = _sm89_torch_ref_mqa_logits((q, None), (k, kscale), weights, ks, ke)
    want = _naive(q, k * kscale.view(N, 1), weights, ks, ke)

    finite = torch.isfinite(want)
    assert torch.equal(got.isfinite(), finite)
    assert torch.allclose(got[finite], want[finite], atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton path needs GPU")
@pytest.mark.parametrize("bf16", [False, True])
def test_triton_matches_naive(bf16):
    from vllm.utils.sm89_dsa_triton import triton_mqa_logits

    torch.manual_seed(0)
    M, H, D, N = 1300, 8, 128, 900  # M/N span several blocks, D power-of-2
    dev = "cuda"
    q = torch.randn(M, H, D, device=dev)
    k = torch.randn(N, D, device=dev)
    kscale = (torch.rand(N, device=dev) + 0.5)
    weights = torch.rand(M, H, device=dev)
    ks = torch.randint(0, N // 2, (M,), dtype=torch.int32, device=dev)
    ke = torch.randint(N // 2, N, (M,), dtype=torch.int32, device=dev)

    k_f = k * kscale.view(N, 1)
    got = triton_mqa_logits(q, k_f, weights, ks, ke, bf16_dot=bf16)
    want = _naive(q, k_f, weights, ks, ke)

    finite = torch.isfinite(want)
    assert torch.equal(got.isfinite(), finite)
    if not bf16:
        assert torch.allclose(got[finite], want[finite], atol=1e-3, rtol=1e-3)
    else:
        # bf16 shifts absolute values; what matters is the topk ranking that
        # feeds index selection. Require high overlap of the top-64 per row.
        kk = 64
        gi = got.masked_fill(~got.isfinite(), float("-inf")).topk(kk, -1).indices
        wi = want.masked_fill(~want.isfinite(), float("-inf")).topk(kk, -1).indices
        overlap = (gi.sort(-1).values == wi.sort(-1).values).float().mean()
        assert overlap > 0.95


if __name__ == "__main__":
    test_tiled_matches_naive()
    print("ok")
