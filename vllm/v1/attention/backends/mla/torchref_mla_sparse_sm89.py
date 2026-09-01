# sm89 portable sparse-MLA: chunked torch-reference over gathered topk KV.
import torch
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
)

_CHUNK = 64
_DBG = "/root/sm89dbg.txt"



import os as _os
_DBG_ON = _os.environ.get("SM89_MLA_LOG") == "1"
def _log(msg):
    if not _DBG_ON:
        return
    try:
        with open(_DBG, "a") as fh:
            fh.write(msg + "\n")
    except Exception:
        pass


class TorchRefMLASparseSM89Impl(FlashMLASparseImpl):
    def __init__(self, *a, **k):
        _log("IMPL __init__")
        super().__init__(*a, **k)
    def forward(self, *a, **k):
        _log("forward() ENTER")
        return super().forward(*a, **k)

    def forward_mha(self, *a, **k):
        _log("forward_mha CALLED")
        return super().forward_mha(*a, **k)

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        _log("forward_mqa CALLED q_is_tuple=%s kv_dtype=%s" % (isinstance(q, tuple), self.kv_cache_dtype))
        import os as _os
        if _os.environ.get('SM89_DUMP_QKV') == '1':
            _qv = q[0] if isinstance(q, tuple) else q
            _kf = kv_c_and_k_pe_cache.reshape(-1, kv_c_and_k_pe_cache.shape[-1]).float()
            _nz = int((_kf.abs().sum(-1)>1e-4).sum())
            _log('QKV q=%s qnorm=%.2f kv=%s kvnorm=%.2f nzrows=%d' % (tuple(_qv.shape), _qv.float().norm().item(), tuple(_kf.shape), _kf.norm().item(), _nz))
            if self.topk_indices_buffer is not None:
                _tb = self.topk_indices_buffer[:_qv.shape[0]]
                _log('  topk[0,:8]=%s valid=%d' % (str(_tb[0,:8].tolist()), int((_tb[0]>=0).sum())))
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            q = ql_nope if (q_pe is None or q_pe.shape[-1] == 0) else torch.cat([ql_nope, q_pe], dim=-1)
        actual_num_heads = q.shape[1]
        num_actual_toks = q.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        attn_out, lse = self._forward_bf16_kv(
            q, kv_c_and_k_pe_cache, topk_indices, attn_metadata, actual_num_heads
        )
        if not self.need_to_return_lse_for_decode:
            lse = None
        return attn_out, lse

    def _bf16_flash_mla_kernel(
        self, q, kv_c_and_k_pe_cache, topk_indices, topk_length=None, actual_num_heads=None
    ):
        nt, H, D = q.shape
        if actual_num_heads is None:
            actual_num_heads = H
        kv = kv_c_and_k_pe_cache.reshape(-1, kv_c_and_k_pe_cache.shape[-1])
        kvdim = kv.shape[-1]
        vdim = min(512, kvdim)
        d = min(D, kvdim)
        _log("kernel q=%s kv=%s topk=%s d=%d vdim=%d scale=%s" % (
            tuple(q.shape), tuple(kv.shape), tuple(topk_indices.shape), d, vdim, self.softmax_scale))
        topk = topk_indices.shape[-1]
        ti = topk_indices.reshape(nt, topk).long()
        out = q.new_empty((nt, H, vdim))
        lse_out = q.new_empty((nt, H), dtype=torch.float32)
        for s in range(0, nt, _CHUNK):
            e = min(s + _CHUNK, nt)
            tic = ti[s:e]
            valid = tic >= 0
            g = kv[tic.clamp(min=0)]
            if topk_length is not None:
                ar = torch.arange(topk, device=q.device)[None, :]
                valid = valid & (ar < topk_length[s:e].reshape(-1, 1))
            sc = torch.einsum("nhd,ntd->nht", q[s:e, :, :d].float(), g[..., :d].float()) * self.softmax_scale
            sc = sc.masked_fill(~valid[:, None, :], float("-inf"))
            lse_out[s:e] = torch.logsumexp(sc, dim=-1)
            at = torch.softmax(sc, dim=-1)
            out[s:e] = torch.einsum("nht,ntd->nhd", at, g[..., :vdim].float()).to(q.dtype)
        return out[:, :actual_num_heads, :], lse_out[:, :actual_num_heads].to(q.dtype)


class TorchRefMLASparseSM89Backend(FlashMLASparseBackend):
    @classmethod
    def get_impl_cls(cls):
        return TorchRefMLASparseSM89Impl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major in [8, 9, 10]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 576]

_log("MODULE_IMPORTED")
