import torch
import torch.nn.functional as F
from source.scfa_paper.scfa_wrapper import hash_sparse_attention

def vector_hash_fn(x: torch.Tensor, num_buckets: int, R: torch.Tensor) -> torch.Tensor:
    """
    x: (..., D)
    R: (D, b/2)
    """
    D = x.shape[-1]
    assert R.shape == (D, num_buckets // 2)
    return torch.argmax(torch.cat([x @ R, -x @ R], dim=-1), dim=-1)

def get_vector_hash(D: int, num_buckets: int, device: torch.device = "cpu", dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    R = torch.randn(D, num_buckets // 2, device = device, dtype = dtype)
    return lambda x: vector_hash_fn(x, num_buckets, R)

def scfa_hash_attn(q, k, v, num_buckets: int, sm_scale: float = 1.0, vector_hash = None):
    """
    q: (B, H, N, D)
    k: (B, H, N, D)
    v: (B, H, N, D)
    """
    B, H, N, D = q.shape
    if vector_hash is None:
        vector_hash = get_vector_hash(D = D, num_buckets = num_buckets, device = q.device)
    assert num_buckets % 2 == 0, "num_buckets must be even"
    q_hashes = vector_hash(q).to(torch.int32).transpose(1, 2).contiguous() # (B, N, H)
    k_hashes = vector_hash(k).to(torch.int32).transpose(1, 2).contiguous() # (B, N, H)
    assert q_hashes.max() < num_buckets, "q_hashes must be less than num_buckets"
    assert k_hashes.max() < num_buckets, "k_hashes must be less than num_buckets"
    assert q_hashes.min() >= 0, "q_hashes must be non-negative"
    assert k_hashes.min() >= 0, "k_hashes must be non-negative"
    scfa_q = q.transpose(1, 2).contiguous() # (B, N, H, D)
    scfa_k = k.transpose(1, 2).contiguous() # (B, N, H, D)
    scfa_v = v.transpose(1, 2).contiguous() # (B, N, H, D)
    return hash_sparse_attention(scfa_q, scfa_k, scfa_v, q_hashes, k_hashes, sm_scale).transpose(1, 2).contiguous()

def reference_hash_attn(q, k, v, num_buckets: int, sm_scale: float = 1.0, vector_hash = None):
    """
    q: (B, H, N, D)
    k: (B, H, N, D)
    v: (B, H, N, D)

    Note: This implementation sucks! It's just a sanity check.
    """
    assert num_buckets % 2 == 0, "num_buckets must be even"
    if vector_hash is None:
        vector_hash = get_vector_hash(D = q.shape[-1], num_buckets = num_buckets, device = q.device)
    B, H, N, D = q.shape
    q_hashes = vector_hash(q) # (B, H, N)
    k_hashes = vector_hash(k) # (B, H, N)
    out = torch.zeros_like(q)
    for i in range(num_buckets):
        for b in range(B):
            for h in range(H):
                q_mask = (q_hashes[b][h] == i) # (N)
                k_mask = (k_hashes[b][h] == i) # (N)
                q_indices = torch.nonzero(q_mask, as_tuple=False).squeeze() # (N)
                k_indices = torch.nonzero(k_mask, as_tuple=False).squeeze() # (N)
                if len(q_indices.shape) == 0 or len(k_indices.shape) == 0:
                    continue
                q_bucket = q[b, h, q_indices] # (N, D)
                k_bucket = k[b, h, k_indices] # (N, D)
                v_bucket = v[b, h, k_indices] # (N, D)
                attn_mask = q_indices.unsqueeze(-1) >= k_indices.unsqueeze(-2)
                attn_scores = torch.matmul(q_bucket, k_bucket.transpose(-2, -1)) * sm_scale
                attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))
                attn = F.softmax(attn_scores, dim=-1)
                attn = attn.nan_to_num(0.0) # some cols will be totally masked out and softmax will produce NaNs
                # sns.heatmap(attn.cpu().numpy().squeeze(), annot = False, mask = ~attn_mask.cpu().numpy().squeeze())
                # plt.show()
                # return
                partial_prod = torch.matmul(attn, v_bucket)
                out[b, h, q_indices] += partial_prod.squeeze(0)
    return out