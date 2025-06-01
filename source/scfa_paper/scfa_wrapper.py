# Reference code from the SCFA paper: https://arxiv.org/pdf/2306.01160
# Source: https://github.com/epfml/dynamic-sparse-flash-attention/blob/main/runtime-experiments/timeperf-hash-and-qk-sparse.ipynb

import torch
import math
from source.scfa_paper.scfa_hash_sparse import hash_sparse_attention_kernel
from source.scfa_paper.scfa_qk_sparse import qk_sparse_attention_kernel

def dynamic_sparse_attention(q, k, v, q_idx, k_idx, sm_scale=None, sparsity_mode='hash'):
    """ 
    Keyword arguments:
    q: query tensor of shape (BATCH, N_CTX_Q, H, D_HEAD)
    k: key tensor of shape (BATCH, N_CTX_KV, H, D_HEAD)
    v: value tensor of shape (BATCH, N_CTX_KV, H, D_HEAD)
    q_idx: tensor of shape (BATCH, N_CTX_Q, H) for each sequence in the batch, for each query in the sequence, for each head, 
        represents either the bucket index if sparsity_mode=='hash' or the whether to keep that given head if sparsity_mode=='qk'. 
        The type should be torch.int32 if sparsity_mode=='hash' and torch.float if sparsity_mode=='qk'.
    k_idx: tensor of shape (BATCH, N_CTX_KV, H) for each sequence in the batch, for each key in the sequence, for each head, 
        represents either the bucket index if sparsity_mode=='hash' or the whether to keep that given head if sparsity_mode=='qk'.
        The type should be torch.int32 if sparsity_mode=='hash' and torch.float if sparsity_mode=='qk'
    sm_scale: normalization constant, 1/sqrt(D_HEAD) unless specified
    sparsity_mode: 'hash' to select the hash-sparse implementation and 'qk' for the qk-sparse implementation
    """

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))

    if sparsity_mode == 'hash':
        return hash_sparse_attention(q, k, v, q_hash=q_idx, k_hash=k_idx, sm_scale=sm_scale)
    elif sparsity_mode == 'qk':
        return qk_sparse_attention(q, k, v, q_keep=q_idx, k_keep=k_idx, sm_scale=sm_scale)
    else:
        raise KeyError(f"Unknown sparsity_mode: '{sparsity_mode}', should be in  ['hash', 'qk']")


def compact(keep_tensor, x, index=None):
  """ Build a compact representation of x
  Keyword arguments:
  x: input tensor to compact, x.shape = (BATCH, N_CTX, H, D_HEAD) 
  keep_tensor: float tensor of shape (BATCH, N_CTX, H) containing a 1 when the head is kept, else 0
  """
  BATCH, T, H, D_HEAD = x.shape
  if index is None:
    with torch.no_grad():
        indices_per_head = keep_tensor.sum(dim=-2) 
        buffer_size = indices_per_head.max().int() # first sum computes the num of non-killed elem per head, we take to max of that
        # sorting: it is very important that the sorting is stable, else we cannot use causal masking
        sorted = keep_tensor.sort(dim=-2, descending=True, stable=True) # sorted.indices.shape == (BATCH x T x H) , now sorted over sequence T
        index = sorted.indices[:,:buffer_size,:] # (BATCH x buffer_size x H) expand indices to cover all the dimensions for each heads
  else:
    indices_per_head = None
  compact_x = x.gather(dim=-3, index=index.unsqueeze(-1).expand(-1,-1,-1,D_HEAD)) # (BATCH x buffer_size x H x D_HEAD) / expand indices to cover all the dimensions for each heads
  return compact_x, index, indices_per_head


@torch.no_grad()
def pad_index(index, indices_per_head, pad_idx=-1):
  """ Pad the index tensor to comply with the kernel, returns a copy.
  Keyword arguments:
  index: original index tensor to pad given by `compact`, index.shape = (BATCH, buffer_size, H). For each batch and timestep, reprsents the head idx it's originating from.
  indices_per_head: of shape (BATCH, H), for each head, contains how many indices have not been dropped.
  """
  BATCH, buffer_size, H = index.shape
  index_copy = torch.clone(index).type(torch.int32)
  mask = torch.arange(buffer_size, device=index.device).view(1,-1,1).expand(BATCH,buffer_size,H) >= indices_per_head.view(BATCH,1,-1)
  index_copy[mask] = pad_idx
  return index_copy


def qk_sparse_attention(q, k, v, q_keep, k_keep, sm_scale):
    assert q_keep.dtype == torch.float and k_keep.dtype == torch.float

    BATCH, N_CTX_Q, H, D_HEAD = q.shape 

    # Building compact representations
    q_c, q_idx, iph_q = compact(q_keep, q) # q_c.shape = (BATCH, compact_N_CTX_Q, H)
    k_c, k_idx, iph_k = compact(k_keep, k) # k_c.shape = (BATCH, compact_N_CTX_KV, H)
    v_c, _, _ = compact(k_keep, v, index=k_idx) # v_c.shape = (BATCH, compact_N_CTX_KV, H)
    q_idx_padded = pad_index(q_idx, iph_q, pad_idx=-1) # (B, compact_N_CTX_Q, H)
    k_idx_padded = pad_index(k_idx, iph_k, pad_idx=1e9) # (B, compact_N_CTX_KV, H)

    # We need to transpose everything
    q_c = q_c.transpose(1, 2).contiguous() # (BATCH, H, compact_N_CTX_Q, D_HEAD)
    k_c = k_c.transpose(1, 2).contiguous() # (BATCH, H, compact_N_CTX_KV, D_HEAD)
    v_c = v_c.transpose(1, 2).contiguous() # (BATCH, H, compact_N_CTX_KV, D_HEAD)
    q_idx_padded = q_idx_padded.transpose(1, 2).contiguous() # (BATCH, H, compact_N_CTX_Q)
    k_idx_padded = k_idx_padded.transpose(1, 2).contiguous() # (BATCH, H, compact_N_CTX_KV)

    y_c = qk_sparse_attention_kernel(q_c, k_c, v_c, q_idx_padded, k_idx_padded, sm_scale).transpose(1,2)
    y = torch.zeros_like(q).scatter(dim=1, index=q_idx.long().view(BATCH,-1,H,1).expand(BATCH, -1, H, D_HEAD), src=y_c)
    return y


def hash_sparse_attention(q, k, v, q_hash, k_hash, sm_scale):
    assert q_hash.dtype == torch.int32 and k_hash.dtype == torch.int32

    BATCH, N_CTX_Q, H, D_HEAD = q.shape 

    q = q.transpose(1, 2) # (BATCH, H, N_CTX_Q, D_HEAD)
    k = k.transpose(1, 2) # (BATCH, H, N_CTX_KV, D_HEAD)
    v = v.transpose(1, 2) # (BATCH, H, N_CTX_KV, D_HEAD)
    q_hash = q_hash.transpose(1, 2).contiguous() # (BATCH, H, N_CTX_Q)
    k_hash = k_hash.transpose(1, 2).contiguous() # (BATCH, H, N_CTX_KV)

    # Re-order the queries,keys,values according q_hash and k_hash
    q_hash = q_hash.sort(dim=-1, stable=True) # q_hash.shape = (BATCH, H, N_CTX_Q), stable sort to keep time ordering within a bucket
    k_hash = k_hash.sort(dim=-1, stable=True) # k_hash.shape = (BATCH, H, N_CTX_KV)

    q_idx = q_hash.indices 
    k_idx = k_hash.indices

    q_hash = q_hash.values
    k_hash = k_hash.values

    q_idx_extended = q_idx.unsqueeze(-1).expand_as(q)
    k_idx_extended = k_idx.unsqueeze(-1).expand_as(k)

    q = torch.gather(q, dim=-2, index=q_idx_extended).contiguous()
    k = torch.gather(k, dim=-2, index=k_idx_extended).contiguous()
    v = torch.gather(v, dim=-2, index=k_idx_extended).contiguous()
    
    y = hash_sparse_attention_kernel(q, k, v, q_idx, k_idx, q_hash, k_hash, sm_scale)
    y = torch.zeros((BATCH, H, N_CTX_Q, D_HEAD), dtype=q.dtype, device=q.device).scatter(dim=2, index=q_idx_extended, src=y).transpose(1,2).contiguous()
    return y

"""
Utility function to generate random tensors for testing.
"""
def get_tensors(BATCH, H, N_CTX, D_HEAD):

    q = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=torch.bfloat16, device="cuda", requires_grad=True)

    return q, k, v

"""
Utility function to run the standard torch flash attention kernel.
"""
def flashattention(q, k, v):
    
    BATCH, N_CTX, H, D_HEAD = q.shape 

    q = q.view(BATCH, N_CTX, H, D_HEAD).transpose(1, 2) # (BATCH, H, N_CTX, D_HEAD)
    k = k.view(BATCH, N_CTX, H, D_HEAD).transpose(1, 2) # (BATCH, H, N_CTX, D_HEAD)
    v = v.view(BATCH, N_CTX, H, D_HEAD).transpose(1, 2) # (BATCH, H, N_CTX, D_HEAD)
  
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=None, is_causal=True)
    return y.transpose(1,2).contiguous()