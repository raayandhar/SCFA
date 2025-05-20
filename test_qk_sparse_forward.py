from torch.utils.cpp_extension import load
mod = load(
    name="qk_sparse_forward",
    sources=["qk_sparse_forward.cu"],
    extra_cuda_cflags=[
        "-O3","--use_fast_math",
        "-gencode=arch=compute_89,code=sm_89"
    ])
print("compiled OK:", mod)

"""
y = attn.hash_sparse_forward(q_c, k_c, v_c,
                             q_idx, k_idx,
                             q_hash, k_hash,
                             float(sm_scale))
"""