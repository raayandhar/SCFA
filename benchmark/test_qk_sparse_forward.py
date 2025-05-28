from torch.utils.cpp_extension import load
import torch
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
mod = load(
    name="qk_sparse_forward",
    sources=["../source/qk_sparse_forward.cu"],
    extra_cuda_cflags=[
        "-O2","--use_fast_math",
        "-gencode=arch=compute_86,code=sm_86"
    ])
print("compiled OK:", mod)

"""
y = attn.hash_sparse_forward(q_c, k_c, v_c,
                             q_idx, k_idx,
                             q_hash, k_hash,
                             float(sm_scale))
"""