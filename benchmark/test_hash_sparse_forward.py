from torch.utils.cpp_extension import load
import torch
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
os.environ["MAX_JOBS"] = "16" # use some of Bentham's many cores
hash_sparse_forward = load(
    name="hash_sparse_forward",
    sources=[os.path.join(os.path.dirname(__file__), "../source/hash_sparse_forward.cu")],
    extra_cuda_cflags=[
        "-O2","--use_fast_math",
        "-gencode=arch=compute_86,code=sm_86"
    ],
    verbose = True)
print("compiled OK:", hash_sparse_forward)

# Test 1: ===== Dense Attention (standard attn) =====
# Define a simple test case
Q = torch.randn(1, 1, 1024, 64, dtype=torch.bfloat16, device = "cuda")
K = torch.randn(1, 1, 1024, 64, dtype=torch.bfloat16, device = "cuda")
V = torch.randn(1, 1, 1024, 64, dtype=torch.bfloat16, device = "cuda")

Q_idx = torch.randint(0, 1024, (1, 1, 1024), dtype=torch.int32, device = "cuda")
K_idx = torch.randint(0, 1024, (1, 1, 1024), dtype=torch.int32, device = "cuda")
Q_hash = torch.randint(0, 1024, (1, 1, 1024), dtype=torch.int32, device = "cuda")
K_hash = Q_hash.clone() # same hash for K to make sure that all attends to all

reference_attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
kernel_attn = hash_sparse_forward.forward(Q, K, V, Q_idx, K_idx, Q_hash, K_hash, 1.0)
assert torch.allclose(reference_attn.to(torch.float32), kernel_attn, atol=1e-5, rtol=1e-5)
print("Dense attention test passed.")