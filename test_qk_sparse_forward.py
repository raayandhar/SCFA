from torch.utils.cpp_extension import load
mod = load(
    name="qk_sparse_forward",
    sources=["qk_sparse_forward.cu"],
    extra_cuda_cflags=[
        "-O3","--use_fast_math",
        "-gencode=arch=compute_89,code=sm_89"
    ])
print("compiled OK:", mod)
