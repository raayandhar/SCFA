import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6" # for our RTX 3090. Modify for your own GPU.
os.environ["MAX_JOBS"] = "16" # use some of Bentham's many cores
hash_attn_kernel = load(
    name="hash_attn", sources=[os.path.normpath(os.path.join(os.path.dirname(__file__), "../source/splash_main.cpp")), 
                                  os.path.normpath(os.path.join(os.path.dirname(__file__), "../source/splash.cu"))], 
                                  extra_cuda_cflags=["-O2"]
)

print("Kernel loaded")