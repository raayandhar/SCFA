// hash_sparse_forward.cu
// This is based on the dynamic-sparse-flash-attention paper.
// I am porting their hash-sparse triton kernel to CUDA.
// They use q,k,v bfloat16

/*
 * Dimension Key.
 * Z = batch size (B is reserved for block size)
 * H = number of heads
 * N_CTX_Q = sequence length of the query side after we drop the "killed" tokens
 * N_CTX_KV = sequence length of the key-value side after we drop the "killed" tokens
 * D = per-head embedding
*/

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cstdint>
#include <cuda_runtime.h>
#include <math_constants.h>

#define DIVUP(x, y) (((x) + (y) - 1) / (y))

// ======== Warp-level reductions ========

// Find the max value across threads in a warp.
__device__ inline float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}
// Find min values across threads in a warp.
__device__ inline float warp_reduce_min(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fminf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}
// Find sum of values across a warp.
__device__ inline float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

// Conversion helpers.
__device__ inline float bf16_to_fp32(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ inline __nv_bfloat16 fp32_to_bf16(float x) { return __float2bfloat16(x); }

// ======== Kernel-tunable parameters ========
constexpr int BLOCK_M = 128; // queries per thread-block (rows)
constexpr int BLOCK_N = 128; // keys per thread-block (cols)
constexpr int D_HEAD = 64; // per-head embedding size
constexpr int MAX_UNROLL = 16;

// Shared-memory static buffers, with hash bucket ids now
struct SharedHashStore { // per-CTA scratchpad
    __nv_bfloat16 K[BLOCK_N][D_HEAD]; // stream keys
    __nv_bfloat16 V[BLOCK_N][D_HEAD]; // stream values re-used for logits
    int32_t    Kidx[BLOCK_N]; // original time indicies
    int32_t      Kh[BLOCK_N]; // hash bucket id    
};

__global__ void hash_sparse_forward(
    const __nv_bfloat16* __restrict__ Q, // Query matrix [H, N_CTX_Q, D]
    const __nv_bfloat16* __restrict__ K, // Key matrix [H, N_CTX_KV, D]
    const __nv_bfloat16* __restrict__ V, // Value matrix [H, N_CTX_KV, D]
    const int32_t*       __restrict__ Q_idx, // Query index [H, N_CTX_Q]
    const int32_t*       __restrict__ K_idx, // Key index [H, N_CTX_KV]
    const int32_t*       __restrict__ Q_hash, // Query hash [H, N_CTX_Q]
    const int32_t*       __restrict__ K_hash, // Key hash [H, N_CTX_KV]
    const float                       sm_scale, // Softmax scaling (1/sqrt(D))
    float*               __restrict__ O, // Output matrix [H, N_CTX_Q, D]
    float*               __restrict__ L, // Logits [H, N_CTX_Q]
    float*               __restrict__ M, // Max logits [H, N_CTX_Q]
    // contiguous 4‑D layout strides (head‑major):
    int64_t sqm,int64_t sqd,  // Q: (time, dim)
    int64_t skn,int64_t skd,  // K
    int64_t svn,int64_t svd,  // V
    int64_t som,int64_t sod,  // O
    int64_t sqim,             // Q_idx stride over time
    int64_t skim,             // K_idx stride over time
    int64_t sqhm,             // Q_hash stride over time
    int64_t skhm,             // K_hash stride over time
    // Dimension key
    int Nq, int Nk) 
{    
    extern __shared__ SharedHashStore shmem[];
    SharedHashStore& store = *shmem;
    const int lane = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;
    const int tile_q0 = blockIdx.x * BLOCK_M;
    const int head_id = blockIdx.y;
    if (tile_q0 >= Nq) return;

    // base pointers for this head
    const __nv_bfloat16* Qh = Q + head_id * Nq * D_HEAD;   // contiguous in (time, dim)
    const __nv_bfloat16* Kh = K + head_id * Nk * D_HEAD;
    const __nv_bfloat16* Vh = V + head_id * Nk * D_HEAD;
    const int32_t*      QI = Q_idx  + head_id * Nq;
    const int32_t*      KI = K_idx  + head_id * Nk;
    const int32_t*      QH = Q_hash + head_id * Nq;
    const int32_t*      KH = K_hash + head_id * Nk;
    float*              Oh = O      + head_id * Nq * D_HEAD;
    float*              Lh = L      + head_id * Nq;
    float*              Mh = M      + head_id * Nq;

    // Load BLOCK_M queries (one dim component per thread)    
    int32_t qi_local[BLOCK_M];
    int32_t qh_local[BLOCK_M];
    float   qvec_local[BLOCK_M]; // single component (== lane) of each query row

    #pragma unroll MAX_UNROLL 
    for (int mi = 0; mi < BLOCK_M; ++mi) {
        int gq = tile_q0 + mi;
        bool in_range = gq < Nq;
        qi_local[mi] = in_range ? QI[gq * sqim] : -1;
        qh_local[mi] = in_range ? QH[gq * sqhm] : INT_MAX;

        if (in_range) {
            __nv_bfloat16 qvec = Qh[gq * sqm + lane * sqd];
            qvec_local[mi] = bf16_to_fp32(qvec);
        } else {
            qvec_local[mi] = 0.0f;
        }
    }

    int max_qi = -1;
    int min_qh = INT_MAX;
    int max_qh = -1;

    #pragma unroll MAX_UNROLL
    for (int mi = 0; mi < BLOCK_M; ++mi) {
        max_qi = max(max_qi, qi_local[mi]);
        min_qh = min(min_qh, qh_local[mi]);
        max_qh = max(max_qh, qh_local[mi]);
    }
    max_qi = warp_reduce_max(max_qi);
    min_qh = warp_reduce_min(min_qh);
    max_qh = warp_reduce_max(max_qh);

    // per-row streaming softmax accumulators
    float m_prev[BLOCK_M];
    float l_prev[BLOCK_M];

    #pragma unroll MAX_UNROLL
    for (int mi = 0; mi < BLOCK_M; ++mi) { m_prev[mi] = -CUDART_INF_F; l_prev[mi] = 0.0f; }    
    
    float acc[BLOCK_M][D_HEAD];
    #pragma unroll MAX_UNROLL
    for (int mi = 0; mi < BLOCK_M; ++mi) {
        #pragma unroll MAX_UNROLL
        for (int d = 0; d < D_HEAD; ++d) {
            acc[mi][d] = 0.0f;  
        }
    }

    // iterate over key tiles
    const int n_key_tiles = DIVUP(Nk, BLOCK_N);
    for (int tb = 0; tb < n_key_tiles; ++tb) { 
        const int tile_k0 = tb * BLOCK_N;

        // load key has + index into shared (1 elem per lane)
        int local_min_H = INT_MAX;
        int local_max_H = -1;

        if (lane < BLOCK_N) {
            int gk = tile_k0 + lane;
            int32_t kh = (gk < Nk) ? KH[gk * skhm] : INT_MAX;
            store.Kh[lane] = kh;
            local_min_H = kh;
            local_max_H = kh;
            store.Kidx[lane] = (gk < Nk) ? KI[gk * skim] : INT_MAX;
        }
        local_min_H = warp_reduce_min(local_min_H);
        local_max_H = warp_reduce_max(local_max_H);
        __syncthreads();

        if (local_min_H > max_qh) continue;
        if (local_max_H < min_qh) continue;

        // load key and value tiles
        for (int idx = lane; idx < BLOCK_N * D_HEAD; idx += 32) {
            int row = idx / D_HEAD;
            int col = idx % D_HEAD;
            int gk = tile_k0 + row;

            __nv_bfloat16 kv = (gk < Nk) ? Kh[gk * skn + col * skd] : fp32_to_bf16(0.0f);
            __nv_bfloat16 vv = (gk < Nk) ? Vh[gk * svn + col * svd] : fp32_to_bf16(0.0f);

            store.K[row][col] = kv;
            store.V[row][col] = vv;
        }
        __syncthreads();

        // per-query dot, mask, streaming softmax
        #pragma unroll MAX_UNROLL
        for (int mi = 0; mi < BLOCK_M; ++mi) {
            int gq = tile_q0 + mi;
            if (gq >= Nq) continue;

            // compute qk^T for *this lane* across BLOCK_N keys

            float dots[BLOCK_N];
            #pragma unroll MAX_UNROLL
            for (int nk = 0; nk < BLOCK_N; ++nk)
                dots[nk] = qvec_local[mi] * bf16_to_fp32(store.K[nk][lane]);

            #pragma unroll MAX_UNROLL
            for (int nk = 0; nk < BLOCK_N; ++nk) 
                dots[nk] = warp_reduce_sum(dots[nk]);

            if (lane == 0) {
                float* logits = reinterpret_cast<float*>(store.V);
                #pragma unroll MAX_UNROLL
                for (int nk = 0; nk < BLOCK_N; ++nk) {
                    int32_t ki = store.Kidx[nk];
                    int32_t kh = store.Kh[nk];
                    float val = (qi_local[mi] >= ki &&
                                 qh_local[mi] == kh) 
                                 ? dots[nk] * sm_scale : -CUDART_INF_F;
                    logits[mi * BLOCK_N + nk] = val;
                }
            }
        }
        __syncthreads();
        
        // softmax-update and accumulate
        float* logits = reinterpret_cast<float*>(store.V);
        
        #pragma unroll MAX_UNROLL
        for (int mi = 0; mi < BLOCK_M; ++mi) {
            int gq = tile_q0 + mi; if (gq >= Nq) continue;
            if (lane >= BLOCK_N) continue; // spare lanes idle

            float logit = logits[mi * BLOCK_N + lane];
            float row_max = warp_reduce_max(logit);
            float ex = __expf(logit - row_max);
            float row_sum = warp_reduce_sum(ex);
            float p = ex / row_sum;

            // convert V to fp32 and accumulate
            // V[col == lane because thread==dim]
            float vcomp = bf16_to_fp32(store.V[lane][threadIdx.x]);
            float contrib = p * vcomp;

            contrib = warp_reduce_sum(contrib);

            if (lane == 0) {
                m_prev[mi] = fmaxf(m_prev[mi], row_max);
                l_prev[mi] = l_prev[mi] * __expf(m_prev[mi] - row_max) + row_sum;

                #pragma unroll MAX_UNROLL
                for (int d = 0; d < D_HEAD; ++d) acc[mi][d] += contrib;
            }
        }
        __syncthreads();
    }

    // store output + L/M
    if (lane == 0) {
        for (int mi = 0; mi < BLOCK_M; ++mi) {
            int gq = tile_q0 + mi; if (gq >= Nq) continue;
            Lh[gq] = l_prev[mi];
            Mh[gq] = m_prev[mi];

            #pragma unroll MAX_UNROLL
            for (int d = 0; d < D_HEAD; ++d) {
                Oh[gq * som + d] = acc[mi][d];
            }
        }
    }
}

// untested.
static void hash_sparse_forward_launch(const at::Tensor& Q,
                                       const at::Tensor& K,
                                       const at::Tensor& V,
                                       const at::Tensor& Q_idx,
                                       const at::Tensor& K_idx,
                                       const at::Tensor& Q_hash,
                                       const at::Tensor& K_hash,
                                       at::Tensor& Out,
                                       at::Tensor& L,
                                       at::Tensor& M,
                                       double sm_scale){
  TORCH_CHECK(Q.scalar_type()==at::kBFloat16,"Q must be bf16");
  int ZH = Q.size(0)*Q.size(1);   // already flattened outside in Python
  int Nq = Q.size(2);
  int Nk = K.size(2);

  dim3 grid(DIVUP(Nq,BLOCK_M), ZH);
  dim3 block(D_HEAD);
  size_t shmem = sizeof(SharedHashStore);

  hash_sparse_forward<<<grid,block,shmem,at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(V.data_ptr<at::BFloat16>()),
      Q_idx.data_ptr<int32_t>(), K_idx.data_ptr<int32_t>(),
      Q_hash.data_ptr<int32_t>(), K_hash.data_ptr<int32_t>(),
      static_cast<float>(sm_scale),
      Out.data_ptr<float>(), L.data_ptr<float>(), M.data_ptr<float>(),
                  Q.stride(2),Q.stride(3), K.stride(2),K.stride(3), V.stride(2),V.stride(3),
                  Out.stride(2),Out.stride(3),
                  Q_idx.stride(2), K_idx.stride(2), Q_hash.stride(2), K_hash.stride(2),
      Nq,Nk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor hash_sparse_forward_py(at::Tensor Q,at::Tensor K,at::Tensor V,
                                  at::Tensor Q_idx,at::Tensor K_idx,
                                  at::Tensor Q_hash,at::Tensor K_hash,
                                  double sm_scale){
  auto opts=Q.options().dtype(torch::kFloat32);
  auto Out=torch::zeros_like(Q,opts);
  auto L  =torch::empty({Q.size(0)*Q.size(1),Q.size(2)},opts);
  auto M  =torch::empty_like(L);
  hash_sparse_forward_launch(Q,K,V,Q_idx,K_idx,Q_hash,K_hash,Out,L,M,sm_scale);
  return Out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("forward", &hash_sparse_forward_py, "Hash‑sparse forward (bf16)");
}