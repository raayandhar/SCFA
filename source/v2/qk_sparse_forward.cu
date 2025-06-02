// qk_sparse_forward.cu
// This is based on the dynamic-sparse-flash-attention paper.
// I am porting their qk-sparse triton kernel to CUDA.
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
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#define DIVUP(x, y) (((x) + (y) - 1) / (y))

// Warp-level reductions.
__device__ inline float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}
__device__ inline float warp_reduce_min(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fminf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}
__device__ inline float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

// Conversion helpers.
__device__ inline float bf16_to_fp32(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ inline __nv_bfloat16 fp32_to_bf16(float x) { return __float2bfloat16(x); }

//  Kernel-tunable parameters.
constexpr int BLOCK_M = 128; // queries per thread-block (rows)
constexpr int BLOCK_N = 128; // keys per thread-block (cols)
constexpr int D_HEAD  = 64;  // per-head embedding size

// Shared-memory static buffers
struct SharedStore {
    __nv_bfloat16 K[BLOCK_N][D_HEAD];
    __nv_bfloat16 V[BLOCK_N][D_HEAD];
    int32_t    Kidx[BLOCK_N];
};

__global__ void qk_sparse_forward(
    const __nv_bfloat16* __restrict__ Q, // Query matrix [H, N_CTX_Q, D]
    const __nv_bfloat16* __restrict__ K, // Key matrix [H, N_CTX_KV, D]
    const __nv_bfloat16* __restrict__ V, // Value matrix [H, N_CTX_KV, D]
    const int32_t*       __restrict__ Q_idx, // Query index [H, N_CTX_Q]
    const int32_t*       __restrict__ K_idx, // Key index [H, N_CTX_KV]
    const float                       sm_scale, // Softmax scaling (1/sqrt(D))
    float*               __restrict__ O, // Output matrix [H, N_CTX_Q, D]
    float*               __restrict__ L, // Logits [H, N_CTX_Q]
    float*               __restrict__ M, // Max logits [H, N_CTX_Q]
    // Strides, so we can be flexible about transposes and layouts.
    int64_t sqh, int64_t sqm, int64_t sqd,
    int64_t skh, int64_t skn, int64_t skd,
    int64_t svh, int64_t svn, int64_t svd,
    int64_t soh, int64_t som, int64_t sod,
    int64_t sqih, int64_t sqim,
    int64_t skih, int64_t skin,
    // Dimension key
    int Z, int H, int N_CTX_Q, int N_CTX_KV) 
{    
    extern __shared__ SharedStore shmem[];
    SharedStore& store = *shmem;

    const int tid = threadIdx.x; // Assume 0-31, also for warpId and lane.
    const int warpId = tid >> 5;
    const int lane = tid & 31;

    const int block_q0 = blockIdx.x * BLOCK_M; // First query for the block.
    const int head_id = blockIdx.y; // Flattened z*h index.

    if (block_q0 >= N_CTX_Q) return; // OOB; return early. TODO: throw an error.

    // Base pointers for this head.
    const __nv_bfloat16* Qh = Q + head_id * sqh;
    const __nv_bfloat16* Kh = K + head_id * skh;
    const __nv_bfloat16* Vh = V + head_id * svh;

    const int32_t* Qi = Q_idx + head_id * sqih;
    const int32_t* Ki = K_idx + head_id * skih;

    float* Oh = O + head_id * soh;
    float* Lh = L + head_id * N_CTX_Q;
    float* Mh = M + head_id * N_CTX_Q;
    
    int32_t qi_local[BLOCK_M];
    float q_local[BLOCK_M];

    #pragma unroll
    for (int mi = 0; mi < BLOCK_M; ++mi) {
        int q_idx = block_q0 + mi;
        int qvalid = q_idx < N_CTX_Q;
        qi_local[mi] = qvalid ? Qi[q_idx * sqim] : -1;

        // Gather this dim (lane) of query mi.
        if (qvalid) {
            __nv_bfloat16 qv = Qh[q_idx * sqm + lane * sqd];
            q_local[mi] = bf16_to_fp32(qv);
        // If it's not qvalid we maybe just kill the computation?
        } else {
            q_local[mi] = 0.0f;
        }
    }

    // Compute max qi in the tile for early pruning cooperatively.
    int max_qi_tile = -1;
    #pragma unroll
    for (int mi = 0; mi < BLOCK_M; ++mi) max_qi_tile =max(max_qi_tile, qi_local[mi]);
    max_qi_tile = warp_reduce_max(max_qi_tile);


    const int num_key_blocks = DIVUP(N_CTX_KV, BLOCK_N);
    for (int nb = 0; nb < num_key_blocks; ++nb) {
        const int block_k0 = nb * BLOCK_N;

        // 1. Load key indices into smem and compute the min index in the tile
        int local_min_ki = 1e9;
        if (lane < BLOCK_N) {
            int gk = block_k0 + lane;
            int32_t val = (gk < N_CTX_KV) ? Ki[gk * skin] : 1e9;
            store.Kidx[lane] = val;
            local_min_ki = val;
        }
        local_min_ki = warp_reduce_min(local_min_ki);
        __syncthreads();
        // Recall that when (min_ki > max_qi), no query in the tile can attend to any key in this block.
        if (local_min_ki > max_qi_tile) continue;

        // 2. Load keys and values into smem.
        for (int n = lane; n < BLOCK_N * D_HEAD; n += 32) {
            const int row = n / D_HEAD;
            const int col = n % D_HEAD;
            const int gk = block_k0 + row;
            __nv_bfloat16 kv = (gk < N_CTX_KV)
                 ? Kh[gk * skn + col * skd]
                 : __float2bfloat16(0.0f);

            __nv_bfloat16 vv = (gk < N_CTX_KV)
                 ? Vh[gk * svn + col * svd]
                 : __float2bfloat16(0.0f);

            store.K[row][col] = kv;
            store.V[row][col] = vv;
        }
        __syncthreads();

        // 3. Process the current query in the tile.
        #pragma unroll
        for (int mi = 0; mi < BLOCK_M; ++mi) {
            int q_global = block_q0 + mi;
            if (q_global >= N_CTX_Q) continue;

            float qd = q_local[mi];
            float dot_partial[BLOCK_N];
            #pragma unroll
            for (int nk = 0; nk < BLOCK_N; ++nk) 
                dot_partial[nk] = qd * bf16_to_fp32(store.K[nk][lane]);

            // Reduce over D_HEAD dims with warp_reduce_sum.
            #pragma unroll
            for (int nk = 0; nk < BLOCK_N; ++nk) {
                float dotv = warp_reduce_sum(dot_partial[nk]); // every lane contains a full dot (q, k_n)
                if (lane == 0) {
                    // Causal sparse masking
                    int32_t ki = store.Kidx[nk];
                    float logit = (qi_local[mi] < ki) ? dotv * sm_scale : -1e9;
                    // 128x128 float scratchpad for logits, but we get V reuse.
                    reinterpret_cast<float*>(store.V)[mi * BLOCK_N + nk] = logit;                    
                }
            }
            __syncthreads();

            // 4. Compute the max logit for this query.
            #pragma unroll
            for (int mi = 0; mi < BLOCK_M; ++mi) {
                int q_global = block_q0 + mi;
                if (q_global >= N_CTX_Q) continue;
                if (lane >= BLOCK_N) return;

                float logit = reinterpret_cast<float*>(store.V)[mi * BLOCK_N + lane];
                float row_max = warp_reduce_max(logit);
                float ex = __expf(logit - row_max);
                float row_sum = warp_reduce_sum(ex);
                float prob = ex / row_sum;
                
                // Multiply by the corresponding V value in smem (bf16->fp32 translation), then accumulate into O.
                float v_comp = bf16_to_fp32(store.V[lane][threadIdx.x]);
                float contrib = prob * v_comp;
                contrib = warp_reduce_sum(contrib);
                if (lane == 0) {
                    Oh[q_global * som + head_id * sod] += contrib; // additive across key-tiles
                    Lh[q_global] = row_sum;
                    Mh[q_global] = row_max;
                }                
            }
            __syncthreads();
        }
    }
}

// Launch for testing.
/*
static void qk_sparse_forward_launch(const at::Tensor& Q,
                                     const at::Tensor& K,
                                     const at::Tensor& V,
                                     const at::Tensor& Q_idx,
                                     const at::Tensor& K_idx,
                                     at::Tensor& Out,
                                     at::Tensor& L,
                                     at::Tensor& M,
                                     double sm_scale_d) {
  TORCH_CHECK(Q.scalar_type() == at::kBFloat16, "Q/K/V must be bf16");
  TORCH_CHECK(Q.is_cuda(), "tensors must be on CUDA device");
  const int Z = Q.size(0);
  const int H = Q.size(1);
  const int Nq = Q.size(2);
  const int Nk = K.size(2);
  const int Dh = Q.size(3);
  TORCH_CHECK(Dh == D_HEAD, "only D_HEAD=64 supported in first pass");

  dim3 grid(DIVUP(Nq, BLOCK_M), Z * H);
  dim3 block(D_HEAD);  // one thread per dim (64) – OK because shared‑mem fits

  const size_t shmem_sz = sizeof(SharedStore);

  qk_sparse_forward<<<grid, block, shmem_sz, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(V.data_ptr<at::BFloat16>()),
      Q_idx.data_ptr<int32_t>(),
      K_idx.data_ptr<int32_t>(),
      static_cast<float>(sm_scale_d),
      Out.data_ptr<float>(),
      L.data_ptr<float>(),
      M.data_ptr<float>(),
      // strides (flattened inside head‑major layout (Z,H,N,D))
      Q.stride(1), Q.stride(2), Q.stride(3),
      K.stride(1), K.stride(2), K.stride(3),
      V.stride(1), V.stride(2), V.stride(3),
      Out.stride(1), Out.stride(2), Out.stride(3),
      Q_idx.stride(1), Q_idx.stride(2),
      K_idx.stride(1), K_idx.stride(2),
      Z, H, Nq, Nk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor qk_sparse_forward_py(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                   torch::Tensor Q_idx, torch::Tensor K_idx, double sm_scale) {
  TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
              "Q/K/V must be contiguous (transpose beforehand)");

  auto opts_fp32 = Q.options().dtype(torch::kFloat32);
  auto Out = torch::zeros_like(Q, opts_fp32);
  auto L   = torch::empty({Q.size(0) * Q.size(1), Q.size(2)}, opts_fp32);
  auto M   = torch::empty({Q.size(0) * Q.size(1), Q.size(2)}, opts_fp32);

  qk_sparse_forward_launch(Q, K, V, Q_idx, K_idx, Out, L, M, sm_scale);
  return Out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qk_sparse_forward", &qk_sparse_forward_py, "QK‑sparse forward (bf16, causal)");
  // TODO: expose backward once written
}
*/

    










