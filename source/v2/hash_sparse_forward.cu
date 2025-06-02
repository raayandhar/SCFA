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
constexpr int BLOCK_M = 64; // queries per thread-block (rows)
constexpr int BLOCK_N = 64; // keys per thread-block (cols)
constexpr int D_HEAD = 64; // per-head embedding size
constexpr int MAX_UNROLL = 16;

// Shared-memory static buffers, with hash bucket ids now
struct SharedHashStore {
  __nv_bfloat16 q_vals[BLOCK_M * D_HEAD];
  int32_t qi_vals[BLOCK_M];
  int32_t qh_vals[BLOCK_M];
  int32_t ki_vals[BLOCK_N];
  int32_t kh_vals[BLOCK_N];
  __nv_bfloat16 k_vals[BLOCK_N * D_HEAD];
  __nv_bfloat16 v_vals[BLOCK_N * D_HEAD];
  __nv_bfloat16 qk_vals[BLOCK_M * BLOCK_N];
  __nv_bfloat16 p_vals[BLOCK_M * BLOCK_N];
};

__global__ void hash_sparse_forward(
  const __nv_bfloat16* __restrict__ Q, // Query matrix [Z, H, N_CTX_Q, D]
  const __nv_bfloat16* __restrict__ K, // Key matrix [Z, H, N_CTX_KV, D]
  const __nv_bfloat16* __restrict__ V, // Value matrix [Z, H, N_CTX_KV, D]
  const int32_t*       __restrict__ Q_idx, // Query index [Z, H, N_CTX_Q]
  const int32_t*       __restrict__ K_idx, // Key index [Z, H, N_CTX_KV]
  const int32_t*       __restrict__ Q_hash, // Query hash [Z, H, N_CTX_Q]
  const int32_t*       __restrict__ K_hash, // Key hash [Z, H, N_CTX_KV]
  const float                       sm_scale, // Softmax scaling (1/sqrt(D))
  __nv_bfloat16*       __restrict__ O, // Output matrix [Z, H, N_CTX_Q, D] - NOW BFLOAT16!
  float*               __restrict__ L, // Logits [Z*H, N_CTX_Q] - Keep float for precision
  float*               __restrict__ M, // Max logits [Z*H, N_CTX_Q] - Keep float for precision
  int64_t sqz, int64_t sqh, int64_t sqm, int64_t sqd, // Q: (batch, head, seq_length, dim)
  int64_t skz, int64_t skh, int64_t skn, int64_t skd,  // K: (batch, head, seq_length, dim)
  int64_t svz, int64_t svh, int64_t svn, int64_t svd, // V: (batch, head, seq_length, dim)
  int64_t soz, int64_t soh, int64_t som, int64_t sod, // O: (batch, head, seq_length, dim)
  int64_t sqiz, int64_t sqih, int64_t sqim, // Q_idx: (batch, head, seq_length)
  int64_t skiz, int64_t skih, int64_t skin, // K_idx: (batch, head, seq_length)
  int64_t sqhz, int64_t sqhh, int64_t sqhm, // Q_hash: (batch, head, seq_length)     
  int64_t skhz, int64_t skhh, int64_t skhn, // K_hash: (batch, head, seq_length)
  // Dimension key
  int Z, int H, int N_CTX_Q, int N_CTX_KV) 
{    
  extern __shared__ SharedHashStore shmem[];
  SharedHashStore& store = *shmem;
  const int tx = threadIdx.x;
  const int start_m = blockIdx.x;
  const int off_hz = blockIdx.y;
  const int tile_q0 = start_m * BLOCK_M;
  
  if (tile_q0 >= N_CTX_Q) return;

  const int batch_idx = off_hz / H;
  const int head_idx = off_hz % H;
  
  // Base pointers for this specific batch and head
  const __nv_bfloat16* Qh = Q + batch_idx * sqz + head_idx * sqh;
  const __nv_bfloat16* Kh = K + batch_idx * skz + head_idx * skh;
  const __nv_bfloat16* Vh = V + batch_idx * svz + head_idx * svh;
  const int32_t*      QI = Q_idx  + batch_idx * sqiz + head_idx * sqih;
  const int32_t*      KI = K_idx  + batch_idx * skiz + head_idx * skih;
  const int32_t*      QH = Q_hash + batch_idx * sqhz + head_idx * sqhh;
  const int32_t*      KH = K_hash + batch_idx * skhz + head_idx * skhh;
  __nv_bfloat16*      Oh = O      + batch_idx * soz + head_idx * soh;
  float*              Lh = L      + off_hz * N_CTX_Q;
  float*              Mh = M      + off_hz * N_CTX_Q;

  float m_prev[BLOCK_M], l_prev[BLOCK_M]; 
  float acc[BLOCK_M][D_HEAD]; 
  
  for (int mi = tx; mi < BLOCK_M; mi += blockDim.x) {
    m_prev[mi] = -CUDART_INF_F;
    l_prev[mi] = 0.0f;
    #pragma unroll MAX_UNROLL
    for (int d = 0; d < D_HEAD; ++d) {
      acc[mi][d] = 0.0f;
    }
  }

  // load q_vals, qi, qh
  for (int mi = tx; mi < BLOCK_M; mi += blockDim.x){
    if (tile_q0 + mi >= N_CTX_Q) continue;
    #pragma unroll MAX_UNROLL
    for (int d = 0; d < D_HEAD; ++d) {
      store.q_vals[mi * D_HEAD + d] = Qh[(tile_q0 + mi) * sqm + d * sqd];
      //store.k_vals[mi * D_HEAD + d] = Kh[(tile_q0 + mi) * skn + d * skd];
      //store.v_vals[mi * D_HEAD + d] = Vh[(tile_q0 + mi) * svn + d * svd];
    }
    store.qi_vals[mi] = QI[(tile_q0 + mi) * sqim];
    store.qh_vals[mi] = QH[(tile_q0 + mi) * sqhm];
    store.kh_vals[mi] = KH[(tile_q0 + mi) * skhn];
    store.ki_vals[mi] = KI[(tile_q0 + mi) * skin];
  }
  __syncthreads();

  int max_qi = -1, min_qh = INT_MAX, max_qh = -1;
  int min_kh = 1e9, max_kh = -1e9, min_ki = INT_MAX;
  for (int mi = 0; mi < BLOCK_M; ++mi) {
    if (tile_q0 + mi >= N_CTX_Q) continue;
    max_qi = max(max_qi, store.qi_vals[mi]);
    min_qh = min(min_qh, store.qh_vals[mi]);
    max_qh = max(max_qh, store.qh_vals[mi]);
    min_kh = min(min_kh, store.kh_vals[mi]);
    max_kh = max(max_kh, store.kh_vals[mi]);
    min_ki = min(min_ki, store.ki_vals[mi]);
  }

  int end_n = 0, start_n = 0;

  // find start and end blocks
  //int min_kh = 1e9, max_kh = -1e9, min_ki = INT_MAX;
  const int N_CTX_KV_BLOCKS = DIVUP(N_CTX_KV, BLOCK_N);
  for (int i = 0; i < N_CTX_KV_BLOCKS; ++i){
    if (min_kh <= max_qh && min_kh != 1e9) {
      end_n++;
    }
    if (max_kh < min_qh && max_kh != -1e9) {
      start_n++;
    }
  }

  int causal_end_n = end_n;

  // remove unnecessary trailing blocks based on causal structure
  for (int block_idx = start_n; block_idx < end_n; ++block_idx) {
    if (min_ki <= max_qi && min_ki != 1e9) {
      causal_end_n = block_idx + 1;
    }
  }

  // main attention loop
  for (int block_k = start_n; block_k < causal_end_n; ++block_k) {
    int tile_k0 = block_k * BLOCK_N;
    
    // load values for K and K_idx and K_hash
    for (int ki = tx; ki < BLOCK_N; ki += blockDim.x) {
      if (tile_k0 + ki < N_CTX_KV) {
        store.ki_vals[ki] = KI[(tile_k0 + ki) * skin];
        store.kh_vals[ki] = KH[(tile_k0 + ki) * skhn];
        
        #pragma unroll MAX_UNROLL
        for (int d = 0; d < D_HEAD; ++d) {
          store.k_vals[ki * D_HEAD + d] = Kh[(tile_k0 + ki) * skn + d * skd];
        }
      } else {
        store.ki_vals[ki] = 1e9;
        store.kh_vals[ki] = -1e9;
      }
    }
    
    // load v_vals
    for (int ki = tx; ki < BLOCK_N; ki += blockDim.x) {
      if (tile_k0 + ki < N_CTX_KV) {
        #pragma unroll MAX_UNROLL
        for (int d = 0; d < D_HEAD; ++d) {
          store.v_vals[ki * D_HEAD + d] = Vh[(tile_k0 + ki) * svn + d * svd];
        }
      }
    }
    __syncthreads();
    
    for (int mi = tx; mi < BLOCK_M; mi += blockDim.x) {
      if (tile_q0 + mi >= N_CTX_Q) continue;
      float row_max = -CUDART_INF_F;
      #pragma unroll MAX_UNROLL
      for (int ki = 0; ki < BLOCK_N; ++ki) {
        if (tile_k0 + ki >= N_CTX_KV) {
          store.qk_vals[mi * BLOCK_N + ki] = fp32_to_bf16(-CUDART_INF_F);
          continue;
        }
       
        // compute qk
        float qk_score = 0.0f;
        #pragma unroll MAX_UNROLL
        for (int d = 0; d < D_HEAD; ++d) {
          //qk_score += bf16_to_fp32(store.q_vals[mi * D_HEAD + d]) * bf16_to_fp32(store.k_vals[ki * D_HEAD + d]);
          //qk_score += fmaf(bf16_to_fp32(store.q_vals[mi * D_HEAD + d]), bf16_to_fp32(store.k_vals[ki * D_HEAD + d]), qk_score);
          qk_score += bf16_to_fp32(__hmul(store.q_vals[mi * D_HEAD + d], store.k_vals[ki * D_HEAD + d]));
        }
        qk_score *= sm_scale;
        
        // causal and hash masking
        bool causal_mask = (store.qi_vals[mi] >= store.ki_vals[ki]);
        bool hash_mask = (store.qh_vals[mi] == store.kh_vals[ki]);
        if (causal_mask && hash_mask) {
          store.qk_vals[mi * BLOCK_N + ki] = fp32_to_bf16(qk_score);
          row_max = fmaxf(row_max, qk_score);
        } else {
          store.qk_vals[mi * BLOCK_N + ki] = fp32_to_bf16(-CUDART_INF_F);
        }
      }
      
      // compute attention weights
      float m_curr = fmaxf(m_prev[mi], row_max);
      float m_curr_ = (m_curr != -CUDART_INF_F) ? m_curr : 0.0f;
      l_prev[mi] *= expf(m_prev[mi] - m_curr_);
      float l_curr = 0.0f;
      for (int ki = 0; ki < BLOCK_N; ++ki) {
        float qk_f32 = bf16_to_fp32(store.qk_vals[mi * BLOCK_N + ki]);
        float p = expf(qk_f32 - m_curr_);
        store.p_vals[mi * BLOCK_N + ki] = fp32_to_bf16(p);
        l_curr += p;
      }
      l_curr += l_prev[mi];
      float l_rcp = fdividef(1.0f, l_curr);
      
      #pragma unroll MAX_UNROLL
      for (int d = 0; d < D_HEAD; ++d) {
        acc[mi][d] *= (l_prev[mi] * l_rcp);
      }
     
      // update acc
      #pragma unroll MAX_UNROLL
      for (int d = 0; d < D_HEAD; ++d) {
        float inner_prod = 0.0f;
        for (int ki = 0; ki < BLOCK_N; ++ki) {
          float p = bf16_to_fp32(store.p_vals[mi * BLOCK_N + ki]) * l_rcp; // make this faster
          //float p = bf16_to_fp32(__hmul(store.p_vals[mi * BLOCK_N + ki], l_rcp)); // make this faster
          inner_prod += p * bf16_to_fp32(store.v_vals[ki * D_HEAD + d]);
        }
        acc[mi][d] += inner_prod;
      }
      l_prev[mi] = l_curr;
      m_prev[mi] = m_curr;
    }
    //__syncthreads();
  }

  // store results
  for (int mi = tx; mi < BLOCK_M; mi += blockDim.x) {
    if (tile_q0 + mi < N_CTX_Q) {
      #pragma unroll MAX_UNROLL
      for (int d = 0; d < D_HEAD; ++d) {
        Oh[(tile_q0 + mi) * som + d * sod] = fp32_to_bf16(acc[mi][d]);
      }
      Lh[tile_q0 + mi] = l_prev[mi];
      Mh[tile_q0 + mi] = m_prev[mi];
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
  TORCH_CHECK(Out.scalar_type()==at::kBFloat16,"Out must be bf16"); 
  int Z = Q.size(0);         
  int H = Q.size(1);         
  int N_CTX_Q = Q.size(2);   
  int N_CTX_KV = K.size(2);  

  dim3 grid(DIVUP(N_CTX_Q, BLOCK_M), Z * H);
  dim3 block(BLOCK_M); // parallelize block_m/block_n over a thread block
  size_t shmem = sizeof(SharedHashStore);

  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
//  printf("Max shared memory: %d bytes, requested shared memory: %zu bytes\n", max_sram_size, shmem);

  hash_sparse_forward<<<grid,block,shmem,at::cuda::getCurrentCUDAStream()>>>(
    reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr<at::BFloat16>()),
    reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>()),
    reinterpret_cast<const __nv_bfloat16*>(V.data_ptr<at::BFloat16>()),
    Q_idx.data_ptr<int32_t>(), K_idx.data_ptr<int32_t>(),
    Q_hash.data_ptr<int32_t>(), K_hash.data_ptr<int32_t>(),
    static_cast<float>(sm_scale),
    reinterpret_cast<__nv_bfloat16*>(Out.data_ptr<at::BFloat16>()), 
    L.data_ptr<float>(), M.data_ptr<float>(),
    Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
    K.stride(0), K.stride(1), K.stride(2), K.stride(3),
    V.stride(0), V.stride(1), V.stride(2), V.stride(3),
    Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
    Q_idx.stride(0), Q_idx.stride(1), Q_idx.stride(2),
    K_idx.stride(0), K_idx.stride(1), K_idx.stride(2),
    Q_hash.stride(0), Q_hash.stride(1), Q_hash.stride(2),
    K_hash.stride(0), K_hash.stride(1), K_hash.stride(2),
    Z, H, N_CTX_Q, N_CTX_KV);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor hash_sparse_forward_py(at::Tensor Q, at::Tensor K, at::Tensor V,
                                  at::Tensor Q_idx, at::Tensor K_idx,
                                  at::Tensor Q_hash, at::Tensor K_hash,
                                  double sm_scale){
  int64_t Z = Q.size(0);     
  int64_t H = Q.size(1);     
  int64_t N_CTX_Q = Q.size(2);
  int64_t d = Q.size(3);     
  
  auto Out = torch::zeros_like(Q); 
  
  auto opts_float = Q.options().dtype(torch::kFloat32);
  auto L = torch::empty({Z * H, N_CTX_Q}, opts_float);
  auto M = torch::empty_like(L);
  
  hash_sparse_forward_launch(Q, K, V, Q_idx, K_idx, Q_hash, K_hash, Out, L, M, sm_scale);
  return Out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("forward", &hash_sparse_forward_py, "Hash‑sparse forward (bf16)");
}