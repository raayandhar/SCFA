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
#include <mma.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cstdint>
#include <cuda_runtime.h>
#include <math_constants.h>

#define DIVUP(x, y) (((x) + (y) - 1) / (y))

using namespace nvcuda;

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
constexpr int BLOCK_M = 64; // queries per thread-block
constexpr int BLOCK_N = 64; // keys per thread-block
constexpr int D_HEAD = 64; // per-head embedding size
constexpr int MAX_UNROLL = 16;
const int WMMA_M = 16; // wmma M dimension
const int WMMA_N = 16; // wmma N dimension
const int WMMA_K = 16; // wmma K dimension

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
  // ideally both would be the size of BLOCK_M * BLOCK_N
  float qk_wmma[WMMA_M * WMMA_N];
  float pv_wmma[WMMA_M * WMMA_N]; 
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
  float acc[D_HEAD * BLOCK_M]; 
  
  for (int mi = tx; mi < BLOCK_M; mi += blockDim.x) {
    m_prev[mi] = -CUDART_INF_F;
    l_prev[mi] = 0.0f;
    for (int d = 0; d < D_HEAD; ++d) {
      acc[d * BLOCK_M + mi] = 0.0f;
    }
  }

  // load q_vals, qi, qh
  for (int mi = tx; mi < BLOCK_M; mi += blockDim.x){
    if (tile_q0 + mi >= N_CTX_Q) continue;
    for (int d = 0; d < D_HEAD; d+=16) {
      // load 16 qvals at a time
      const __nv_bfloat162* bf16_ptr = reinterpret_cast<const __nv_bfloat162*>(Qh + (tile_q0 + mi) * sqm + d * sqd);
      __nv_bfloat162 bf16_4[8];  

      bf16_4[0] = *(bf16_ptr + 0);
      bf16_4[1] = *(bf16_ptr + 1);
      bf16_4[2] = *(bf16_ptr + 2);
      bf16_4[3] = *(bf16_ptr + 3);
      bf16_4[4] = *(bf16_ptr + 4);
      bf16_4[5] = *(bf16_ptr + 5);
      bf16_4[6] = *(bf16_ptr + 6);
      bf16_4[7] = *(bf16_ptr + 7);
      
      store.q_vals[(d + 0) * BLOCK_M + mi] = bf16_4[0].x;
      store.q_vals[(d + 1) * BLOCK_M + mi] = bf16_4[0].y;
      store.q_vals[(d + 2) * BLOCK_M + mi] = bf16_4[1].x;
      store.q_vals[(d + 3) * BLOCK_M + mi] = bf16_4[1].y;
      store.q_vals[(d + 4) * BLOCK_M + mi] = bf16_4[2].x;
      store.q_vals[(d + 5) * BLOCK_M + mi] = bf16_4[2].y;
      store.q_vals[(d + 6) * BLOCK_M + mi] = bf16_4[3].x;
      store.q_vals[(d + 7) * BLOCK_M + mi] = bf16_4[3].y;
      store.q_vals[(d + 8) * BLOCK_M + mi] = bf16_4[4].x;
      store.q_vals[(d + 9) * BLOCK_M + mi] = bf16_4[4].y;
      store.q_vals[(d + 10) * BLOCK_M + mi] = bf16_4[5].x;
      store.q_vals[(d + 11) * BLOCK_M + mi] = bf16_4[5].y;
      store.q_vals[(d + 12) * BLOCK_M + mi] = bf16_4[6].x;
      store.q_vals[(d + 13) * BLOCK_M + mi] = bf16_4[6].y;
      store.q_vals[(d + 14) * BLOCK_M + mi] = bf16_4[7].x;
      store.q_vals[(d + 15) * BLOCK_M + mi] = bf16_4[7].y;
    }
    store.qi_vals[mi] = QI[(tile_q0 + mi) * sqim];
    store.qh_vals[mi] = QH[(tile_q0 + mi) * sqhm];
  }
  __syncthreads();

  int max_qi = -1, min_qh = INT_MAX, max_qh = -1;
  int min_kh = 1e9, max_kh = -1e9, min_ki = INT_MAX;
  for (int mi = 0; mi < BLOCK_M; ++mi) {
    if (tile_q0 + mi >= N_CTX_Q) continue;
    max_qi = max(max_qi, store.qi_vals[mi]);
    min_qh = min(min_qh, store.qh_vals[mi]);
    max_qh = max(max_qh, store.qh_vals[mi]);
  }

  int end_n = 0, start_n = 0;

  // find start and end blocks
  const int N_CTX_KV_BLOCKS = DIVUP(N_CTX_KV, BLOCK_N);
  for (int i = 0; i < N_CTX_KV_BLOCKS; ++i){
    for (int ki = 0; ki < BLOCK_N; ++ki) {
      int tile_k0 = i * BLOCK_N;
      if (tile_k0 + ki < N_CTX_KV) {
        min_kh = min(min_kh, KH[(tile_k0 + ki) * skhn]);
        max_kh = max(max_kh, KH[(tile_k0 + ki) * skhn]);
        min_ki = min(min_ki, KI[(tile_k0 + ki) * skin]);
      }
    }
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
    int tile_k0 = block_idx * BLOCK_N;
    int min_ki = INT_MAX;
    for (int ki = 0; ki < BLOCK_N; ++ki) {
      if (tile_k0 + ki < N_CTX_KV) {
        min_ki = min(min_ki, KI[(tile_k0 + ki) * skin]);
      }
    }
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
      } else {
        store.ki_vals[ki] = 1e9;
        store.kh_vals[ki] = -1e9;
      }
    }
    
    // load values for K and V
    for (int ki = tx; ki < BLOCK_N; ki += blockDim.x) {
      for (int d = 0; d < D_HEAD; d+=16) {
        if (tile_k0 + ki < N_CTX_KV) {
          const __nv_bfloat162* bf16_ptr_k = reinterpret_cast<const __nv_bfloat162*>(Kh + (tile_k0 + ki) * skn + d * skd);
          const __nv_bfloat162* bf16_ptr_v = reinterpret_cast<const __nv_bfloat162*>(Vh + (tile_k0 + ki) * svn + d * svd);
          __nv_bfloat162 bf16_4_k[8];
          __nv_bfloat162 bf16_4_v[8];
          bf16_4_k[0] = *(bf16_ptr_k + 0);
          bf16_4_k[1] = *(bf16_ptr_k + 1);
          bf16_4_k[2] = *(bf16_ptr_k + 2);
          bf16_4_k[3] = *(bf16_ptr_k + 3);
          bf16_4_k[4] = *(bf16_ptr_k + 4);
          bf16_4_k[5] = *(bf16_ptr_k + 5);
          bf16_4_k[6] = *(bf16_ptr_k + 6);
          bf16_4_k[7] = *(bf16_ptr_k + 7);

          bf16_4_v[0] = *(bf16_ptr_v + 0);
          bf16_4_v[1] = *(bf16_ptr_v + 1);
          bf16_4_v[2] = *(bf16_ptr_v + 2);
          bf16_4_v[3] = *(bf16_ptr_v + 3);
          bf16_4_v[4] = *(bf16_ptr_v + 4);
          bf16_4_v[5] = *(bf16_ptr_v + 5);
          bf16_4_v[6] = *(bf16_ptr_v + 6);
          bf16_4_v[7] = *(bf16_ptr_v + 7);

          store.k_vals[d * BLOCK_N + ki] = bf16_4_k[0].x;
          store.k_vals[(d+1) * BLOCK_N + ki] = bf16_4_k[0].y;
          store.k_vals[(d+2) * BLOCK_N + ki] = bf16_4_k[1].x;
          store.k_vals[(d+3) * BLOCK_N + ki] = bf16_4_k[1].y;
          store.k_vals[(d+4) * BLOCK_N + ki] = bf16_4_k[2].x;
          store.k_vals[(d+5) * BLOCK_N + ki] = bf16_4_k[2].y;
          store.k_vals[(d+6) * BLOCK_N + ki] = bf16_4_k[3].x;
          store.k_vals[(d+7) * BLOCK_N + ki] = bf16_4_k[3].y;
          store.k_vals[(d+8) * BLOCK_N + ki] = bf16_4_k[4].x;
          store.k_vals[(d+9) * BLOCK_N + ki] = bf16_4_k[4].y;
          store.k_vals[(d+10) * BLOCK_N + ki] = bf16_4_k[5].x;
          store.k_vals[(d+11) * BLOCK_N + ki] = bf16_4_k[5].y;
          store.k_vals[(d+12) * BLOCK_N + ki] = bf16_4_k[6].x;
          store.k_vals[(d+13) * BLOCK_N + ki] = bf16_4_k[6].y;
          store.k_vals[(d+14) * BLOCK_N + ki] = bf16_4_k[7].x;
          store.k_vals[(d+15) * BLOCK_N + ki] = bf16_4_k[7].y;

          store.v_vals[d * BLOCK_N + ki] = bf16_4_v[0].x;
          store.v_vals[(d+1) * BLOCK_N + ki] = bf16_4_v[0].y;
          store.v_vals[(d+2) * BLOCK_N + ki] = bf16_4_v[1].x;
          store.v_vals[(d+3) * BLOCK_N + ki] = bf16_4_v[1].y;
          store.v_vals[(d+4) * BLOCK_N + ki] = bf16_4_v[2].x;
          store.v_vals[(d+5) * BLOCK_N + ki] = bf16_4_v[2].y;
          store.v_vals[(d+6) * BLOCK_N + ki] = bf16_4_v[3].x;
          store.v_vals[(d+7) * BLOCK_N + ki] = bf16_4_v[3].y;
          store.v_vals[(d+8) * BLOCK_N + ki] = bf16_4_v[4].x;
          store.v_vals[(d+9) * BLOCK_N + ki] = bf16_4_v[4].y;
          store.v_vals[(d+10) * BLOCK_N + ki] = bf16_4_v[5].x;
          store.v_vals[(d+11) * BLOCK_N + ki] = bf16_4_v[5].y;
          store.v_vals[(d+12) * BLOCK_N + ki] = bf16_4_v[6].x;
          store.v_vals[(d+13) * BLOCK_N + ki] = bf16_4_v[6].y;
          store.v_vals[(d+14) * BLOCK_N + ki] = bf16_4_v[7].x;
          store.v_vals[(d+15) * BLOCK_N + ki] = bf16_4_v[7].y;
        }
      }
    }
    __syncthreads();

    float row_max = -CUDART_INF_F;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> q_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> k_frag; // for some reason, wmma::row_major is needed for k_frag
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> qk_frag; // (WMMA_M, WMMA_N)
    for (int tile_m = 0; tile_m < BLOCK_M; tile_m += WMMA_M) {
      for (int tile_k = 0; tile_k < BLOCK_N; tile_k += WMMA_N) {
        wmma::fill_fragment(qk_frag, 0.0f);
        for (int d = 0; d < D_HEAD; d += WMMA_K) {
          wmma::load_matrix_sync(q_frag, &store.q_vals[d * BLOCK_M + tile_m], BLOCK_M);
          wmma::load_matrix_sync(k_frag, &store.k_vals[d * BLOCK_N + tile_k], BLOCK_N);
          wmma::mma_sync(qk_frag, q_frag, k_frag, qk_frag);
        }
        wmma::store_matrix_sync(store.qk_wmma, qk_frag, WMMA_N, wmma::mem_row_major);

        // apply masking
        for (int i = 0; i < 16; i++) {
          for (int j = 0; j < 16; j++) {
            int ki = tile_k + j;
            int mi = tile_m + i;
            if (tile_q0 + mi < N_CTX_Q && tile_k0 + ki < N_CTX_KV) {
              int mi_qi = store.qi_vals[mi];
              int ki_KI = store.ki_vals[ki];
              int qh_QH = store.qh_vals[mi];
              int kh_KH = store.kh_vals[ki];
              
              bool causal_mask = (mi_qi >= ki_KI);       
              bool hash_mask = (qh_QH == kh_KH);         
              if (causal_mask && hash_mask) {
                float qk_score = store.qk_wmma[i * 16 + j] * sm_scale;
                store.qk_vals[ki * BLOCK_M + mi] = fp32_to_bf16(qk_score);
                row_max = fmaxf(row_max, qk_score);
              } else {
                store.qk_vals[ki * BLOCK_M + mi] = fp32_to_bf16(-CUDART_INF_F);
              }
            } 
          }
        }
      }
    }
    //__syncthreads();

    // compute attention weights
    for (int mi = tx; mi < BLOCK_M; mi += blockDim.x) {
      float m_curr = fmaxf(m_prev[mi], row_max);
      float m_curr_ = (m_curr != -CUDART_INF_F) ? m_curr : 0.0f;
      l_prev[mi] *= expf(m_prev[mi] - m_curr_);
      float l_curr = 0.0f;
      for (int ki = 0; ki < BLOCK_N; ++ki) {
        float qk_f32 = bf16_to_fp32(store.qk_vals[ki * BLOCK_M + mi]);
        float p = expf(qk_f32 - m_curr_);
        store.p_vals[ki * BLOCK_M + mi] = fp32_to_bf16(p);
        l_curr += p;
      }
      l_curr += l_prev[mi];
      float l_rcp = fdividef(1.0f, l_curr);
      
      for (int d = 0; d < D_HEAD; ++d) {
        acc[d*BLOCK_M+mi] *= (l_prev[mi] * l_rcp);
      }

      for(int ki = 0; ki < BLOCK_N; ++ki){
        float p = bf16_to_fp32(store.p_vals[ki * BLOCK_M + mi]) * l_rcp;
        store.p_vals[ki*BLOCK_M+mi] = fp32_to_bf16(p);
      }
      l_prev[mi] = l_curr;
      m_prev[mi] = m_curr;
    }
    __syncthreads();

    // update acc (pv)
    // WMMA_M | BLOCK_M
    // WMMA_N | D_HEAD
    // WMMA_K | BLOCK_N

    // p,v are transposed
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> p_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> v_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_frag;
    for (int tile_m = 0; tile_m < BLOCK_M; tile_m += WMMA_M) {
      for (int tile_d = 0; tile_d < D_HEAD; tile_d += WMMA_N) {
        wmma::fill_fragment(pv_frag, 0.0f);
        for (int tile_k = 0; tile_k < BLOCK_N; tile_k += WMMA_K) {
          wmma::load_matrix_sync(p_frag, &store.p_vals[tile_k * BLOCK_M + tile_m], BLOCK_M);
          wmma::load_matrix_sync(v_frag, &store.v_vals[tile_d * BLOCK_N + tile_k], BLOCK_N);
          wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
        }
        wmma::store_matrix_sync(store.pv_wmma, pv_frag, WMMA_N, wmma::mem_row_major);

        // have to use a smaller pv_wmma due to shared mem size constraints
        for (int i = 0; i < 16; i++) {
          for (int j = 0; j < 16; j++) {
            int d_idx = tile_d + j;
            int m_idx = tile_m + i;
            if (d_idx < D_HEAD && m_idx < BLOCK_M) {
              acc[d_idx * BLOCK_M + m_idx] += store.pv_wmma[i * 16 + j];
            }
          }
        }
      }
    }
    //__syncthreads();
  }
  // store results
  for (int mi = tx; mi < BLOCK_M; mi += blockDim.x) {
    if (tile_q0 + mi < N_CTX_Q) {
      for (int d = 0; d < D_HEAD; ++d) {
        Oh[(tile_q0 + mi) * som + d * sod] = fp32_to_bf16(acc[d*BLOCK_M+mi]);
      }
      Lh[tile_q0 + mi] = l_prev[mi];
      Mh[tile_q0 + mi] = m_prev[mi];
    }
  }
}


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
  dim3 block(32); // parallelize block_m/block_n over a thread block
  size_t shmem = sizeof(SharedHashStore);

  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  printf("Max shared memory: %d bytes, requested shared memory: %zu bytes\n", max_sram_size, shmem);

  printf("Q.stride(3) hi: %d\n", Q.stride(3));
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