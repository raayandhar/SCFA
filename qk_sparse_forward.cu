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

#include <cuda_bf16.h>

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
            __nv_bfloat16 kv = (gk < N_CTX_KV) ? Kh[gk * skn + col * skd] : __nv_bfloat16(0.0f);
            __nv_bfloat16 vv = (gk < N_CTX_KV) ? Vh[gk * svn + col * svd] : __nv_bfloat16(0.0f);
            store.K[row][col] = kv;
            store.V[row][col] = vv;
        }
        __syncthreads();

        // 3. Proceed its dim across all queries in the tile.
        #pragma unroll
        for (int mi = 0; mi < BLOCK_M; ++mi) {
            int q_global = block_q0 + mi;
            if (q_global >= N_CTX_Q) continue;

            float qd = q_local[mi];
            float dot_partial[BLOCK_N];
            #pragma unroll
            for (int nk = 0; nk < BLOCK_N; ++nk) dot_partial[nk] = qd * bf16_to_fp32(store.K[nk][lane]);

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

// TODO: host launcher.
    

    










