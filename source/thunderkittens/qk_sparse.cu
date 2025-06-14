#include <torch/extension.h>
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kittens.cuh"

using namespace kittens;

constexpr int NUM_WORKERS = 4;
constexpr int PIPE_STAGES = 3; 
constexpr int PAD_KEY = 1'000'000'000;
constexpr float NEG_INF = base_types::constants<float>::neg_infty();

template<int D> constexpr size_t ROWS = 16*(128/D);
template<int D, typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, ROWS<D>, D, L>;
template<int D, typename T=float> using attn_tile = rt<T, ROWS<D>, ROWS<D>>;
template<int D> using shared_tile = st_bf<ROWS<D>, D>;
template<int D> using global_layout = gl<bf16, -1, -1, -1, D>;

template<int D> using idx_layout = gl<int32_t, -1, 1, -1, kittens::TILE_COL_DIM<int32_t>>;
template<int D> using idx_tile = st<int32_t, ROWS<D>, kittens::TILE_COL_DIM<int32_t>>; 

template<int D> struct globals { 
    global_layout<D> Qg, Kg, Vg, Og; 
    idx_layout<D> Q_idx_g, K_idx_g;
    int H_per_batch;
};

template<int D> __launch_bounds__(NUM_WORKERS*WARP_THREADS, 1)
__global__ void attend_ker(const __grid_constant__ globals<D> g) {
    
    using load_group = kittens::group<2>;
    int loadid = load_group::groupid(), workerid = kittens::warpid();
    constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS;
    const int batch = blockIdx.z, head = blockIdx.y, q_seq = blockIdx.x * NUM_WORKERS + workerid;

    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);
    
    shared_tile<D> (&k_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    shared_tile<D> (&v_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    idx_tile<D> (&q_idx_smem)[NUM_WORKERS] = al.allocate<idx_tile<D>, NUM_WORKERS>();
    idx_tile<D> (&k_idx_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<idx_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    
    shared_tile<D> (&qo_smem)[NUM_WORKERS] = reinterpret_cast<shared_tile<D>(&)[NUM_WORKERS]>(k_smem);
    
    qkvo_tile<D, bf16> q_reg, k_reg;
    qkvo_tile<D, bf16, col_l> v_reg;
    qkvo_tile<D, float> o_reg;
    attn_tile<D, float> att_block;
    attn_tile<D, bf16> att_block_mma;
    typename attn_tile<D, float>::col_vec max_vec_last, max_vec, norm_vec;

    if (q_seq*ROWS<D> < g.Qg.depth()) {
        load<1, false>(qo_smem[workerid], g.Qg, {batch, q_seq, head, 0});
        __syncwarp();
        load(q_reg, qo_smem[workerid]);
        load(q_idx_smem[workerid], g.Q_idx_g, {batch * g.H_per_batch + head, 0, q_seq * ROWS<D>, 0});
    }
    __syncthreads();

    constexpr int IDX_COLS = kittens::TILE_COL_DIM<int32_t>;  // 8
    const int32_t* q_idx_ptr = reinterpret_cast<const int32_t*>(&q_idx_smem[workerid]);
    int q_orig[ROWS<D>];
    #pragma unroll
    for (int r = 0; r < ROWS<D>; ++r)
        q_orig[r] = q_idx_ptr[r * IDX_COLS];

    if constexpr(D == 64) q_reg *= __float2bfloat16(0.125f * 1.44269504089f);
    else if constexpr(D == 128) q_reg *= __float2bfloat16(0.08838834764f * 1.44269504089f);

    max_vec = base_types::constants<float>::neg_infty();
    norm_vec = 0.f;
    o_reg = 0.f;

    int kv_blocks = (g.Kg.depth() + LOAD_BLOCKS*ROWS<D>-1) / (LOAD_BLOCKS*ROWS<D>), tic = 0;
    load_group::load_async<1, false>(k_smem[loadid][0], g.Kg, {batch, loadid, head, 0});
    load_group::load_async<1, false>(v_smem[loadid][0], g.Vg, {batch, loadid, head, 0});
    load_group::load_async<1,false>(k_idx_smem[loadid][0], g.K_idx_g, {batch * g.H_per_batch + head, 0, loadid * ROWS<D>, 0});

    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic=(tic+1)%3) {
        int next_load_idx = (kv_idx+1)*LOAD_BLOCKS + loadid;
        if(next_load_idx*ROWS<D> < g.Kg.depth()) {
            int next_tic = (tic+1)%3;
            load_group::load_async<1, false>(k_smem[loadid][next_tic], g.Kg, {batch, next_load_idx, head, 0});
            load_group::load_async<1, false>(v_smem[loadid][next_tic], g.Vg, {batch, next_load_idx, head, 0});
            load_group::load_async<1,false>(k_idx_smem[loadid][next_tic], g.K_idx_g, {batch * g.H_per_batch + head, 0, next_load_idx * ROWS<D>, 0});
            load_async_wait<1>(); // next k, v can stay in flight.
        }
        else load_async_wait();
        __syncthreads();

        #pragma unroll LOAD_BLOCKS
        for(int subtile = 0; subtile < LOAD_BLOCKS && (kv_idx*LOAD_BLOCKS + subtile)*ROWS<D> < g.Kg.depth(); subtile++) {
            load(k_reg, k_smem[subtile][tic]);
            att_block = 0.f;

            mma<transpose::N, transpose::T>(att_block, q_reg, k_reg, att_block); // Q@K.T
            int first_index = (kv_idx*LOAD_BLOCKS + subtile)*ROWS<D>; // one past the last KV index of this tile
            int start_fill = g.Kg.depth()-first_index < ROWS<D> ? g.Kg.depth()-first_index : ROWS<D>;
            if (start_fill == 0) break;
            right_fill(att_block, att_block, start_fill, base_types::constants<float>::neg_infty());

            const int32_t* k_idx_ptr = reinterpret_cast<const int32_t*>(&k_idx_smem[subtile][tic]);
            int k_idx_vec[ROWS<D>];
            #pragma unroll
            for (int r = 0; r < ROWS<D>; ++r)
                k_idx_vec[r] = k_idx_ptr[r * IDX_COLS];

            constexpr int TILE_W = ROWS<D>; // 16 or 32
            float* ab_ptr = reinterpret_cast<float*>(&att_block);

            #pragma unroll
            for (int row = 0; row < TILE_W; ++row)
                if (q_orig[row] == -1)
                    for (int col = 0; col < TILE_W; ++col)
                        ab_ptr[row*TILE_W + col] = NEG_INF;

            #pragma unroll
            for (int col = 0; col < TILE_W; ++col)
                if (k_idx_vec[col] == PAD_KEY)
                    for (int row = 0; row < TILE_W; ++row)
                        ab_ptr[row*TILE_W + col] = NEG_INF;

            max_vec_last = max_vec;
            max_vec = max<axis::COL>(att_block, max_vec);
            att_block = exp2(att_block - max_vec);

            max_vec_last = exp2(max_vec_last - max_vec);
            norm_vec *= max_vec_last;
            norm_vec = sum<axis::COL>(att_block, norm_vec);

            att_block_mma = att_block;

            load(v_reg, v_smem[subtile][tic]);
            o_reg *= max_vec_last; 
            mma<transpose::N, transpose::N>(o_reg, att_block_mma, v_reg, o_reg);
        }
    }

    o_reg /= norm_vec;
    __syncthreads();
    if (q_seq*ROWS<D> < g.Og.depth()) {
        store(qo_smem[workerid], o_reg);
        __syncwarp();
        store<1, false>(g.Og, qo_smem[workerid], {batch, q_seq, head, 0});
    }
}