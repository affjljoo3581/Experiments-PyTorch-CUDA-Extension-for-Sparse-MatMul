#pragma once

#include <sys/types.h>


#define TILE_32x32_WIDTH        32
#define TILE_32x32_SIZE         (TILE_32x32_WIDTH * TILE_32x32_WIDTH)


enum class SparseMode {
    SDD,
    DSD,
    DDS
};


void batched_sparse_smm_op_32x32(
    const   float*          __restrict__    matrix_a,
    const   float*          __restrict__    matrix_b,
            float*          __restrict__    matrix_c,
    const   ushort*         __restrict__    chunk_table,
            uint                            total_chunks,
            uint                            total_batches,
            uint                            total_m,
            uint                            total_n,
            uint                            total_k,
            bool                            trans_a,
            bool                            trans_b,
            SparseMode                      mode
);
