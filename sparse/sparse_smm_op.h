#pragma once

#include <sys/types.h>


#define TILE_32x32_WIDTH        32
#define TILE_32x32_SIZE         (TILE_32x32_WIDTH * TILE_32x32_WIDTH)


void batched_sparse_smm_op_32x32_sdd(
    const   float*      matrix_a,
    const   float*      matrix_b,
            float*      matrix_c,
    const   short*      sparse_table,
            uint        total_blocks,
            uint        total_batches,
            uint        total_m,
            uint        total_n,
            uint        total_k,
            bool        trans_a,
            bool        trans_b
);

void batched_sparse_smm_op_32x32_dsd(
    const   float*      matrix_a,
    const   float*      matrix_b,
            float*      matrix_c,
    const   short*      sparse_table,
    const   int*        sparse_table_ptr,
            uint        total_blocks,
            uint        total_batches,
            uint        total_m,
            uint        total_n,
            uint        total_k,
            bool        trans_a,
            bool        trans_b
);

void batched_sparse_smm_op_32x32_dds(
    const   float*      matrix_a,
    const   float*      matrix_b,
            float*      matrix_c,
    const   short*      sparse_table,
    const   int*        sparse_table_ptr,
            uint        total_blocks,
            uint        total_batches,
            uint        total_m,
            uint        total_n,
            uint        total_k,
            bool        trans_a,
            bool        trans_b
);
