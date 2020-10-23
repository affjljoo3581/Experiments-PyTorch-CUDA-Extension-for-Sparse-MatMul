#pragma once

#include <sys/types.h>


void batched_sparse_matmul_op_32x32_sdd(
    const   float*      matrix_a,
    const   float*      matrix_b,
            float*      matrix_c,
    const   short*      sparse_blocks,
            uint        total_blocks,
            uint        total_batches,
            uint        total_m,
            uint        total_n,
            uint        total_k,
            bool        trans_a,
            bool        trans_b
);

void batched_sparse_matmul_op_32x32_dsd(
    const   float*      matrix_a,
    const   float*      matrix_b,
            float*      matrix_c,
    const   short*      sparse_blocks,
    const   int*        sparse_table,
            uint        total_blocks,
            uint        total_batches,
            uint        total_m,
            uint        total_n,
            uint        total_k,
            bool        trans_a,
            bool        trans_b
);

void batched_sparse_matmul_op_32x32_dds(
    const   float*      matrix_a,
    const   float*      matrix_b,
            float*      matrix_c,
    const   short*      sparse_blocks,
    const   int*        sparse_table,
            uint        total_blocks,
            uint        total_batches,
            uint        total_m,
            uint        total_n,
            uint        total_k,
            bool        trans_a,
            bool        trans_b
);
