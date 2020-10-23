#pragma once

#include <sys/types.h>


void sparse_softmax_op_32x32_forward(
    const   float*      matrix_x,
            float*      matrix_y,
    const   short*      sparse_blocks,
    const   int*        sparse_table,
            uint        total_blocks,
            uint        total_batches,
            uint        total_rows
);
