#include <mma.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <torch/extension.h>



/**
 * Simple matrix multiplication with shared memory.
 * 
 * Blocks: (Batches, Tiled Rows, Tiled Cols)
 * Threads per block: (32, 32)
 */
__global__ void __launch_bounds__(1024) matmul_simple_shared_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    uint size_m, uint size_n, uint size_k
) {
    __shared__ float shared_a[32][32 + 1];
    __shared__ float shared_b[32][32 + 1];

    uint x = threadIdx.x;
    uint y = threadIdx.y;
    uint offset_row = blockIdx.y * 32;
    uint offset_col = blockIdx.z * 32;

    // Move to current batch matrices.
    matrix_a += blockIdx.x * size_m * size_k;
    matrix_b += blockIdx.x * size_k * size_n;
    matrix_c += blockIdx.x * size_m * size_n;

    float sum = 0.0f;
    for (uint k = 0; k < size_k; k += 32) {
        shared_a[y][x] = matrix_a[(offset_row + y) * size_k + (k + x)];
        shared_b[x][y] = matrix_b[(k + y) * size_n + (offset_col + x)];
        __syncthreads();

        for (uint i = 0; i < 32; i ++)
            sum += shared_a[y][i] * shared_b[x][i];
    }

    matrix_c[(offset_row + y) * size_n + (offset_col + x)] = sum;
}

/**
 * Very optimized matrix multiplication with shared memory and register files.
 * 
 * Blocks: (Batches, Tiled Rows, Tiled Cols)
 * Threads per block: (256)
 */
__global__ __launch_bounds__(256) void matmul_optimized_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    uint size_m, uint size_n, uint size_k
) {
    __shared__ float shared_a[8 * (32 + 1)];
    __shared__ float shared_b[8 * (32 + 1)];

    uint tid = threadIdx.x;
    uint lid = tid % 32;
    uint wid = tid / 32;

    uint offset_row = blockIdx.y * 32;
    uint offset_col = blockIdx.z * 32;

    // Move to current batch matrices.
    matrix_a += blockIdx.x * size_m * size_k;
    matrix_b += blockIdx.x * size_k * size_n;
    matrix_c += blockIdx.x * size_m * size_n;

    float register_a[2], register_b[2], register_c[2][2] = { 0.0f, };

    uint k_iter = 0;
    uint k_loops = (size_k + 7) / 8;

    #pragma unroll 1
    //for (uint k_iter = 0; k_iter < (size_k + 7) / 8; ++k_iter) {
    do {
        // Load tiles from global memory to shared memory.
        //shared_a[(lid % 8) * (32 + 1) + (wid * 4 + lid / 8)] = matrix_a[(offset_row + wid * 4 + lid / 8) * size_k + (k + lid % 8)];
        //shared_b[wid * (32 + 1) + lid] = matrix_b[(k + wid) * size_k + (offset_col + lid)];
        shared_a[(lid % 8) * (32 + 1) + (wid * 4 + lid / 8)] = matrix_a[(offset_row + wid * 4 + lid / 8) * size_k + (k_iter * 32 + lid % 8)];
        shared_b[wid * (32 + 1) + lid] = matrix_b[(k_iter * 32 + wid) * size_k + (offset_col + lid)];

        __syncthreads();

        // Load subtiles from shared memory to register file.
        #pragma unroll
        for (uint i = 0; i < 8; i ++) {
            register_a[0] = shared_a[i * (32 + 1) + wid * 4 + lid / 16 * 2];
            register_a[1] = shared_a[i * (32 + 1) + wid * 4 + lid / 16 * 2 + 1];
            register_b[0] = shared_b[i * (32 + 1) + lid % 16 * 2];
            register_b[1] = shared_b[i * (32 + 1) + lid % 16 * 2 + 1];

            /*
            #pragma unroll
            for (uint x = 0; x < 2; x ++)
                #pragma unroll
                for (uint y = 0; y < 2; y ++)
                    register_c[y][x] += register_a[y] * register_b[x];
            */
            register_c[0][0] += register_a[0] * register_b[0];
            register_c[0][1] += register_a[0] * register_b[1];
            register_c[1][0] += register_a[1] * register_b[0];
            register_c[1][1] += register_a[1] * register_b[1];
        }
    } while (++k_iter < k_loops);

    uint pos_c = (blockIdx.y * 32 + wid * 4 + lid / 16 * 2) * size_n + blockIdx.z * 32 + lid % 16 * 2;
    #pragma unroll
    for (uint x = 0; x < 2; x ++)
        #pragma unroll
        for (uint y = 0; y < 2; y ++)
            matrix_c[pos_c + y * size_n + x] = register_c[y][x];
}

/**
 * Very optimized matrix multiplication V2.
 * 
 * Blocks: (Batches, Tiled Rows, Tiled Cols)
 * Threads per block: (256)
 */
__global__ __launch_bounds__(256) void matmul_optimized_v2_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    uint size_m, uint size_n, uint size_k
) {
    __shared__ float shared_a[8 * (32 + 1)];
    __shared__ float shared_b[8 * (32 + 1)];


    uint tid = threadIdx.x;
    uint lid = tid % 32;
    uint wid = tid / 32;

    uint offset_row = blockIdx.y * 32;
    uint offset_col = blockIdx.z * 32;

    uint k_iter = 0;
    uint k_loops = (size_k + 8 - 1) / 8;

    // Move to current batch matrices.
    uint offset_a = blockIdx.x * size_m * size_k;
    uint offset_b = blockIdx.x * size_k * size_n;
    uint offset_c = blockIdx.x * size_m * size_n;

    float register_a[2], register_b[2], register_c[2][2] = { 0.0f, };

    #pragma unroll 1
    do {
        // Load tiles from global memory to shared memory.
        shared_a[(lid % 8) * (32 + 1) + (wid * 4 + lid / 8)] = matrix_a[offset_a + (offset_row + wid * 4 + lid / 8) * size_k + (k_iter * 32 + lid % 8)];
        shared_b[wid * (32 + 1) + lid] = matrix_b[offset_b + (k_iter * 32 + wid) * size_k + (offset_col + lid)];

        __syncthreads();

        // Load subtiles from shared memory to register file.
        #pragma unroll
        for (uint i = 0; i < 8; i ++) {
            register_a[0] = shared_a[i * (32 + 1) + wid * 4 + lid / 16 * 2];
            register_a[1] = shared_a[i * (32 + 1) + wid * 4 + lid / 16 * 2 + 1];
            register_b[0] = shared_b[i * (32 + 1) + lid % 16 * 2];
            register_b[1] = shared_b[i * (32 + 1) + lid % 16 * 2 + 1];

            #pragma unroll
            for (uint x = 0; x < 2; x ++)
                #pragma unroll
                for (uint y = 0; y < 2; y ++)
                    register_c[y][x] += register_a[y] * register_b[x];
        }
    } while (++k_iter < k_loops);

    uint pos_c = (blockIdx.y * 32 + wid * 4 + lid / 16 * 2) * size_n + blockIdx.z * 32 + lid % 16 * 2;
    #pragma unroll
    for (uint x = 0; x < 2; x ++)
        #pragma unroll
        for (uint y = 0; y < 2; y ++)
            matrix_c[offset_c + pos_c + y * size_n + x] = register_c[y][x];
}

/**
 * Very optimized matrix multiplication V3.
 * 
 * Blocks: (Batches, Tiled Rows, Tiled Cols)
 * Threads per block: (256)
 */
__global__ __launch_bounds__(256) void matmul_optimized_v3_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    uint size_m, uint size_n, uint size_k
) {
__shared__ float shared_buffers[(8 * (32 + 1) + 24) * 4];

    const uint tid = threadIdx.x;
    const uint warp = tid / 32;
    const uint lane = tid % 32;

    const uint m = blockIdx.y * 32;
    const uint n = blockIdx.z * 32;

    // Move to current batch
    const uint offset_a = blockIdx.x * size_m * size_k;
    const uint offset_b = blockIdx.x * size_k * size_n;
    const uint offset_c = blockIdx.x * size_m * size_n;

    float buffer_a, buffer_b;
    float register_a[2][4], register_b[2], register_c[4] = { 0.0f, };


    buffer_a = matrix_a[offset_a + (m + warp * 4 + lane / 8) * size_k + (0 + lane % 8)];
    buffer_b = matrix_b[offset_b + (0 + warp) * size_n + (n + lane)];
    
    shared_buffers[(8 * (32 + 1) + 24) * 0 + warp * (32 + 1) + warp + lane / 8 * 8] = buffer_a;
    shared_buffers[(8 * (32 + 1) + 24) * (2 + 0) + warp * (32 + 1) + lane] = buffer_b;
    __syncthreads();

    #pragma unroll 1
    for (uint k = 0; k < size_k; k += 8) {
        const uint page = k / 8 % 2;

        buffer_a = matrix_a[offset_a + (m + warp * 4 + lane / 8) * size_k + (k + lane % 8 + 8)];
        buffer_b = matrix_b[offset_b + (k + warp + 8) * size_n + (n + lane)];

        #pragma unroll
        for (uint i = 0; i < 8; ++ i) {
            #pragma unroll
            for (uint j = 0; j < 4; ++ j)
                register_a[i % 2][j] = shared_buffers[(8 * (32 + 1) + 24) *page + i * (32 + 1) + warp * 4 + j];
            register_b[i % 2] = shared_buffers[(8 * (32 + 1) + 24) * (2 + page) + i * (32 + 1) + lane];

            #pragma unroll
            for (uint j = 0; j < 4; ++ j)
                register_c[j] = __fmaf_rn(register_a[i % 2][j], register_b[i % 2], register_c[j]);
        }

        shared_buffers[(8 * (32 + 1) + 24) * (1 - page) + warp * (32 + 1) + warp + lane / 8 * 8] = buffer_a;
        shared_buffers[(8 * (32 + 1) + 24) * (2 + 1 - page) + warp * (32 + 1) + lane] = buffer_b;
        
        __syncthreads();
    }

    matrix_c[offset_c + (m + warp * 4 + 0) * size_n + (n + lane)] = register_c[0];
    matrix_c[offset_c + (m + warp * 4 + 1) * size_n + (n + lane)] = register_c[1];
    matrix_c[offset_c + (m + warp * 4 + 2) * size_n + (n + lane)] = register_c[2];
    matrix_c[offset_c + (m + warp * 4 + 3) * size_n + (n + lane)] = register_c[3];
}


/**
 * Very optimized matrix multiplication V3.
 * 
 * Blocks: (Batches, Tiled Rows, Tiled Cols)
 * Threads per block: (256)
 */
__global__ __launch_bounds__(256) void matmul_optimized_v4_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    uint size_m, uint size_n, uint size_k
) {
    __shared__ float shared_buffers[(8 * (32 + 1) + 24) * 4];

    const uint tid = threadIdx.x;
    const uint warp = tid / 32;
    const uint lane = tid % 32;

    const uint m = blockIdx.y * 32;
    const uint n = blockIdx.z * 32;

    // Move to current batch
    const uint offset_a = blockIdx.x * size_m * size_k;
    const uint offset_b = blockIdx.x * size_k * size_n;
    const uint offset_c = blockIdx.x * size_m * size_n;

    float buffer_a, buffer_b;
    float register_a[2][4], register_b[2], register_c[2][4] = { 0.0f, };


    buffer_a = matrix_a[offset_a + (m + warp * 4 + lane / 8) * size_k + (0 + lane % 8)];
    buffer_b = matrix_b[offset_b + (0 + warp) * size_n + (n + lane)];
    
    shared_buffers[(8 * (32 + 1) + 24) * 0 + warp * (32 + 1) + warp + lane / 8 * 8] = buffer_a;
    shared_buffers[(8 * (32 + 1) + 24) * (2 + 0) + warp * (32 + 1) + lane] = buffer_b;
    __syncthreads();

    #pragma unroll 1
    for (uint k = 0; k < size_k; k += 8) {
        const uint page = k / 8 % 2;

        buffer_a = matrix_a[offset_a + (m + warp * 4 + lane / 8) * size_k + (k + lane % 8 + 8)];
        buffer_b = matrix_b[offset_b + (k + warp + 8) * size_n + (n + lane)];

        #pragma unroll
        for (uint i = 0; i < 8; ++ i) {
            #pragma unroll
            for (uint j = 0; j < 4; ++ j)
                register_a[i % 2][j] = shared_buffers[(8 * (32 + 1) + 24) *page + i * (32 + 1) + warp * 4 + j];
            register_b[i % 2] = shared_buffers[(8 * (32 + 1) + 24) * (2 + page) + i * (32 + 1) + lane];

            #pragma unroll
            for (uint j = 0; j < 4; ++ j)
                register_c[i % 2][j] = __fmaf_rn(register_a[i % 2][j], register_b[i % 2], register_c[i % 2][j]);
        }

        shared_buffers[(8 * (32 + 1) + 24) * (1 - page) + warp * (32 + 1) + warp + lane / 8 * 8] = buffer_a;
        shared_buffers[(8 * (32 + 1) + 24) * (2 + 1 - page) + warp * (32 + 1) + lane] = buffer_b;
        
        __syncthreads();
    }

    matrix_c[offset_c + (m + warp * 4 + 0) * size_n + (n + lane)] = register_c[0][0] + register_c[1][0];
    matrix_c[offset_c + (m + warp * 4 + 1) * size_n + (n + lane)] = register_c[0][1] + register_c[1][1];
    matrix_c[offset_c + (m + warp * 4 + 2) * size_n + (n + lane)] = register_c[0][2] + register_c[1][2];
    matrix_c[offset_c + (m + warp * 4 + 3) * size_n + (n + lane)] = register_c[0][3] + register_c[1][3];
}


torch::Tensor matmul_simple_shared(torch::Tensor a, torch::Tensor b) {
    int64_t size_m = a.size(-2);
    int64_t size_n = b.size(-1);
    int64_t size_k = a.size(-1);

    auto c = a.new_empty({ a.size(0), size_m, size_n });

    dim3 blocks(a.size(0), (size_m + 32 - 1) / 32, (size_n + 32 - 1) / 32);
    dim3 threadsPerBlock(32, 32);

    matmul_simple_shared_kernel<<<blocks, threadsPerBlock>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        size_m, size_n, size_k
    );

    return c;
}

torch::Tensor matmul_optimized(torch::Tensor a, torch::Tensor b) {
    int64_t size_m = a.size(-2);
    int64_t size_n = b.size(-1);
    int64_t size_k = a.size(-1);

    auto c = a.new_empty({ a.size(0), size_m, size_n });

    dim3 blocks(a.size(0), (size_m + 32 - 1) / 32, (size_n + 32 - 1) / 32);
    matmul_optimized_kernel<<<blocks, 256>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        size_m, size_n, size_k
    );

    return c;
}

torch::Tensor matmul_optimized_v2(torch::Tensor a, torch::Tensor b) {
    int64_t size_m = a.size(-2);
    int64_t size_n = b.size(-1);
    int64_t size_k = a.size(-1);

    auto c = a.new_empty({ a.size(0), size_m, size_n });

    dim3 blocks(a.size(0), (size_m + 32 - 1) / 32, (size_n + 32 - 1) / 32);
    matmul_optimized_v2_kernel<<<blocks, 256>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        size_m, size_n, size_k
    );

    return c;
}

torch::Tensor matmul_optimized_v3(torch::Tensor a, torch::Tensor b) {
    int64_t size_m = a.size(-2);
    int64_t size_n = b.size(-1);
    int64_t size_k = a.size(-1);

    auto c = a.new_empty({ a.size(0), size_m, size_n });

    dim3 blocks(a.size(0), (size_m + 32 - 1) / 32, (size_n + 32 - 1) / 32);
    matmul_optimized_v3_kernel<<<blocks, 256>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        size_m, size_n, size_k
    );

    return c;
}

torch::Tensor matmul_optimized_v4(torch::Tensor a, torch::Tensor b) {
    int64_t size_m = a.size(-2);
    int64_t size_n = b.size(-1);
    int64_t size_k = a.size(-1);

    auto c = a.new_empty({ a.size(0), size_m, size_n });

    dim3 blocks(a.size(0), (size_m + 32 - 1) / 32, (size_n + 32 - 1) / 32);
    matmul_optimized_v4_kernel<<<blocks, 256>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        size_m, size_n, size_k
    );

    return c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_simple_shared", &matmul_simple_shared);
    m.def("matmul_optimized", &matmul_optimized);
    m.def("matmul_optimized_v2", &matmul_optimized_v2);
    m.def("matmul_optimized_v3", &matmul_optimized_v3);
    m.def("matmul_optimized_v4", &matmul_optimized_v4);
}
