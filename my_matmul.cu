#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <torch/extension.h>


class tile_storage {
public:
    constexpr static uint TILE_ROW = 8;
    constexpr static uint TILE_COLUMN = 32;
    constexpr static uint TILE_SIZE = TILE_ROW * TILE_COLUMN;

    constexpr static uint BANK_SIZE = 32;
    constexpr static uint FLOAT_SKEW = 1;

    constexpr static uint BUFFER_STRIDE = TILE_COLUMN + FLOAT_SKEW;
    constexpr static uint BUFFER_SIZE =
        (TILE_ROW * BUFFER_STRIDE + BANK_SIZE - 1) / BANK_SIZE * BANK_SIZE;

    __device__ __forceinline__ float& access(uint page, uint i, uint j) {
        return buffers[page][i * BUFFER_STRIDE + j];
    }
public:
    float buffers[2][BUFFER_SIZE];
};


class tile_loader {
public:
    __device__ __forceinline__ tile_loader(
        const float* __restrict__ matrix, tile_storage &storage,
        uint stride, bool trans
    ) : matrix(matrix), storage(storage), stride(stride)
    {
        uint x = threadIdx.x % tile_storage::TILE_COLUMN;
        uint y = threadIdx.x / tile_storage::TILE_COLUMN;

        if (trans) {
            from_coord.x = to_coord.y = x % tile_storage::TILE_ROW;
            from_coord.y = to_coord.x = (x / tile_storage::TILE_ROW
                                           * tile_storage::TILE_ROW) + y;
        } else {
            from_coord.x = to_coord.x = x;
            from_coord.y = to_coord.y = y;
        }
    }

    __device__ __forceinline__ void fetch(uint page, uint row, uint col) {
        prefetch(row, col);
        commit(page);
    }

    __device__ __forceinline__ void prefetch(uint row, uint col) {
        buffer = matrix[(row + from_coord.y) * stride + (col + from_coord.x)];
    }

    __device__ __forceinline__ void commit(uint page) {
        storage.access(page, to_coord.y, to_coord.x) = buffer;
    }
private:
    const float*    matrix;
    uint            stride;

    tile_storage&   storage;
    float           buffer;

    uint2           from_coord;
    uint2           to_coord;
};



__global__ void matmul_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    uint size_m, uint size_n, uint size_k
) {
    __shared__ tile_storage tile_a, tile_b;

    uint warp = threadIdx.x / 32;
    uint lane = threadIdx.x % 32;

    uint m = blockIdx.y * 32;
    uint n = blockIdx.z * 32;

    float local_a[4], local_b, local_c[4] = { 0.0f, };

    tile_loader loader_a(matrix_a + blockIdx.x * size_m * size_k,
                         tile_a, size_k, true);
    tile_loader loader_b(matrix_b + blockIdx.x * size_k * size_n,
                         tile_b, size_n, false);

    loader_a.fetch(0, m, 0);
    loader_b.fetch(0, 0, n);
    __syncthreads();

    #pragma unroll 1
    for (uint k = 0; k < size_k; k += tile_storage::TILE_ROW) {
        uint page = k / tile_storage::TILE_ROW % 2;

        if (k + tile_storage::TILE_ROW < size_k) {
            loader_a.prefetch(m, k + tile_storage::TILE_ROW);
            loader_b.prefetch(k + tile_storage::TILE_ROW, n);
        }

        #pragma unroll
        for (uint i = 0; i < tile_storage::TILE_ROW; ++ i) {
            #pragma unroll
            for (uint j = 0; j < 4; ++ j)
                local_a[j] = tile_a.access(page, i, warp * 4 + j);
            local_b = tile_b.access(page, i, lane);

            #pragma unroll
            for (uint j = 0; j < 4; ++ j)
                local_c[j] += local_a[j] * local_b;
        }

        if (k + tile_storage::TILE_ROW < size_k) {
            loader_a.commit(page ? 0 : 1);
            loader_b.commit(page ? 0 : 1);
            __syncthreads();
        }
    }

    #pragma unroll
    for (uint i = 0; i < 4; ++ i)
        matrix_c[blockIdx.x * size_m * size_n
                 + (m + warp * 4 + i) * size_n
                 + (n + lane)] = local_c[i];
}

torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
    int64_t size_m = a.size(-2);
    int64_t size_n = b.size(-1);
    int64_t size_k = a.size(-1);

    auto c = a.new_empty({ a.size(0), size_m, size_n });

    dim3 blocks(a.size(0), (size_m + 32 - 1) / 32, (size_n + 32 - 1) / 32);
    matmul_kernel<<<blocks, 256>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        size_m, size_n, size_k
    );

    return c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul);
}
