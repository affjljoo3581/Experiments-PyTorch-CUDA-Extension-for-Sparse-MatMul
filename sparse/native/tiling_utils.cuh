#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>


/** An empty type-wrapper class to separate routines by corresponding type. */
template <typename T> struct caseof {};


template <typename T, uint ROWS, uint COLUMNS>
struct tile {
    constexpr static uint WARPS = 32;

    constexpr static uint PACKED = 4 / sizeof(T);
    constexpr static uint THREADS = ROWS * COLUMNS / PACKED;

    /**
     * Tile storage class to store a part of matrix in shared memory.
     * 
     * To increase matrix multiplication performance, the matrix data should be
     * stored into shared memory to reduce memory access latency. Futhermore, it
     * provides double-buffering to hide the latency by software-pipelining.
     */
    class storage {
    public:
        constexpr static uint PAGES = 2;

        constexpr static uint UNIT = WARPS * PACKED;
        constexpr static uint STRIDE = COLUMNS + PACKED;

        constexpr static uint SIZE = (ROWS * STRIDE + UNIT - 1) / UNIT * UNIT;

        __device__ __forceinline__ T& get(uint page, uint i, uint j) {
            return data[page][i * STRIDE + j];
        }
    private:
        T data[PAGES][SIZE];
    };

    /**
     * Tile loader class to load a part of matrix to a tile storage in sync.
     * 
     * This class provides a memory access interface according to the given
     * access options (e.g. stride size and transposition). Especially it
     * supports half-precision transposition by shuffling registers in
     * warp-level for efficient tile loading.
     */
    class loader {
    public:
        __device__ __forceinline__ loader(const T* __restrict__ src,
                                          storage &dst,
                                          uint stride,
                                          bool trans)
            : src(src), dst(dst), stride(stride), trans(trans)
        {
            x = threadIdx.x * PACKED % (trans ? ROWS : COLUMNS);
            y = threadIdx.x * PACKED / (trans ? ROWS : COLUMNS);
        }

        __device__ __forceinline__ void prefetch(uint row, uint col) {
            *(uint *) &buffer = *(uint *) &src[(row + y) * stride + (col + x)];
        }

        __device__ __forceinline__ void commit(uint page) {
            commit(page, caseof<T>());
        }

        __device__ __forceinline__ void commit(uint page, caseof<float>) {
            *(float *) &dst.get(page, trans ? x : y, trans ? y : x)
                = *(float *) &buffer;
        }

        __device__ __forceinline__ void commit(uint page, caseof<half>) {
            half2 coupled = *(half2 *) &buffer;

            if (trans) {
                // Get another coupled `half2` variable from neighbor row.
                half2 neighbor = __shfl_xor_sync(
                    0xffffffff, coupled, warpSize / PACKED, warpSize
                );

                // Mix the original coupled variable and neighbor's one to
                // create new transposed `half2` vector.
                if (y % 2 == 0) coupled = __lows2half2(coupled, neighbor);
                else coupled = __highs2half2(neighbor, coupled);
            }

            *(half2 *) &dst.get(page, trans ? x : y, trans ? y : x) = coupled;
        }
    private:
        const T* __restrict__ src;
        storage& dst;

        uint stride, x, y;
        bool trans;

        T buffer[PACKED];
    };


    class accumulator {
    public:
        __device__ __forceinline__ accumulator(storage &src_a, storage &src_b)
            : src_a(src_a), src_b(src_b)
        {
            x = threadIdx.x * PACKED % WARPS;
            y = threadIdx.x * PACKED / WARPS * (ROWS / COLUMNS);
        }

        __device__ __forceinline__ void apply(
            T* __restrict__ dst, uint m, uint n, uint stride
        ) {
            #pragma unroll
            for (uint i = 0; i < PACKED; ++ i) {
                #pragma unroll
                for (uint j = 0; j < ROWS / COLUMNS; j += PACKED) {
                    *(uint *) &dst[(m + x) * stride + (n + y + j)]
                        = *(uint *) &data[i][j];
                }
            }
        }

        __device__ __forceinline__ void product(uint page) {
            product(page, caseof<T>());
        }

        __device__ __forceinline__ void product(uint page, caseof<float>) {
            #pragma unroll
            for (uint i = 0; i < COLUMNS; ++ i) {
                float local_a, local_b[ROWS / COLUMNS];

                #pragma unroll
                for (uint j = 0; j < ROWS / COLUMNS; ++ j)
                    local_b[j] = src_b.get(page, y + j, i);
                local_a = src_a.get(page, x, i);

                #pragma unroll
                for (uint j = 0; j < ROWS / COLUMNS; ++ j)
                    data[0][j] += local_a * local_b[j];
            }
        }

        __device__ __forceinline__ void product(uint page, caseof<half>) {
            
        }
    private:
        storage &src_a, &src_b;
        uint x, y;

        T data[PACKED][ROWS / COLUMNS] = { 0.0f, };
    };
};
