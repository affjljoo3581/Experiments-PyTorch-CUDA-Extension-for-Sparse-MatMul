#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>


/** An empty type-wrapper class to separate routines by corresponding type. */
template <typename T> struct caseof {};


template <typename T> struct packed {};
template <> struct packed<float> { using type = float; };
template <> struct packed<half> { using type = half2; };


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
        __device__ __forceinline__ loader(bool trans)
            : trans(trans)
        {
            x = threadIdx.x * PACKED % (trans ? ROWS : COLUMNS);
            y = threadIdx.x * PACKED / (trans ? ROWS : COLUMNS);
        }

        __device__ __forceinline__ void prefetch(const T* __restrict__ src, uint row, uint col, uint stride) {
            *(uint *) &buffer = *(uint *) &src[(row + y) * stride + (col + x)];
        }

        __device__ __forceinline__ void commit(storage &dst, uint page) {
            commit(dst, page, caseof<T>());
        }

        __device__ __forceinline__ void commit(storage &dst, uint page, caseof<float>) {
            *(float *) &dst.get(page, trans ? x : y, trans ? y : x)
                = *(float *) &buffer;
        }

        __device__ __forceinline__ void commit(storage &dst, uint page, caseof<half>) {
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
        uint x, y;
        bool trans;

        T buffer[PACKED];
    };

    /**
     * Accumulator class for tile matrix multiplication.
     * 
     * After loading tiles to shared memory, vector-products would be computed
     * with the loaded tiles. This class fetches subvectors from the storages to
     * the local register files and compute a part of vector-products.
     */
    class accumulator {
    public:
        __device__ __forceinline__ accumulator()
        {
            x = threadIdx.x * PACKED % WARPS;
            y = threadIdx.x * PACKED / WARPS * (ROWS / COLUMNS);
        }

        __device__ __forceinline__ void apply(
            T* __restrict__ dst, uint m, uint n, uint stride
        ) {
            #pragma unroll
            for (uint i = 0; i < ROWS / COLUMNS; i += PACKED)
                *(uint *) &dst[(m + x) * stride + (n + y + i)]
                    = *(uint *) &data[i];
        }

        __device__ __forceinline__ void product(storage &src_a, storage &src_b, uint page) {
            product(src_a, src_b, page, caseof<T>());
        }

        __device__ __forceinline__ void product(storage &src_a, storage &src_b, uint page, caseof<float>) {
            #pragma unroll
            for (uint i = 0; i < COLUMNS; ++ i) {
                float local_a, local_b[ROWS / COLUMNS];

                #pragma unroll
                for (uint j = 0; j < ROWS / COLUMNS; ++ j)
                    local_b[j] = src_b.get(page, y + j, i);
                local_a = src_a.get(page, x, i);

                #pragma unroll
                for (uint j = 0; j < ROWS / COLUMNS; ++ j)
                    data[j] += local_a * local_b[j];
            }
        }

        __device__ __forceinline__ void product(storage &src_a, storage &src_b, uint page, caseof<half>) {
            half2 local_c[ROWS / COLUMNS];

            #pragma unroll
            for (uint i = 0; i < COLUMNS; i += 2) {
                half2 local_a, local_b[ROWS / COLUMNS];

                #pragma unroll
                for (uint j = 0; j < ROWS / COLUMNS; ++ j)
                    local_b[j] = *(half2 *) &src_b.get(page, y + j, i);
                local_a = *(half2 *) &src_a.get(page, x, i);

                #pragma unroll
                for (uint j = 0; j < ROWS / COLUMNS; ++ j)
                    local_c[j] += local_a * local_b[j];
            }

            #pragma unroll
            for (uint i = 0; i < ROWS / COLUMNS; ++ i)
                data[i] = __low2half(local_c[i]) + __high2half(local_c[i]);
        }
    private:
        uint x, y;

        T data[ROWS / COLUMNS] = { 0, };
    };
};
