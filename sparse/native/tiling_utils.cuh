#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <type_traits>


template <typename T> struct packed {};
template <> struct packed<float> { using type = float; };
template <> struct packed<half> { using type = half2; };


template <typename T, uint ROWS, uint COLUMNS>
struct tile {
    constexpr static uint WARPS = 32;

    constexpr static uint PACKED = 4 / sizeof(T);
    constexpr static uint THREADS = ROWS * COLUMNS / PACKED;


    using packed_t = typename packed<T>::type;

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
        __device__ __forceinline__ loader(bool trans) : trans(trans) {}

        __device__ __forceinline__ void prefetch(
            const T* __restrict__ src, uint stride,
            uint row, uint col
        ) {
            uint x = threadIdx.x * PACKED % (trans ? ROWS : COLUMNS);
            uint y = threadIdx.x * PACKED / (trans ? ROWS : COLUMNS);

            *(uint *) &buffer = *(uint *) &src[(row + y) * stride + (col + x)];
        }

        __device__ __forceinline__ void commit(storage &dst, uint page) {
            uint x = threadIdx.x * PACKED % (trans ? ROWS : COLUMNS);
            uint y = threadIdx.x * PACKED / (trans ? ROWS : COLUMNS);

            if (std::is_same<T, half>::value && trans) {
                // Get another coupled `half2` variable from neighbor row.
                half2 neighbor = __shfl_xor_sync(
                    0xffffffff, buffer, warpSize / PACKED, warpSize
                );

                // Mix the original coupled variable and neighbor's one to
                // create new transposed `half2` vector.
                if (y % 2 == 0) buffer = __lows2half2(coupled, neighbor);
                else buffer = __highs2half2(neighbor, coupled);
            }

            *(packed_t *) &dst.get(page, trans ? x : y, trans ? y : x) = buffer;
        }
    private:
        packed_t buffer;

        bool trans;
    };
};
