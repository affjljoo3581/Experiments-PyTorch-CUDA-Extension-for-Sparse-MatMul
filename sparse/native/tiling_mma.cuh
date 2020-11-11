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


template <typename T, int ROWS, int COLUMNS>
struct tile {
    constexpr static int BANKS = 32;

    constexpr static int PACKED = 4 / sizeof(T);
    constexpr static int THREADS = ROWS * COLUMNS / PACKED;

    using packed_t = typename packed<T>::type;

    /**
     * Tile storage which stores sub-matrix data.
     * 
     * To increase matrix multiplication performance, matrix data should be
     * stored into shared memory by fetching in sync to reduce memory access
     * latency. `storage` provides a convenient interface to access the
     * sub-matrix data which are in shared memory.
     * 
     * This class is designed to store any type of data with packing them into
     * words (4 bytes) for efficient memory accessing. Futhermore, it supports
     * double-buffering to hide latency by software-pipelining.
     * 
     * Note that `storage` object must be defined in shared memory.
     */
    class storage {
    public:
        constexpr static int PAGES = 2;

        constexpr static int STRIDE = COLUMNS / PACKED;
        constexpr static int SKEWS = ROWS * COLUMNS / BANKS / PACKED;

        constexpr static int SIZE = (ROWS * STRIDE + SKEWS + 32 - 1) / 32 * 32;

        __device__ __forceinline__ packed_t& get(int page, int i, int j) {
            return data[page][i * STRIDE + j / PACKED + (i * STRIDE / BANKS)];
        }
    private:
        packed_t data[PAGES][SIZE];
    };

    /**
     * 
     * 
     */
    class loader {
    public:
        __device__ __forceinline__ loader(
            const T* __restrict__ src, storage &dst, int stride, bool trans
        ) : src(src), dst(dst), stride(stride), trans(trans) {}

        __device__ __forceinline__ void prefetch(int row, int col) {
            int x = threadIdx.x * PACKED % (trans ? ROWS : COLUMNS);
            int y = threadIdx.x * PACKED / (trans ? ROWS : COLUMNS);

            buffer = *(packed_t *) &src[(row + y) * stride + (col + x)];
        }

        __device__ __forceinline__ void commit(int page) {
            int x = threadIdx.x * PACKED % (trans ? ROWS : COLUMNS);
            int y = threadIdx.x * PACKED / (trans ? ROWS : COLUMNS);

            if (std::is_same<T, half>::value && trans) {
                half2 coupled = buffer;
                half2 neighbor = __shfl_xor_sync(
                    0xffffffff, coupled, COLUMNS / 2, warpSize);

                if (y % 2 == 0) buffer = __lows2half2(coupled, neighbor);
                else buffer = __highs2half2(neighbor, coupled);

                x = x / 2 + y % 2;
                y = y / 2 * 2;
            }

            dst.get(page, trans ? x : y, trans ? y : x) = buffer;
        }
    private:
        packed_t buffer;

        const T* __restrict__ src;
        storage &dst;

        int stride;
        bool trans;
    };
};
