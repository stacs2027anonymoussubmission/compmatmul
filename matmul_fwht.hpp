#ifndef COMPMATMUL_MATMUL_FWHT_HPP
#define COMPMATMUL_MATMUL_FWHT_HPP

#include "fwht.hpp"
#include "libarray/array.hpp"
#include "libarray/random.hpp"
#include <omp.h>
#include <cassert>
#include <vector>

namespace compmatmul {
    using array::index_t;
    using array::MemoryLayout;
    using array::Transpose;

#ifdef COMPMATMUL_ENABLE_FWHT

    /**
    * Computes the size of the scratch space needed for the FWHT-based matrix 
    * multiplication.
    * The size depends on the number of threads; providing a nonpositive value 
    * for num_threads will infer the maximum number of threads from the OpenMP 
    * environment.
    * @param d The number of sketches (polynomials) to compute
    * @param b The length of each polynomial (must be a power of 2)
    * @param num_threads The number of threads to use in the computation.
    * @return The size of the scratch space in number of elements
    */
    inline index_t matmul_compress_and_multiply_fwht_scratch_size(int d, 
        int b, int num_threads = 0) {
        assert(b > 0 && (b & (b - 1)) == 0); // b must be a power of 2
        assert(d > 0);
        if (num_threads <= 0) {
            num_threads = omp_get_max_threads();
        }
        // each thread needs d*b elements for the pa polynomials, and one 
        // polynomial of size b for pb
        return static_cast<index_t>(d+1)*static_cast<index_t>(b)*static_cast<index_t>(num_threads);
    }

    /**
     * Compresses a the product of two matrices AB into a polynomial 
     * sketch p using the FWHT and the given hash functions. 
     * @tparam Hash The hash function type
     * @tparam T The type of the matrix elements (double or float)
     * @tparam Layout The memory layout of the matrix (row-major or column-major)
     * @tparam TransposeA Whether to transpose the matrix A before compressing
     * @tparam TransposeB Whether to transpose the matrix B before compressing
     * @param d The number of sketches (polynomials) to compute
     * @param b The length of each polynomial (must be a power of 2)
     * @param p The output polynomials (array of length d*b, must be aligned to 64 bytes)
     * @param n_left The number of rows (or columns if transposed) of the matrix A
     * @param n_inner The number of columns (or rows if transposed) of the matrix A and rows (or columns if transposed) of the matrix B
     * @param n_right The number of columns (or rows if transposed) of the matrix B
     * @param A The input matrix (array of length n_left*n_inner, must be aligned to 64 bytes)
     * @param B The input matrix (array of length n_inner*n_right, must be aligned to 64 bytes)
     * @param ha_params The parameters for the hash functions for matrix A (array of length d*Hash::NUM_PARAMS)
     * @param sa_params The parameters for the sign functions for matrix A (array of length d*Hash::NUM_PARAMS)
     * @param hb_params The parameters for the hash functions for matrix B (array of length d*Hash::NUM_PARAMS)
     * @param sb_params The parameters for the sign functions for matrix B (array of length d*Hash::NUM_PARAMS)
     * @param scratch A scratch space for temporary computations (array of length returned by matmul_compress_and_multiply_fwht_scratch_size(), must be aligned to 64 bytes)
     */
    template<
        typename Hash, 
        typename T, 
        MemoryLayout Layout = MemoryLayout::RowMajor, 
        Transpose TransposeA = Transpose::NoTranspose,
        Transpose TransposeB = Transpose::NoTranspose
    >
    void matmul_compress_and_multiply_fwht(int d, int b, T* __restrict__ p, 
        index_t n_left, index_t n_inner, index_t n_right, 
        const T* __restrict__ A, const T* __restrict__ B,
        const typename Hash::param_t* ha_params, 
        const typename Hash::param_t* sa_params,
        const typename Hash::param_t* hb_params,
        const typename Hash::param_t* sb_params,
        T* scratch) {
        assert(b > 0 && (b & (b - 1)) == 0); // b must be a power of 2
        assert(d > 0);
        assert(reinterpret_cast<uintptr_t>(scratch) % 64 == 0);
        const int blog2 = ilog2(b);

        // each thread needs d*b elements for the pa polynomials, and
        // one polynomial of size b for pb
        const index_t thread_scratch_size = (d+1)*b; 

        FWHTTransformer<T> fwht(b);
        array::zero(d*b, p); // initialize the output polynomials to zero

        std::vector<omp_lock_t> p_locks(d);
        for (int t = 0; t < d; ++t)
            omp_init_lock(&p_locks[t]);

#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            T* thread_scratch = scratch + thread_id*thread_scratch_size;
            // temporary array for the polynomials A (d such polynomials)
            T* pas = thread_scratch; 
            thread_scratch += d*b;
            // temporary array for the polynomial B (just one!)
            T* pb = thread_scratch; 

#pragma omp for schedule(guided)
            for (int k = 0; k < n_inner; ++k) {
                for (int t = 0; t < d; ++t) {
                    T* pa = pas + t*b;
                    array::zero(b, pa);

                    // first construct all pas and transform them
                    for (index_t i = 0; i < n_left; ++i) {
                        index_t idx; // the index of A[i][k]
                        if constexpr (Layout == MemoryLayout::RowMajor) {
                            if constexpr (TransposeA == Transpose::NoTranspose) {
                                idx = i*n_inner + k;
                            } 
                            else {
                                idx = k*n_left + i;
                            }
                        } 
                        else {
                            if constexpr (TransposeA == Transpose::NoTranspose) {
                                idx = k*n_left + i;
                            } 
                            else {
                                idx = i*n_inner + k;
                            }
                        }
                        index_t h = Hash::bucket_hash_pow2(i, 
                            ha_params + t*Hash::NUM_PARAMS, blog2);
                        int sign = Hash::sign_hash(i, 
                            sa_params + t*Hash::NUM_PARAMS);
                        pa[h] += sign * A[idx];
                    }
                    fwht(pa);
                }

                // multiplication and pb construction is interleaved                   
                for (int t = 0; t < d; ++t) {
                    array::zero(b, pb);
                    for (index_t j = 0; j < n_right; ++j) {
                        index_t idx; // the index of B[k][j]
                        if constexpr (Layout == MemoryLayout::RowMajor) {
                            if constexpr (TransposeB == Transpose::NoTranspose) {
                                idx = k*n_right + j;
                            } 
                            else {
                                idx = j*n_inner + k;
                            }
                        } 
                        else {
                            if constexpr (TransposeB == Transpose::NoTranspose) {
                                idx = j*n_inner + k;
                            } 
                            else {
                                idx = k*n_right + j;
                            }
                        }
                        index_t h = Hash::bucket_hash_pow2(j, 
                            hb_params + t*Hash::NUM_PARAMS, blog2);
                        int sign = Hash::sign_hash(j, 
                            sb_params + t*Hash::NUM_PARAMS);
                        pb[h] += sign * B[idx];
                    }
                    fwht(pb);

                    T* pa = pas + t*b;
                    array::mul(b, pb, pa);

                    omp_set_lock(&p_locks[t]);
                    array::add(b, p + t*b, pb);
                    omp_unset_lock(&p_locks[t]);
                }
            }
        }

#pragma omp parallel for
        for (int t = 0; t < d; ++t) {
            fwht(p + t*b);
            array::mul(b, p + t*b, static_cast<T>(1.0)/b); // normalize
        }

        for (int t = 0; t < d; ++t)
            omp_destroy_lock(&p_locks[t]);
    }



    /**
     * Decompress element C[i,j] from the sketch
     * There should be d elements of scratch space
     * @tparam Hash The hash function type
     * @tparam T The type of the matrix elements (double or float)
     * @tparam Layout The memory layout of the matrix (row-major or column-major)
     * @param n The number of columns (or rows if column-major) of the matrix
     * @param C The output matrix (array whose last dimension is n, must be aligned to 64 bytes)
     * @param i The row index of the element to decompress
     * @param j The column index of the element to decompress
     * @param p The sketch containing the compressed product
     * @param d The number of polynomials in the sketch
     * @param blog2 The base-2 logarithm of the length of each polynomial in the sketch
     * @param ha_params The parameters for the hash functions for matrix A (array of length d*Hash::NUM_PARAMS)
     * @param sa_params The parameters for the sign functions for matrix A (array of length d*Hash::NUM_PARAMS)
     * @param hb_params The parameters for the hash functions for matrix B (array of length d*Hash::NUM_PARAMS)
     * @param sb_params The parameters for the sign functions for matrix B (array of length d*Hash::NUM_PARAMS)
     * @param scratch A scratch space for temporary computations (array of length d, must be aligned to 64 bytes)
     */
    template<typename Hash, typename T, MemoryLayout Layout = MemoryLayout::RowMajor>
    void matmul_decompress_fwht(index_t n, T *C, index_t i, index_t j, 
        const T* p, int d, int b, int blog2,
        const typename Hash::param_t* ha_params, 
        const typename Hash::param_t* sa_params,
        const typename Hash::param_t* hb_params, 
        const typename Hash::param_t* sb_params,
        T *scratch) {
        for (int t = 0; t < d; ++t) {
            const typename Hash::param_t* ha = ha_params + t*Hash::NUM_PARAMS;
            const typename Hash::param_t* sa = sa_params + t*Hash::NUM_PARAMS;
            const typename Hash::param_t* hb = hb_params + t*Hash::NUM_PARAMS;
            const typename Hash::param_t* sb = sb_params + t*Hash::NUM_PARAMS;
            typename Hash::hash_t h1 = Hash::bucket_hash_pow2(i, ha, blog2);
            typename Hash::hash_t h2 = Hash::bucket_hash_pow2(j, hb, blog2);
            auto h = h1 ^ h2;
            int s = Hash::sign_hash(i, sa) * Hash::sign_hash(j, sb);
            scratch[t] = s * p[t*b + h];
        }
        index_t idx = (Layout == MemoryLayout::RowMajor) ? i*n + j : j*n + i;
        C[idx] = array::median(d, scratch);
    }



    /**
     * Decompresses the entire matrix C from the sketch
     * There should be max_threads * d elements of scratch space
     * @tparam Hash The hash function type
     * @tparam T The type of the matrix elements (double or float)
     * @tparam Layout The memory layout of the matrix (row-major or column-major)
     * @param n_left The number of rows (or columns if column-major) of the matrix C
     * @param n_right The number of columns (or rows if column-major) of the matrix C
     * @param C The output matrix (array of length n_left*n_right, must be aligned to 64 bytes)
     * @param p The sketch containing the compressed product
     * @param d The number of polynomials in the sketch
     * @param b The length of each polynomial in the sketch (must be a power of 2)
     * @param ha_params The parameters for the hash functions for matrix A (array of length d*Hash::NUM_PARAMS)
     * @param sa_params The parameters for the sign functions for matrix A (array of length d*Hash::NUM_PARAMS)
     * @param hb_params The parameters for the hash functions for matrix B (array of length d*Hash::NUM_PARAMS)
     * @param sb_params The parameters for the sign functions for matrix B (array of length d*Hash::NUM_PARAMS)
     * @param scratch A scratch space for temporary computations (array of length d*max_threads, must be aligned to 64 bytes)
     */
    template<typename Hash, typename T, MemoryLayout Layout = MemoryLayout::RowMajor>
    void matmul_decompress_fwht(index_t n_left, index_t n_right, T *C,
        const T* p, int d, int b,
        const typename Hash::param_t* ha_params, 
        const typename Hash::param_t* sa_params,
        const typename Hash::param_t* hb_params, 
        const typename Hash::param_t* sb_params,
        T *scratch) {
        const int blog2 = ilog2(static_cast<unsigned int>(b));
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            T* thread_scratch = scratch + thread_id*d;
            #pragma omp for schedule(guided)
            for (index_t i = 0; i < n_left; ++i) {
                for (index_t j = 0; j < n_right; ++j) {
                    matmul_decompress_fwht<Hash, T, Layout>(
                        Layout == MemoryLayout::RowMajor ? n_right : n_left, 
                        C, i, j, 
                        p, d, b, blog2,
                        ha_params, sa_params,
                        hb_params, sb_params,
                        thread_scratch
                    );
                }
            }
        }
    }





    /**
     * This is a convenience function that draws the hash functions and 
     * allocates all necessary scratch space for computing the compressed 
     * product, and then computes it.
     * @tparam Hash The hash function type
     * @tparam T The type of the matrix elements (double or float)
     * @tparam Layout The memory layout of the matrix (row-major or column-major)
     * @tparam TransposeA Whether to transpose the matrix A before compressing
     * @tparam TransposeB Whether to transpose the matrix B before compressing
     * @param d The number of sketches (polynomials) to compute
     * @param b The length of each polynomial (must be a power of 2)
     * @param n_left The number of rows (or columns if transposed) of the matrix A
     * @param n_inner The number of columns (or rows if transposed) of the matrix A and rows (or columns if transposed) of the matrix B
     * @param n_right The number of columns (or rows if transposed) of the matrix B
     * @param C The output matrix (array of length n_left*n_right, must be aligned to 64 bytes)
     * @param A The input matrix (array of length n_left*n_inner, must be aligned to 64 bytes)
     * @param B The input matrix (array of length n_inner*n_right, must be aligned to 64 bytes)
     * @param seed An optional seed for the random number generator used to draw the hash functions. If not provided, a random seed will be used.
     */
    template<
        typename Hash, 
        typename T, 
        MemoryLayout Layout = MemoryLayout::RowMajor, 
        Transpose TransposeA = Transpose::NoTranspose,
        Transpose TransposeB = Transpose::NoTranspose
    >
    void matmul_fwht(int d, int b, 
        index_t n_left, index_t n_inner, index_t n_right, 
        T* __restrict__ C, 
        const T* __restrict__ A, const T* __restrict__ B,
        std::optional<uint64_t> seed = std::nullopt) {

        typename Hash::param_t* ha_params = array::allocate<typename Hash::param_t>(d*Hash::NUM_PARAMS);
        typename Hash::param_t* sa_params = array::allocate<typename Hash::param_t>(d*Hash::NUM_PARAMS);
        typename Hash::param_t* hb_params = array::allocate<typename Hash::param_t>(d*Hash::NUM_PARAMS);
        typename Hash::param_t* sb_params = array::allocate<typename Hash::param_t>(d*Hash::NUM_PARAMS);

        array::random::RNG_t rng(seed);
        Hash::draw(d, ha_params, rng.next<uint64_t>());
        Hash::draw(d, sa_params, rng.next<uint64_t>());
        Hash::draw(d, hb_params, rng.next<uint64_t>());
        Hash::draw(d, sb_params, rng.next<uint64_t>());

        T* scratch = array::allocate<T>(
            matmul_compress_and_multiply_fwht_scratch_size(d, b)
        );

        T* p = array::allocate<T>(d*b);

        matmul_compress_and_multiply_fwht<Hash, T, Layout, TransposeA, TransposeB>(
            d, b, p, 
            n_left, n_inner, n_right, 
            A, B,
            ha_params, sa_params,
            hb_params, sb_params,
            scratch
        );

        matmul_decompress_fwht<Hash, T, Layout>(
            n_left, n_right, C,
            p, d, b,
            ha_params, sa_params,
            hb_params, sb_params,
            scratch
        );
        
        array::deallocate(ha_params);
        array::deallocate(sa_params);
        array::deallocate(hb_params);
        array::deallocate(sb_params);
        array::deallocate(scratch);
        array::deallocate(p);
    }


    // TODO:
    // Squared version
    // Include tiling (take a chunk of ks per thread) for better cache utilization
    // Try out the different approaches to construct the polynomials

#endif // COMPMATMUL_ENABLE_FWHT
}

#endif // COMPMATMUL_MATMUL_FWHT_HPP