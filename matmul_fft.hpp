#ifndef COMPMATMUL_MATMUL_FFT_HPP
#define COMPMATMUL_MATMUL_FFT_HPP

#ifdef COMPMATMUL_ENABLE_FFT

#if (defined(COMPMATMUL_USE_FFTW_FFT) + defined(COMPMATMUL_USE_MKL_FFT) + defined(COMPMATMUL_USE_ACCELERATE_FFT)) != 1
#error "You must define exactly one of COMPMATMUL_USE_FFTW_FFT, COMPMATMUL_USE_MKL_FFT, or COMPMATMUL_USE_ACCELERATE_FFT"
#endif

#include "fft.hpp"
#include "libarray/array.hpp"
#include "libarray/random.hpp"

#if defined(COMPMATMUL_USE_FFTW_FFT)
#include <fftw3.h>
#elif defined(COMPMATMUL_USE_MKL_FFT)
#include <mkl.h>
#elif defined(COMPMATMUL_USE_ACCELERATE_FFT)
#include <Accelerate/Accelerate.h>
#else
#error "You must define exactly one of COMPMATMUL_USE_FFTW_FFT, COMPMATMUL_USE_MKL_FFT, or COMPMATMUL_USE_ACCELERATE_FFT"
#endif // defined(COMPMATMUL_USE_FFTW_FFT)

#include <omp.h>

#include <cstdint>
#include <vector>
#include <cassert>
#include <algorithm>

namespace compmatmul {
    using array::index_t;
    using array::MemoryLayout;
    using array::Transpose;
    using array::ilog2;
    using array::complex_t;

    /**
     * This is a helper function that computes the size of the polynomial sketch.
     * The real polynomial must be always at least 64 bytes to enforce alignment.
     * @tparam T The type of the polynomial coefficients (float or double)
     * @param b The length of the polynomial (must be a power of 2)
     * @return The memory size of the polynomial sketch in number of elements of type T
     */
    template<typename T>
    inline index_t real_poly_mem_size(int b) {
        assert(b > 0 && (b & (b - 1)) == 0); // b must be a power of 2
        return std::max<index_t>(64/sizeof(T), b);
    }

    /**
     * This is a helper function that computes the degree of a complex-transformed polynomial
     * It is b/2+1 complex numbers
     * @param b The length of the polynomial (must be a power of 2)
     * @return The degree of the complex-transformed polynomial
     */
    inline index_t fft_complex_poly_size(index_t b) {
        assert(b > 0 && (b & (b - 1)) == 0); // b must be a power of 2
        return b/2 + 1;
    }

    /**
     * This is a helper function that computes the **memory size** of a complex-transformed polynomial.
     * We need to be able to represent b+2 real numbers, but the alignment must be divisible by 64 bytes.
     * @tparam T The type of the polynomial coefficients (float or double)
     * @param b The length of the polynomial (must be a power of 2)
     * @return The memory size of the complex-transformed polynomial in number of elements of type T
     */
    template<typename T>
    inline index_t fft_complex_poly_mem_size(index_t b) {
        assert(b > 0 && (b & (b - 1)) == 0); // b must be a power of 2
        index_t complex_poly_size = fft_complex_poly_size(b);
        index_t mem_size = complex_poly_size * 2; // complex numbers are represented as two real numbers
        // round up to the nearest multiple of 64 bytes
        index_t alignment = 64 / sizeof(T);
        if (mem_size % alignment != 0) {
            mem_size += alignment - (mem_size % alignment);
        }
        return mem_size;
    }

    /**
     * Computes the per-thread scratch space size required for the matmul_compress_and_multiply_fft function.
     * The size of the scratch space depends on the FFT implementation used.
     * @param d The number of sketches (polynomials) to compute
     * @param b The length of each polynomial (must be a power of 2)
     * @return The size of the scratch space in number of elements.
     */
    template<typename T>
    index_t matmul_compress_and_multiply_fft_thread_scratch_size(int d, int b) {
        assert(b > 0 && (b & (b - 1)) == 0); // b must be a power of 2
        assert(d > 0);
        index_t fft_poly_size;
        index_t fft_scratch_size;
#if defined(COMPMATMUL_USE_ACCELERATE_FFT)
        fft_poly_size = b;
        fft_scratch_size = b;
#elif defined(COMPMATMUL_USE_MKL_FFT) || defined(COMPMATMUL_USE_FFTW_FFT)
        // The polynomial requires b/2+1 complex elements, 
        // amounting to b+2 real elements
        // but we must align to 64 bytes
        // fft_poly_size = b+2; 
        fft_poly_size = fft_complex_poly_mem_size<T>(b);
        // MKL and FFTW do not require additional scratch space for the FFT
        fft_scratch_size = 0; 
#else
#error "Not implemented yet for this FFT implementation"
#endif // defined(COMPMATMUL_USE_ACCELERATE_FFT)
        // temporary arrays for pa_fft and pb_fft in frequency domain per thread (2*fft_poly_size)
        // scratch space for untransformed input polynomials (2*d*b, aligned to 64 bytes)
        // FFT-implementation-specific scratch space per thread
        return 2*fft_poly_size + 2*d*real_poly_mem_size<T>(b) + fft_scratch_size;
    }


    /**
     * Computes the size of the required scratch space for the 
     * matmul_compress_and_multiply_fft function. The size of the scratch 
     * space depends on the FFT implementation used.
     * The size also depends on the number of threads; providing a nonpositive 
     * value for num_threads will infer the maximum number of threads from the 
     * OpenMP environment.
     * @param d The number of sketches (polynomials) to compute
     * @param b The length of each polynomial (must be a power of 2)
     * @param num_threads The number of threads to be used in the computation
     * @return The size of the scratch space in number of elements.
     */
    template<typename T>
    index_t matmul_compress_and_multiply_fft_scratch_size(int d, int b, int num_threads = 0) {
        assert(b > 0 && (b & (b - 1)) == 0); // b must be a power of 2
        assert(d > 0);
        if (num_threads <= 0) {
            num_threads = omp_get_max_threads();
        }
        index_t fft_poly_size; // size of the real-transformed polynomial
#if defined(COMPMATMUL_USE_ACCELERATE_FFT)
        fft_poly_size = b; // size of the real-transformed polynomial
#elif defined(COMPMATMUL_USE_MKL_FFT) || defined(COMPMATMUL_USE_FFTW_FFT)
        // The polynomial requires b/2+1 complex elements, 
        // amounting to b+2 real elements
        // but we must align to 64 bytes
        fft_poly_size = fft_complex_poly_mem_size<T>(b);
#else
#error "Not implemented yet for this FFT implementation"
#endif // defined(COMPMATMUL_USE_ACCELERATE_FFT)
        // need to allocate the FFT transformed output polynomials (d*fft_poly_size)
        // plus whatever each thread needs
        return d*fft_poly_size + matmul_compress_and_multiply_fft_thread_scratch_size<T>(d, b) * num_threads;
    }

    /**
     * Compresses a the product of two matrices AB into a polynomial 
     * sketch p using the FFT and the given hash functions. 
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
     * @param scratch A scratch space for temporary computations (array of length returned by matmul_compress_and_multiply_fft_scratch_size(), as it depends on the FFT implementation, must be aligned to 64 bytes)
     */
    template<
        typename Hash, 
        typename T, 
        MemoryLayout Layout = MemoryLayout::RowMajor, 
        Transpose TransposeA = Transpose::NoTranspose,
        Transpose TransposeB = Transpose::NoTranspose
    >
    void matmul_compress_and_multiply_fft(int d, int b, T* __restrict__ p, 
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

#ifdef COMPMATMUL_USE_FFTW_FFT
        // FFTW needs scratch space to construct the plan
        FFTTransformer<T> fft(b, scratch);
#else
        FFTTransformer<T> fft(b);
#endif // COMPMATMUL_USE_FFTW_FFT

        T* p_fft = scratch; // intermediate array for transformed polynomials (d * poly size)
        const index_t real_poly_size = real_poly_mem_size<T>(b);
#if defined(COMPMATMUL_USE_ACCELERATE_FFT)
        const int fft_poly_size = b; // size of the real-transformed polynomial
        using split_complex_t = typename FFTTransformer<T>::split_complex_t;
#elif defined(COMPMATMUL_USE_MKL_FFT) || defined(COMPMATMUL_USE_FFTW_FFT)
        // size of the real-transformed polynomial (in doubles or floats)
        const int fft_poly_size = fft_complex_poly_mem_size<T>(b);
#else
#error "Not implemented yet for this FFT implementation"
#endif // COMPMATMUL_USE_ACCELERATE_FFT
        array::zero(d*fft_poly_size, p_fft);
        scratch += d*fft_poly_size;

        std::vector<omp_lock_t> p_locks(d);
        for (int t = 0; t < d; ++t)
            omp_init_lock(&p_locks[t]);

        const index_t thread_scratch_size = matmul_compress_and_multiply_fft_thread_scratch_size<T>(d, b);
#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            T* thread_scratch = scratch + thread_id*thread_scratch_size; // scratch space for this thread
            T* pa_fft = thread_scratch; // temporary array for transformed polynomial A
            thread_scratch += fft_poly_size; 
            T* pb_fft = thread_scratch; // temporary array for transformed polynomial B
            thread_scratch += fft_poly_size;
            T* pas = thread_scratch; // temporary array for untransformed polynomials A
            thread_scratch += d*real_poly_size;
            T* pbs = thread_scratch; // temporary array for untransformed polynomials B
            thread_scratch += d*real_poly_size;
#if defined(COMPMATMUL_USE_ACCELERATE_FFT)
            T* fft_scratch = thread_scratch; 
            // Accelerate uses split complexes so we need to create a split representation
            split_complex_t pa_fft_split { 
                .realp = pa_fft, 
                .imagp = pa_fft + b/2
            };
            split_complex_t pb_fft_split { 
                .realp = pb_fft, 
                .imagp = pb_fft + b/2
            };
            split_complex_t fft_scratch_split { 
                .realp = fft_scratch, 
                .imagp = fft_scratch + b/2
            };
#elif defined(COMPMATMUL_USE_MKL_FFT) || defined(COMPMATMUL_USE_FFTW_FFT)
            complex_t<T>* pa_fft_complex = reinterpret_cast<complex_t<T>*>(pa_fft);
            complex_t<T>* pb_fft_complex = reinterpret_cast<complex_t<T>*>(pb_fft);
#else
#error "Not implemented yet for this FFT implementation"
#endif // COMPMATMUL_USE_ACCELERATE_FFT

#pragma omp for schedule(guided)
            for(int k = 0; k < n_inner; k++) {
                for (int t = 0; t < d; t++) {
                    T* pa = pas + t*real_poly_size;
                    array::zero(b, pa);

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
#if defined(COMPMATMUL_USE_ACCELERATE_FFT)
                        // accelerate stores real and imaginary parts in 
                        // separate arrays, but for real input, the values are 
                        // packed, so odd go to imaginary array
                        pa[(h & 1) ? b/2 + h/2 : h/2] += sign * A[idx];
#elif defined(COMPMATMUL_USE_MKL_FFT) || defined(COMPMATMUL_USE_FFTW_FFT)
                        // MKL and FFTW use interleaved complex numbers, so we can 
                        // just use the index directly
                        pa[h] += sign * A[idx];
#else
#error "Not implemented yet for this FFT implementation"
#endif // COMPMATMUL_USE_ACCELERATE_FFT
                    }
                }

                for (int t = 0; t < d; t++) {
                    T* pb = pbs + t*real_poly_size;
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
#if defined(COMPMATMUL_USE_ACCELERATE_FFT)
                        // accelerate stores real and imaginary parts in 
                        // separate arrays, but for real input, the values are 
                        // packed, so odd go to imaginary array
                        pb[(h & 1) ? b/2 + h/2 : h/2] += sign * B[idx];
#elif defined(COMPMATMUL_USE_MKL_FFT) || defined(COMPMATMUL_USE_FFTW_FFT)
                        // MKL and FFTW use interleaved complex numbers, so we can 
                        // just use the index directly
                        pb[h] += sign * B[idx];
#else
#error "Not implemented yet for this FFT implementation"
#endif // COMPMATMUL_USE_ACCELERATE_FFT
                    }
                }

                for (int t = 0; t < d; ++t) {
                    T* pa = pas + t*real_poly_size;
                    T* pb = pbs + t*real_poly_size;
#if defined(COMPMATMUL_USE_ACCELERATE_FFT)
                    // Accelerate uses split complexes so we need to create a 
                    // split representation
                    split_complex_t pa_split { 
                        .realp = pa,
                        .imagp = pa + b/2,
                    };
                    split_complex_t pb_split { 
                        .realp = pb,
                        .imagp = pb + b/2 
                    };
                    fft.fwd(&pa_fft_split, &pa_split, &fft_scratch_split);
                    fft.fwd(&pb_fft_split, &pb_split, &fft_scratch_split);
                    // due to packing, the first elements must be handled separately
                    T dc = pa_fft[0] * pb_fft[0];
                    T nf = pa_fft[b/2] * pb_fft[b/2];
                    array::mul(b/2, &pa_fft_split, &pb_fft_split);
                    pa_fft[0] = dc;
                    pa_fft[b/2] = nf;
#elif defined(COMPMATMUL_USE_MKL_FFT) || defined(COMPMATMUL_USE_FFTW_FFT)
                    // MKL/FFTW use interleaved complex numbers, so we can just 
                    // use the arrays directly
                    // MKL/FFTW do not need scratch space
                    fft.fwd(pa_fft_complex, pa);
                    fft.fwd(pb_fft_complex, pb);
                    array::mul(b/2+1, pa_fft_complex, pb_fft_complex);
#else
#error "Not implemented yet for this FFT implementation"
#endif // COMPMATMUL_USE_ACCELERATE_FFT


#if defined(COMPMATMUL_USE_ACCELERATE_FFT)
                    split_complex_t p_fft_split { 
                        .realp = p_fft + t*fft_poly_size, 
                        .imagp = p_fft + t*fft_poly_size + b/2
                    };
#elif defined(COMPMATMUL_USE_MKL_FFT) || defined(COMPMATMUL_USE_FFTW_FFT)
                    complex_t<T>* p_fft_complex = 
                        reinterpret_cast<complex_t<T>*>(
                            p_fft + t*fft_poly_size
                        );
#endif // COMPMATMUL_USE_ACCELERATE_FFT

                    omp_set_lock(&p_locks[t]);
#if defined(COMPMATMUL_USE_ACCELERATE_FFT)
                    // this addition uses split complex numbers
                    array::add(b/2, &p_fft_split, &pa_fft_split);
#elif defined(COMPMATMUL_USE_MKL_FFT) || defined(COMPMATMUL_USE_FFTW_FFT)
                    // this addition uses interleaved complex numbers
                    array::add(b/2+1, p_fft_complex, pa_fft_complex);
#else
#error "Not implemented yet for this FFT implementation"
#endif // COMPMATMUL_USE_ACCELERATE_FFT
                    omp_unset_lock(&p_locks[t]);
                }
            }
        }

        #pragma omp parallel
        {
#ifdef COMPMATMUL_USE_ACCELERATE_FFT
            int thread_id = omp_get_thread_num();
            T* fft_scratch = scratch + thread_id*thread_scratch_size;
            split_complex_t fft_scratch_split { 
                .realp = fft_scratch,
                .imagp = fft_scratch + b/2
            };    
#endif // COMPMATMUL_USE_ACCELERATE_FFT
            #pragma omp for
            for (int t = 0; t < d; ++t) {
#if defined(COMPMATMUL_USE_ACCELERATE_FFT)
                split_complex_t p_fft_split { 
                    .realp = p_fft + t*fft_poly_size, 
                    .imagp = p_fft + t*fft_poly_size + b/2
                };
#elif defined(COMPMATMUL_USE_MKL_FFT) || defined(COMPMATMUL_USE_FFTW_FFT)
                complex_t<T>* p_fft_complex = 
                    reinterpret_cast<complex_t<T>*>(
                        p_fft + t*fft_poly_size
                    );
#endif // COMPMATMUL_USE_ACCELERATE_FFT
                T* pt = p + t*real_poly_size;
#if defined(COMPMATMUL_USE_ACCELERATE_FFT)
                split_complex_t p_split { 
                    .realp = pt,
                    .imagp = pt + b/2
                };
                fft.inv(&p_split, &p_fft_split, &fft_scratch_split);
                // the output must be scaled by 1/4b
                array::mul(b, pt, static_cast<T>(0.25)/b);
#elif defined(COMPMATMUL_USE_MKL_FFT)
                fft.inv(pt, p_fft_complex);
                // for MKL, scaling is handled automatically by the fft object
#elif defined(COMPMATMUL_USE_FFTW_FFT)
                fft.inv(pt, p_fft_complex);
                // for FFTW, we need to scale the output by 1/b
                array::mul(b, pt, static_cast<T>(1)/b);
#else
#error "Not implemented yet for this FFT implementation"
#endif // COMPMATMUL_USE_ACCELERATE_FFT
            }
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
    void matmul_decompress_fft(index_t n, T *C, index_t i, index_t j, 
        const T* p, int d, int b, int blog2,
        const typename Hash::param_t* ha_params, 
        const typename Hash::param_t* sa_params,
        const typename Hash::param_t* hb_params, 
        const typename Hash::param_t* sb_params,
        T *scratch) {
        typename Hash::hash_t mod_mask = ~(~static_cast<Hash::hash_t>(0) << blog2);
        const index_t real_poly_size = real_poly_mem_size<T>(b);
        for (int t = 0; t < d; ++t) {
            const typename Hash::param_t* ha = ha_params + t*Hash::NUM_PARAMS;
            const typename Hash::param_t* sa = sa_params + t*Hash::NUM_PARAMS;
            const typename Hash::param_t* hb = hb_params + t*Hash::NUM_PARAMS;
            const typename Hash::param_t* sb = sb_params + t*Hash::NUM_PARAMS;
            typename Hash::hash_t h1 = Hash::bucket_hash_pow2(i, ha, blog2);
            typename Hash::hash_t h2 = Hash::bucket_hash_pow2(j, hb, blog2);
            auto h = (h1 + h2) & mod_mask;
            int s = Hash::sign_hash(i, sa) * Hash::sign_hash(j, sb);
#if defined(COMPMATMUL_USE_ACCELERATE_FFT)
            // accelerate stores real and imaginary parts in separate arrays,
            // but for real input, the values are packed, so odd indices go to imaginary array
            scratch[t] = s * p[t*real_poly_size + ((h & 1) ? b/2 + h/2 : h/2)];
#elif defined(COMPMATMUL_USE_MKL_FFT) || defined(COMPMATMUL_USE_FFTW_FFT)
            scratch[t] = s * p[t*real_poly_size + h];
#else
#error "Not implemented yet for this FFT implementation"
#endif // COMPMATMUL_USE_ACCELERATE_FFT
        }
        index_t idx = (Layout == MemoryLayout::RowMajor) ? i*n + j : j*n + i;
        C[idx] = array::median(d, scratch);
    }



    /**
     * Computes the scratch space size required for the matmul_decompress_fft function.
     * If the number of threads is nonpositive, the scratch space size is computed for 
     * the maximum number of threads in the OpenMP environment.
     * @param d The number of polynomials in the sketch
     * @param threads The number of threads to be used in the computation
     * @return The size of the scratch space in number of elements.
     */
    index_t matmul_decompress_fft_scratch_size(int d, int threads = 0) {
        if (threads <= 0) {
            threads = omp_get_max_threads();
        }
        return d * threads;
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
    void matmul_decompress_fft(index_t n_left, index_t n_right, T *C,
        const T* p, int d, int b,
        const typename Hash::param_t* ha_params, 
        const typename Hash::param_t* sa_params,
        const typename Hash::param_t* hb_params, 
        const typename Hash::param_t* sb_params,
        T *scratch) {
        const int blog2 = ilog2(b);
#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            T* thread_scratch = scratch + thread_id*d;
#pragma omp for schedule(guided)
            for (index_t i = 0; i < n_left; ++i) {
                for (index_t j = 0; j < n_right; ++j) {
                    matmul_decompress_fft<Hash, T, Layout>(
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
    void matmul_fft(int d, int b, 
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
            matmul_compress_and_multiply_fft_scratch_size<T>(d, b)
        );

        T* p = array::allocate<T>(d*real_poly_mem_size<T>(b));

        matmul_compress_and_multiply_fft<Hash, T, Layout, TransposeA, TransposeB>(
            d, b, p, 
            n_left, n_inner, n_right, 
            A, B,
            ha_params, sa_params,
            hb_params, sb_params,
            scratch
        );

        matmul_decompress_fft<Hash, T, Layout>(
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
}

#endif // COMPMATMUL_ENABLE_FFT

#endif // COMPMATMUL_MATMUL_FFT_HPP
