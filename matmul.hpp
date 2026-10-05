#ifndef COMPMATMUL_MATMUL_HPP
#define COMPMATMUL_MATMUL_HPP

#include "matmul_fwht.hpp"
#include "matmul_fft.hpp"
#include "hashing.hpp"
#include "libarray/array.hpp"

/**
 * @file matmul.hpp
 * This file contains generic helper functions and definitions for 
 * matrix multiplication.
 */

namespace compmatmul {
    using array::index_t;
    using array::MemoryLayout;
    using array::Transpose;

    /**
     * Enumeration of the available matrix multiplication algorithms.
     * The algorithms are:
     * - GEMM: Standard matrix multiplication (GEMM)
     * - FFT: Matrix multiplication using FFT-based polynomial sketching
     * - FWHT: Matrix multiplication using FWHT-based polynomial sketching
     * The availability of the FFT and FWHT algorithms depends on how the
     * library was compiled.
     */
    enum class Algorithm {
        GEMM, ///< Standard matrix multiplication (GEMM)
#ifdef COMPMATMUL_ENABLE_FFT
        FFT, ///< Matrix multiplication using FFT-based polynomial sketching
#endif // COMPMATMUL_ENABLE_FFT
#ifdef COMPMATMUL_ENABLE_FWHT
        FWHT ///< Matrix multiplication using FWHT-based polynomial sketching
#endif // COMPMATMUL_ENABLE_FWHT
    };



    /**
     * Enumeration of the data types supported by the matrix multiplication 
     * functions. Supported data types are:
     * - FLOAT32: 32-bit floating point (float)
     * - FLOAT64: 64-bit floating point (double)
     */
    enum class DType {
        Float32, ///< 32-bit floating point
        Float64 ///< 64-bit floating point
    };



    /**
     * Enumeration of the hash functions supported by the matrix multiplication 
     * functions. Supported hash functions are:
     * - MultiplyAddShift: Multiply-add-shift hash function
     */
    enum class HashFunction {
        MultiplyAddShift ///< Multiply-add-shift hash function
    };



    template<typename Hash, typename T, MemoryLayout Layout, Transpose TransposeA, Transpose TransposeB>
    inline void matmul_h_dt_ml_ta_tb(Algorithm algorithm, index_t n_left, index_t n_inner, index_t n_right,
        T* __restrict__ C, const T* __restrict__ A, const T* __restrict__ B,
        std::optional<int> d,
        std::optional<int> b,
        std::optional<uint64_t> seed) {
        (void) seed; // suppress unused variable warning

        if (algorithm == Algorithm::GEMM) {
            array::gemm<T, Layout, TransposeA, TransposeB>(n_left, n_inner, n_right, C, A, B);
        }
        else {
            if (!d.has_value() || !b.has_value()) {
                throw std::invalid_argument("Parameters d and b must be provided for FFT and FWHT algorithms");
            }

#ifdef COMPMATMUL_ENABLE_FFT
            if (algorithm == Algorithm::FFT) {
                matmul_fft<
                    Hash, T, Layout, TransposeA, TransposeB
                    >(d.value(), b.value(), n_left, n_inner, n_right,
                    C, A, B, seed);
            }
            else 
#endif // COMPMATMUL_ENABLE_FFT
#ifdef COMPMATMUL_ENABLE_FWHT
            if (algorithm == Algorithm::FWHT) {
                matmul_fwht<
                    Hash, T, Layout, TransposeA, TransposeB
                    >(d.value(), b.value(), n_left, n_inner, n_right,
                    C, A, B, seed);
            }
            else
#endif // COMPMATMUL_ENABLE_FWHT  
            {
                throw std::invalid_argument("Unsupported algorithm");
            }
        }
    }

    template<typename Hash, typename T, MemoryLayout Layout, Transpose TransposeA>
    inline void matmul_h_dt_ml_ta(Algorithm algorithm, index_t n_left, index_t n_inner, index_t n_right,
        T* __restrict__ C, const T* __restrict__ A, const T* __restrict__ B,
        std::optional<int> d,
        std::optional<int> b,
        std::optional<uint64_t> seed,
        Transpose transpose_B) {
        if (transpose_B == Transpose::NoTranspose) {
            matmul_h_dt_ml_ta_tb<Hash, T, Layout, TransposeA, Transpose::NoTranspose>(algorithm, n_left, n_inner, n_right,
                C, A, B, d, b, seed);
        } 
        else if (transpose_B == Transpose::Transpose) {
            matmul_h_dt_ml_ta_tb<Hash, T, Layout, TransposeA, Transpose::Transpose>(algorithm, n_left, n_inner, n_right,
                C, A, B, d, b, seed);
        }
        else {
            throw std::invalid_argument("Unsupported transpose option for matrix B");
        }
    }

    template<typename Hash, typename T, MemoryLayout Layout>
    inline void matmul_h_dt_ml(Algorithm algorithm, index_t n_left, index_t n_inner, index_t n_right,
        T* __restrict__ C, const T* __restrict__ A, const T* __restrict__ B,
        std::optional<int> d,
        std::optional<int> b,
        std::optional<uint64_t> seed,
        Transpose transpose_A, Transpose transpose_B) {
        if (transpose_A == Transpose::NoTranspose) {
            matmul_h_dt_ml_ta<Hash, T, Layout, Transpose::NoTranspose>(algorithm, n_left, n_inner, n_right,
                C, A, B, d, b, seed, transpose_B);
        } 
        else if (transpose_A == Transpose::Transpose) {
            matmul_h_dt_ml_ta<Hash, T, Layout, Transpose::Transpose>(algorithm, n_left, n_inner, n_right,
                C, A, B, d, b, seed, transpose_B);
        }
        else {
            throw std::invalid_argument("Unsupported transpose option for matrix A");
        }
    }

    template<typename Hash, typename T>
    inline void matmul_h_dt(Algorithm algorithm, index_t n_left, index_t n_inner, index_t n_right,
        T* __restrict__ C, const T* __restrict__ A, const T* __restrict__ B,
        std::optional<int> d,
        std::optional<int> b,
        std::optional<uint64_t> seed,
        MemoryLayout layout, Transpose transpose_A, Transpose transpose_B) {
        if (layout == MemoryLayout::RowMajor) {
            matmul_h_dt_ml<Hash, T, MemoryLayout::RowMajor>(algorithm, n_left, n_inner, n_right,
                C, A, B, d, b, seed, transpose_A, transpose_B);
        } 
        else if (layout == MemoryLayout::ColumnMajor) {
            matmul_h_dt_ml<Hash, T, MemoryLayout::ColumnMajor>(algorithm, n_left, n_inner, n_right,
                C, A, B, d, b, seed, transpose_A, transpose_B);
        } 
        else {
            throw std::invalid_argument("Unsupported memory layout");
        }
    }



    template<typename Hash>
    inline void matmul_h(Algorithm algorithm, DType dtype,
        index_t n_left, index_t n_inner, index_t n_right,
        void* __restrict__ C, const void* __restrict__ A, const void* __restrict__ B,
        std::optional<int> d = std::nullopt, 
        std::optional<int> b = std::nullopt,
        std::optional<uint64_t> seed = std::nullopt,
        MemoryLayout layout = MemoryLayout::RowMajor,
        Transpose transpose_A = Transpose::NoTranspose,
        Transpose transpose_B = Transpose::NoTranspose) {
        if (dtype == DType::Float32) {
            matmul_h_dt<Hash,float>(algorithm, n_left, n_inner, n_right,
                static_cast<float*>(C), static_cast<const float*>(A), static_cast<const float*>(B),
                d, b, seed, layout, transpose_A, transpose_B);
        } 
        else if (dtype == DType::Float64) {
            matmul_h_dt<Hash,double>(algorithm, n_left, n_inner, n_right,
                static_cast<double*>(C), static_cast<const double*>(A), static_cast<const double*>(B),
                d, b, seed, layout, transpose_A, transpose_B);
        } 
        else {
            throw std::invalid_argument("Unsupported data type");
        }
    }



    /**
     * This is a convenience function that performs matrix multiplication using 
     * the specified algorithm and data type, allowing the caller to avoid
     * manually instantiating the templates or managing resources.
     * @param algorithm The matrix multiplication algorithm to use (GEMM, FFT, or FWHT)
     * @param dtype The data type of the matrices (Float32 or Float64)
     * @param n_left The number of rows of the matrices A (after transposition) and C
     * @param n_inner The number of columns of the matrix A (after transposition) and rows of the matrix B
     * @param n_right The number of columns of the matrices B (after transposition) and C
     * @param C The output matrix (array of length n_left*n_right)
     * @param A The input matrix A (array of length n_left*n_inner)
     * @param B The input matrix B (array of length n_inner*n_right)
     * @param d The number of sketches (polynomials) to compute (optional, only used for FFT and FWHT)
     * @param b The length of each polynomial (must be a power of 2, optional, only used for FFT and FWHT)
     * @param seed The seed for the random number generator (optional, only used for FFT and FWHT)
     * @param layout The memory layout (row-major or column-major)
     * @param transpose_A Whether to transpose the matrix A before multiplication
     * @param transpose_B Whether to transpose the matrix B before multiplication
     */
    inline void matmul(Algorithm algorithm, HashFunction hash,
        DType dtype,
        index_t n_left, index_t n_inner, index_t n_right,
        void* __restrict__ C, const void* __restrict__ A, const void* __restrict__ B,
        std::optional<int> d = std::nullopt, 
        std::optional<int> b = std::nullopt,
        std::optional<uint64_t> seed = std::nullopt,
        MemoryLayout layout = MemoryLayout::RowMajor,
        Transpose transpose_A = Transpose::NoTranspose,
        Transpose transpose_B = Transpose::NoTranspose) {
        if (hash == HashFunction::MultiplyAddShift) {
            matmul_h<MultiplyAddShiftHash>(algorithm, dtype, n_left, n_inner, n_right,
                static_cast<float*>(C), static_cast<const float*>(A), static_cast<const float*>(B),
                d, b, seed, layout, transpose_A, transpose_B);
        } 
        else {
            throw std::invalid_argument("Unsupported hash function");
        }
    }



    template<typename Hash, typename T, MemoryLayout Layout, Transpose TransposeA, Transpose TransposeB>
    inline void matmul_static_h_dt_ml_ta_tb(Algorithm algorithm, 
        index_t n_left, index_t n_inner, index_t n_right,
        T* __restrict__ C, const T* __restrict__ A, 
        const T* __restrict__ B,
        T* __restrict__ p,
        T* __restrict__ scratch,
        typename Hash::param_t* __restrict__ ha_params,
        typename Hash::param_t* __restrict__ sa_params,
        typename Hash::param_t* __restrict__ hb_params,
        typename Hash::param_t* __restrict__ sb_params,
        std::optional<int> d,
        std::optional<int> b) {

        // suppress unused variable warnings for parameters that are not used in GEMM
        (void) p;
        (void) scratch;
        (void) ha_params;
        (void) sa_params;
        (void) hb_params;
        (void) sb_params;

        if (algorithm == Algorithm::GEMM) {
            array::gemm<T, Layout, TransposeA, TransposeB>(n_left, n_inner, n_right, C, A, B);
        }
        else {
            if (!d.has_value() || !b.has_value()) {
                throw std::invalid_argument("Parameters d and b must be provided for FFT and FWHT algorithms");
            }

#ifdef COMPMATMUL_ENABLE_FFT
            if (algorithm == Algorithm::FFT) {
                matmul_compress_and_multiply_fft<Hash, T, Layout, TransposeA, TransposeB>(
                    d.value(), b.value(), p, 
                    n_left, n_inner, n_right,
                    A, B,
                    ha_params, sa_params,
                    hb_params, sb_params,
                    scratch
                );
                matmul_decompress_fft<Hash, T, Layout>(
                    n_left, n_right, C,
                    p, d.value(), b.value(),
                    ha_params, sa_params,
                    hb_params, sb_params,
                    scratch
                );
            }
            else 
#endif // COMPMATMUL_ENABLE_FFT
#ifdef COMPMATMUL_ENABLE_FWHT
            if (algorithm == Algorithm::FWHT) {
                matmul_compress_and_multiply_fwht<Hash, T, Layout, TransposeA, TransposeB>(
                    d.value(), b.value(), p, 
                    n_left, n_inner, n_right,
                    A, B,
                    ha_params, sa_params,
                    hb_params, sb_params,
                    scratch
                );
                matmul_decompress_fwht<Hash, T, Layout>(
                    n_left, n_right, C,
                    p, d.value(), b.value(),
                    ha_params, sa_params,
                    hb_params, sb_params,
                    scratch
                );
            }
            else 
#endif // COMPMATMUL_ENABLE_FWHT
            {
                throw std::invalid_argument("Unsupported algorithm");
            }
        }
    }

    template<typename Hash, typename T, MemoryLayout Layout, Transpose TransposeA>
    inline void matmul_static_h_dt_ml_ta(Algorithm algorithm, 
        index_t n_left, index_t n_inner, index_t n_right,
        T* __restrict__ C, const T* __restrict__ A, 
        const T* __restrict__ B,
        T* __restrict__ p,
        T* __restrict__ scratch,
        typename Hash::param_t* __restrict__ ha_params,
        typename Hash::param_t* __restrict__ sa_params,
        typename Hash::param_t* __restrict__ hb_params,
        typename Hash::param_t* __restrict__ sb_params,
        std::optional<int> d,
        std::optional<int> b,
        Transpose transpose_B) {
        if (transpose_B == Transpose::NoTranspose) {
            matmul_static_h_dt_ml_ta_tb<Hash, T, Layout, TransposeA, Transpose::NoTranspose>(
                algorithm, 
                n_left, n_inner, n_right,
                C,A,B, p,
                scratch, ha_params, sa_params, hb_params, sb_params,
                d, b);
        } 
        else if (transpose_B == Transpose::Transpose) {
            matmul_static_h_dt_ml_ta_tb<Hash, T, Layout, TransposeA, Transpose::Transpose>(
                algorithm, n_left, n_inner, n_right,
                C, A, B, p,
                scratch, ha_params, sa_params, hb_params, sb_params,
                d, b);
        }
        else {
            throw std::invalid_argument("Unsupported transpose option for matrix B");
        }
    }

    template<typename Hash, typename T, MemoryLayout Layout>
    inline void matmul_static_h_dt_ml(Algorithm algorithm, 
        index_t n_left, index_t n_inner, index_t n_right,
        T* __restrict__ C, const T* __restrict__ A, 
        const T* __restrict__ B,
        T* __restrict__ p,
        T* __restrict__ scratch,
        typename Hash::param_t* __restrict__ ha_params,
        typename Hash::param_t* __restrict__ sa_params,
        typename Hash::param_t* __restrict__ hb_params,
        typename Hash::param_t* __restrict__ sb_params,
        std::optional<int> d,
        std::optional<int> b,
        Transpose transpose_A,
        Transpose transpose_B) {
        if (transpose_A == Transpose::NoTranspose) {
            matmul_static_h_dt_ml_ta<Hash, T, Layout, Transpose::NoTranspose>(algorithm, n_left, n_inner, n_right,
                C, A, B, p, scratch, ha_params, sa_params, hb_params, sb_params,
                d, b, transpose_B);
        } 
        else if (transpose_A == Transpose::Transpose) {
            matmul_static_h_dt_ml_ta<Hash, T, Layout, Transpose::Transpose>(algorithm, n_left, n_inner, n_right,
                C, A, B, p, scratch, ha_params, sa_params, hb_params, sb_params,
                d, b, transpose_B);
        }
        else {
            throw std::invalid_argument("Unsupported transpose option for matrix A");
        }
    }

    template<typename Hash, typename T>
    inline void matmul_static_h_dt(Algorithm algorithm, 
        index_t n_left, index_t n_inner, index_t n_right,
        T* __restrict__ C, const T* __restrict__ A, 
        const T* __restrict__ B,
        T* __restrict__ p,
        T* __restrict__ scratch,
        typename Hash::param_t* __restrict__ ha_params,
        typename Hash::param_t* __restrict__ sa_params,
        typename Hash::param_t* __restrict__ hb_params,
        typename Hash::param_t* __restrict__ sb_params,
        std::optional<int> d,
        std::optional<int> b,
        MemoryLayout layout,
        Transpose transpose_A,
        Transpose transpose_B) {
        if (layout == MemoryLayout::RowMajor) {
            matmul_static_h_dt_ml<Hash, T, MemoryLayout::RowMajor>(algorithm, n_left, n_inner, n_right,
                C, A, B, p, scratch, ha_params, sa_params, hb_params, sb_params,
                d, b, transpose_A, transpose_B);
        } 
        else if (layout == MemoryLayout::ColumnMajor) {
            matmul_static_h_dt_ml<Hash, T, MemoryLayout::ColumnMajor>(algorithm, n_left, n_inner, n_right,
                C, A, B, p, scratch, ha_params, sa_params, hb_params, sb_params,
                d, b, transpose_A, transpose_B);
        } 
        else {
            throw std::invalid_argument("Unsupported memory layout");
        }
    }

    template<typename Hash>
    inline void matmul_static_h(Algorithm algorithm, 
        DType dtype,
        index_t n_left, index_t n_inner, index_t n_right,
        void* __restrict__ C, const void* __restrict__ A, 
        const void* __restrict__ B,
        void* __restrict__ p,
        void* __restrict__ scratch,
        typename Hash::param_t* __restrict__ ha_params,
        typename Hash::param_t* __restrict__ sa_params,
        typename Hash::param_t* __restrict__ hb_params,
        typename Hash::param_t* __restrict__ sb_params,
        std::optional<int> d,
        std::optional<int> b,
        MemoryLayout layout,
        Transpose transpose_A,
        Transpose transpose_B) {
        if (dtype == DType::Float32) {
            matmul_static_h_dt<Hash,float>(algorithm, n_left, n_inner, n_right,
                reinterpret_cast<float*>(C), 
                reinterpret_cast<const float*>(A), 
                reinterpret_cast<const float*>(B), 
                reinterpret_cast<float*>(p),
                reinterpret_cast<float*>(scratch), 
                ha_params, sa_params, hb_params, sb_params,
                d, b, layout, transpose_A, transpose_B);
        } 
        else if (dtype == DType::Float64) {
            matmul_static_h_dt<Hash,double>(algorithm, n_left, n_inner, n_right,
                reinterpret_cast<double*>(C), 
                reinterpret_cast<const double*>(A), 
                reinterpret_cast<const double*>(B), 
                reinterpret_cast<double*>(p),
                reinterpret_cast<double*>(scratch), 
                ha_params, sa_params, hb_params, sb_params,
                d, b, layout, transpose_A, transpose_B);
        } 
        else {
            throw std::invalid_argument("Unsupported data type");
        }
    }



    /**
     * A convenience function for calling matrix multiplication with 
     * preallocated resources.
     */
    inline void matmul_static(Algorithm algorithm, HashFunction hash, 
        DType dtype,
        index_t n_left, index_t n_inner, index_t n_right,
        void* __restrict__ C, const void* __restrict__ A, 
        const void* __restrict__ B,
        void* __restrict__ p = nullptr,
        void* __restrict__ scratch = nullptr,
        void* __restrict__ ha_params = nullptr,
        void* __restrict__ sa_params = nullptr,
        void* __restrict__ hb_params = nullptr,
        void* __restrict__ sb_params = nullptr,
        std::optional<int> d = std::nullopt,
        std::optional<int> b = std::nullopt,
        MemoryLayout layout = MemoryLayout::RowMajor,
        Transpose transpose_A = Transpose::NoTranspose,
        Transpose transpose_B = Transpose::NoTranspose) {
        if (hash == HashFunction::MultiplyAddShift) {
            matmul_static_h<MultiplyAddShiftHash>(algorithm, dtype, n_left, n_inner, n_right,
                C, A, B, p, scratch, 
                reinterpret_cast<typename MultiplyAddShiftHash::param_t*>(ha_params),
                reinterpret_cast<typename MultiplyAddShiftHash::param_t*>(sa_params),
                reinterpret_cast<typename MultiplyAddShiftHash::param_t*>(hb_params),
                reinterpret_cast<typename MultiplyAddShiftHash::param_t*>(sb_params),
                d, b, layout, transpose_A, transpose_B);
        } 
        else {
            throw std::invalid_argument("Unsupported hash function");
        }
    }
}

#endif // COMPMATMUL_MATMUL_HPP