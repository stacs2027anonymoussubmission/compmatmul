#ifndef LIBARRAY_ARRAY_HPP
#define LIBARRAY_ARRAY_HPP

#include <omp.h>

/**
 * @file array.hpp
 * The main array library.
 */

#if (defined(LIBARRAY_BLAS_GENERIC) + defined(LIBARRAY_BLAS_OPENBLAS) + \
    defined(LIBARRAY_BLAS_MKL) + defined(LIBARRAY_BLAS_AOCL) + \
    defined(LIBARRAY_BLAS_ACCELERATE) != 1)
 #error "Exactly one BLAS backend must be defined"
 #endif


#if defined(LIBARRAY_BLAS_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(LIBARRAY_BLAS_MKL)
#include <mkl.h>
#endif


#ifdef LIBARRAY_DEBUG
#include <cxxabi.h>
#endif // defined(LIBARRAY_DEBUG)
#include <cstdint>
#include <cstdlib>
#include <source_location>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <bit>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>


/**
 * @namespace array
 * This namespace contains the major components of the array library, 
 * including array types, random number generation, and other utilities.
 */
namespace array {
    /**
     * Type for array indices and sizes.
     */
    using index_t = int64_t;

    template<typename T>
    struct ComplexType;

    /**
     * Floating-point complex number type that is defined depending on 
     * the BLAS backend. The data is laid out in memory as consisting two 
     * floats, the first being the real part and the second being the 
     * imaginary part.
     */
    template <typename T>
    using complex_t = ComplexType<T>::type;

#ifdef LIBARRAY_BLAS_ACCELERATE
    template<typename T>
    struct SplitComplexType;

    /**
     * Floating-point complex type that stores the real and imaginary parts 
     * split in two arrays. This is only available when using the Accelerate 
     * framework as the BLAS backend.
     */
    template<typename T>
    using split_complex_t = SplitComplexType<T>::type;
#endif // defined(LIBARRAY_BLAS_ACCELERATE)


#ifdef LIBARRAY_BLAS_MKL
    template<>
    struct ComplexType<float> {
        using type = MKL_Complex8;
    };

    template<>
    struct ComplexType<double> {
        using type = MKL_Complex16;
    };
#endif

#ifdef LIBARRAY_BLAS_ACCELERATE
    template<>
    struct ComplexType<float> {
        using type = DSPComplex;
    };

    template<>
    struct ComplexType<double> {
        using type = DSPDoubleComplex;
    };

    template<>
    struct SplitComplexType<float> {
        using type = DSPSplitComplex;
    };

    template<>
    struct SplitComplexType<double> {
        using type = DSPDoubleSplitComplex;
    };
#endif

#ifdef LIBARRAY_DEBUG
    /**
     * An auxiliary struct to hold information about an allocation, including the pointer, 
     * size, and source location.
     */
    struct Allocation {
        Allocation(void* p, index_t e, size_t s, std::source_location loc) :
            ptr(p), elem_count(e), size_bytes(s), location(loc) {
        }

        virtual ~Allocation() {

        }

        virtual std::string type_name() const = 0;
        
        void* ptr;
        index_t elem_count;
        size_t size_bytes;
        std::source_location location;
    };


    /**
     * Type specific variant of Allocation that holds the pointer, size, and source location 
     * for a specific type T.
     */
    template<typename T>
    struct TypedAllocation: public Allocation {
        TypedAllocation(T* p, index_t e, size_t s, std::source_location loc) :
            Allocation(static_cast<void*>(p), e, s, loc) {
        }

        std::string type_name() const override {
            int status = 0;
            std::unique_ptr<char, void(*)(void*)> res {
                abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status),
                std::free
            };
            return status ? typeid(T).name() : res.get();
        }
    };


    // hack to ensure that allocations outlives UnfreedKiller
    using allocations_t = std::unordered_map<void*, 
        std::unique_ptr<Allocation>>;
    inline allocations_t* allocations = new allocations_t();    

    struct UnfreedKiller {
        ~UnfreedKiller() {
            if (!allocations->empty()) {
                std::cerr << "WARNING: There are " << allocations->size() 
                        << " unfreed allocations at program exit." << std::endl;
                for (const auto& [ptr, alloc] : *allocations) {
                    std::cerr << "Unfreed allocation of size " << alloc->size_bytes 
                            << " bytes (" << alloc->elem_count 
                            << " elements of type " 
                            << alloc->type_name() << "), allocated at " 
                            << alloc->location.file_name() 
                            << ":" << alloc->location.line() << std::endl;
                }
            }
            delete allocations;
            allocations = nullptr;
        }
    };
    
    inline UnfreedKiller unfreed_killer;
#endif // defined(LIBARRAY_DEBUG)



    /**
     * Allocates a new array (aligned at 64-byte boundary) of the given size and type T.
     * @param size The number of elements in the array.
     * @return A pointer to the newly allocated array or nullptr if the allocation failed.
     */
    template<typename T>
    inline T* allocate(index_t size, std::source_location location = std::source_location::current()) {
        (void)location; // Suppress unused variable warning if DEBUG is not defined
        // must allocate at least 64 bytes to ensure 64-byte alignment
#ifdef LIBARRAY_DEBUG
        if (size < 0) {
            return nullptr;
        }
#endif // defined(LIBARRAY_DEBUG)
        size_t s = sizeof(T) * size;
        if (s % 64 != 0) {
            s += 64 - (s % 64);
        }
        T* ptr = static_cast<T*>(aligned_alloc(64, s));
#ifdef LIBARRAY_DEBUG
        if (ptr) {
            allocations->insert({ptr, 
                std::make_unique<TypedAllocation<T>>(ptr, size, s, location)});
        }
#endif // defined(LIBARRAY_DEBUG)
        return ptr;
    }



    /**
     * Deallocates an array that was previously allocated with allocate().
     * @param ptr Pointer to the array to deallocate.
     */
    template<typename T>
    inline void deallocate(T* ptr, std::source_location location = std::source_location::current()) {
        (void)location; // Suppress unused variable warning if DEBUG is not defined
#ifdef LIBARRAY_DEBUG
        if (ptr != nullptr) {
            if (allocations->find(ptr) == allocations->end()) {
                std::cerr << "WARNING: Attempting to deallocate a pointer "
                    << "that was not allocated by allocate() at " 
                    << location.file_name() << ":" << location.line() 
                    << " or a double free detected." << std::endl;
            } 
            else {
                allocations->erase(ptr);
            }            
        }
#endif // defined(LIBARRAY_DEBUG)
        free(ptr);
    }



    /**
     * Specifies the memroy layout for 2-dimensional arrays.
     */
    enum class MemoryLayout {
        RowMajor,
        ColumnMajor
    };

    /**
     * Specifies whether the array should be transposed or not.
     */
    enum class Transpose {
        NoTranspose,
        Transpose
    };



    /**
     * Returns true if and only if the two arrays A and B of type T and size 'size' are equal.
     * @param size The number of elements in the arrays.
     * @param A Pointer to the first array.
     * @param B Pointer to the second array.
     * @return True if the arrays are equal, false otherwise.
     */
    template<typename T>
    inline bool equal(index_t size, const T* __restrict__ A, 
        const T* __restrict__ B) {
        int equal_result = 1;
#pragma omp parallel for simd if (!omp_in_parallel()) schedule(static) reduction(&:equal_result)
        for (index_t i = 0; i < size; ++i) {
            int eq;
            // we have a specialization for complex types
            if constexpr (std::is_same_v<T, complex_t<float>> || 
                std::is_same_v<T, complex_t<double>>) {
                    eq = (A[i].real == B[i].real && A[i].imag == B[i].imag);
            }
#ifdef LIBARRAY_BLAS_ACCELERATE
            else if constexpr (std::is_same_v<T, split_complex_t<float>> || 
                std::is_same_v<T, split_complex_t<double>>) {
                eq = (A->realp[i] == B->realp[i] && 
                    A->imagp[i] == B->imagp[i]);
            }
#endif // defined(LIBARRAY_BLAS_ACCELERATE)
            else {
                // fallback option
                eq = (A[i] == B[i]);
            }
            equal_result &= eq;
        }
        return equal_result;
    }



    /**
    * Returns true if and only if all elements in the array are equal to the 
    * value v.
    * TODO: implement a more efficient version (requires benchmarking)
    * @tparam T The type of the elements in the array.
    * @param size The number of elements in the array.
    * @param A Pointer to the array.
    * @param v The value to compare against.
    * @return True if all elements are equal to v, false otherwise.
     */
    template<typename T>
    inline bool equal(index_t size, const T* __restrict__ A, const T v) {
        int equal_result = 1;
#pragma omp parallel for simd if (!omp_in_parallel()) schedule(static) reduction(&:equal_result)
        for (index_t i = 0; i < size; ++i) {
            equal_result &= (A[i] == v);
        }
        return equal_result;
    }



    /**
     * Returns true if and only if the two arrays A and B of type T and size 
     * 'size' are almost equal within a tolerance.
     * @param size The number of elements in the arrays.
     * @param A Pointer to the first array.
     * @param B Pointer to the second array.
     * @param tol The tolerance within which the arrays are considered equal.
     * @return True if the arrays are almost equal, false otherwise.
     */
    template<typename T>
    inline bool almost_equal(index_t size, const T* __restrict__ A, 
        const T* __restrict__ B, T tol) {
        int equal_result = 1;
#pragma omp parallel for simd if (!omp_in_parallel()) schedule(static) reduction(&:equal_result)
        for (index_t i = 0; i < size; ++i) {
            equal_result &= (std::abs(A[i] - B[i]) <= tol);
        }
        return equal_result;
    }


    /**
     * Computes the absolute difference between two complex numbers a and b.
     * That is, |a - b| = sqrt((re(a) - re(b))^2 + (im(a) - im(b))^2).
     * @tparam T The type of the elements in the complex numbers.
     * @param a The first complex number.
     * @param b The second complex number.
     * @return The absolute difference between a and b.
     */
    template<typename T>
    inline T abs_diff(const complex_t<T>& a, const complex_t<T>& b) {
        T diff_real = a.real - b.real;
        T diff_imag = a.imag - b.imag;
        return std::sqrt(diff_real * diff_real + diff_imag * diff_imag);
    }


    /**
     * Returns true if and only if all elements in the complex array are 
     * almost equal to one of the values v within an absolute tolerance,
     * with respect to the modulus of the difference between the two complex 
     * numbers.
     * @tparam T The type of the elements in the array.
     * @param size The number of elements in the array.
     * @param A Pointer to the array.
     * @param B Pointer to the array.
     * @param tol The tolerance within which the arrays are considered equal.
     * @return True if all elements are almost equal to v, false otherwise.
     */
    template<typename T>
    inline bool almost_equal(index_t size, const complex_t<T>* __restrict__ A,
        const complex_t<T>* __restrict__ B, T tol) {
        int equal_result = 1;
#pragma omp parallel for simd if (!omp_in_parallel()) schedule(static) reduction(&:equal_result)
        for (index_t i = 0; i < size; ++i) {
            equal_result &= abs_diff<T>(A[i], B[i]) <= tol;
        }
        return equal_result;
    }



#ifdef LIBARRAY_BLAS_ACCELERATE
    /**
     * Returns true if and only if all elements in the complex array are 
     * almost equal to one of the values v within an absolute tolerance,
     * with respect to the modulus of the difference between the two complex 
     * numbers.
     * @tparam T The type of the elements in the array.
     * @param size The number of elements in the array.
     * @param A Pointer to the array.
     * @param B Pointer to the array.
     * @param tol The tolerance within which the arrays are considered equal.
     * @return True if all elements are almost equal to v, false otherwise.
     */
    template<typename T>
    inline bool almost_equal(index_t size, 
        const split_complex_t<T>* __restrict__ A,
        const split_complex_t<T>* __restrict__ B, T tol) {
        int equal_result = 1;
#pragma omp parallel for simd if (!omp_in_parallel()) schedule(static) reduction(&:equal_result)
        for (index_t i = 0; i < size; ++i) {
            equal_result &= abs_diff<T>(complex_t<T> { A->realp[i], A->imagp[i] }, 
                complex_t<T>{ B->realp[i], B->imagp[i] }) <= tol;
        }
        return equal_result;
    }
#endif // defined(LIBARRAY_BLAS_ACCELERATE)



    /**
     * Performs an out-of-place transpose of a matrix A of type T with 
     * dimensions (n_rows x n_cols) and stores the result in matrix B.
     * @tparam T The type of the elements in the matrices.
     * @param n_rows The number of rows in matrix A.
     * @param n_cols The number of columns in matrix A.
     * @param A Pointer to the input matrix A.
     * @param B Pointer to the output matrix B.
     */
    template<typename T, MemoryLayout Layout = MemoryLayout::RowMajor>
    inline void transpose(index_t n_rows, index_t n_cols, const T* __restrict__ A, 
        T* __restrict__ B) {
        static_assert(std::is_same<T, double>::value || 
            std::is_same<T, float>::value, 
            "transpose is only implemented for double and float.");
        if constexpr (std::is_same<T,double>::value) {
#if defined(LIBARRAY_BLAS_ACCELERATE)
            vDSP_mtransD(A, 1, B, 1, 
                Layout == MemoryLayout::RowMajor ? n_cols : n_rows, 
                Layout == MemoryLayout::RowMajor ? n_rows : n_cols);
#elif defined(LIBARRAY_BLAS_MKL)
            mkl_domatcopy(
                Layout == MemoryLayout::RowMajor ? 'R' : 'C', 
                'T', n_rows, n_cols, 1.0, A,
                Layout == MemoryLayout::RowMajor ? n_cols : n_rows,
                B,
                Layout == MemoryLayout::RowMajor ? n_rows : n_cols
            );
#else
#error "Not implemented!"
#endif // defined(LIBARRAY_BLAS_ACCELERATE)
        }
        else {
#if defined(LIBARRAY_BLAS_ACCELERATE)
            vDSP_mtrans(A, 1, B, 1, 
                Layout == MemoryLayout::RowMajor ? n_cols : n_rows, 
                Layout == MemoryLayout::RowMajor ? n_rows : n_cols);
#elif defined(LIBARRAY_BLAS_MKL)
            mkl_somatcopy(
                Layout == MemoryLayout::RowMajor ? 'R' : 'C', 
                'T', n_rows, n_cols, 1.0, A,
                Layout == MemoryLayout::RowMajor ? n_cols : n_rows,
                B,
                Layout == MemoryLayout::RowMajor ? n_rows : n_cols
            );
#else
#error "Not implemented!"
#endif // defined(LIBARRAY_BLAS_ACCELERATE)
        }
    }



    /** 
     * Computes the matrix product of two matrices A and B, storing the result 
     * in C.
     * The Matrix A shall have shape (n_left x n_inner), matrix B shall 
     * have shape (n_inner x n_right), and the result matrix C shall have 
     * shape (n_left x n_right).
     * @tparam T The type of the elements in the matrices.
     * @tparam Layout The memory layout of the result matrix C (RowMajor or ColumnMajor).
     * @tparam TransposeA The transpose operation for the left-hand matrix A (NoTranspose or Transpose).
     * @tparam TransposeB The transpose operation for the right-hand matrix B (NoTranspose or Transpose).
     * @param n_left The number of rows in matrix A.
     * @param n_inner The number of columns in matrix A and rows in matrix B.
     * @param n_right The number of columns in matrix B.
     * @param C Pointer to the result matrix.
     * @param A Pointer to the first matrix.
     * @param B Pointer to the second matrix.
     */
    template<typename T, MemoryLayout Layout = MemoryLayout::RowMajor, 
            Transpose TransposeA = Transpose::NoTranspose,
            Transpose TransposeB = Transpose::NoTranspose>
    void gemm(index_t n_left, index_t n_inner, index_t n_right, 
        T* __restrict__ C, const T* __restrict__ A, const T* __restrict__ B) {
        static_assert(std::is_same<T, double>::value || 
            std::is_same<T, float>::value, 
            "gemm is only implemented for double and float.");
        if constexpr (std::is_same<T,double>::value) {
            cblas_dgemm(
                Layout == MemoryLayout::RowMajor ? CblasRowMajor : CblasColMajor, 
                TransposeA == Transpose::NoTranspose ? CblasNoTrans : CblasTrans,
                TransposeB == Transpose::NoTranspose ? CblasNoTrans : CblasTrans,
                n_left, n_right, n_inner, 1.0, 
                A, 
                Layout == MemoryLayout::RowMajor ? (TransposeA == Transpose::NoTranspose ? n_inner : n_left) :
                (TransposeA == Transpose::NoTranspose ? n_left : n_inner),
                B, Layout == MemoryLayout::RowMajor ? 
                    (TransposeB == Transpose::NoTranspose ? n_right : n_inner) :
                    (TransposeB == Transpose::NoTranspose ? n_inner : n_right),
                0.0, C, Layout == MemoryLayout::RowMajor ? n_right : n_left
            );
        }
        else {
            cblas_sgemm(
                Layout == MemoryLayout::RowMajor ? CblasRowMajor : CblasColMajor, 
                TransposeA == Transpose::NoTranspose ? CblasNoTrans : CblasTrans,
                TransposeB == Transpose::NoTranspose ? CblasNoTrans : CblasTrans,
                n_left, n_right, n_inner, 1.0, 
                A, 
                Layout == MemoryLayout::RowMajor ? (TransposeA == Transpose::NoTranspose ? n_inner : n_left) :
                (TransposeA == Transpose::NoTranspose ? n_left : n_inner),
                B, Layout == MemoryLayout::RowMajor ? 
                    (TransposeB == Transpose::NoTranspose ? n_right : n_inner) :
                    (TransposeB == Transpose::NoTranspose ? n_inner : n_right),
                0.0, C, Layout == MemoryLayout::RowMajor ? n_right : n_left
            );        
        }
    }



    /** 
     * Squares the matrix A and stores the result in C.
     * Computes C = opl(A) * opr(A) where op? is either the identity or the 
     * transpose. If opl = opr = identity, then this computes the square of A,
     * and A shall satissfy n_left = n_right. If opl = identity and 
     * opr = transpose, then this computes A * A^T, and if opl = transpose and 
     * opr = identity, then this computes A^T * A.
     * @tparam T The type of the elements in the matrices.
     * @tparam Layout The memory layout of the result matrix C (RowMajor or ColumnMajor).
     * @tparam TransposeL The transpose operation for the left-hand matrix A (NoTranspose or Transpose).
     * @tparam TransposeR The transpose operation for the right-hand matrix A (NoTranspose or Transpose).
     * @param n_left The number of rows in matrix A.
     * @param n_right The number of columns in matrix A.
     * @param C Pointer to the result matrix.
     * @param A Pointer to the input matrix.
     */
    template<typename T, MemoryLayout Layout = MemoryLayout::RowMajor, 
            Transpose TransposeL = Transpose::NoTranspose,
            Transpose TransposeR = Transpose::NoTranspose>
    void gemm(index_t n_left, index_t n_right, T* __restrict__ C, const T* __restrict__ A) {
        static_assert(std::is_same<T, double>::value || 
            std::is_same<T, float>::value, 
            "gemm is only implemented for double and float.");
        if (TransposeL == Transpose::NoTranspose && 
            TransposeR == Transpose::NoTranspose && n_left != n_right) {
            throw std::invalid_argument("Only square matrices can be squared "
                "without transposition.");
        }
        index_t nl = (TransposeL == Transpose::NoTranspose) ? n_left : n_right;
        index_t ni = (TransposeL == Transpose::NoTranspose) ? n_right : n_left;
        index_t nr = (TransposeR == Transpose::NoTranspose) ? n_right : n_left;
        if constexpr (std::is_same<T,double>::value) {
            cblas_dgemm(
                Layout == MemoryLayout::RowMajor ? CblasRowMajor : CblasColMajor, 
                TransposeL == Transpose::NoTranspose ? CblasNoTrans : CblasTrans,
                TransposeR == Transpose::NoTranspose ? CblasNoTrans : CblasTrans,
                nl, nr, ni, 1.0, 
                A, 
                Layout == MemoryLayout::RowMajor ? (TransposeL == Transpose::NoTranspose ? ni : nl) :
                (TransposeL == Transpose::NoTranspose ? nl : ni),
                A, Layout == MemoryLayout::RowMajor ? 
                    (TransposeR == Transpose::NoTranspose ? nr : ni) :
                    (TransposeR == Transpose::NoTranspose ? ni : nr),
                0.0, C, Layout == MemoryLayout::RowMajor ? nr : nl
            );
        }
        else {
            cblas_sgemm(
                Layout == MemoryLayout::RowMajor ? CblasRowMajor : CblasColMajor, 
                TransposeL == Transpose::NoTranspose ? CblasNoTrans : CblasTrans,
                TransposeR == Transpose::NoTranspose ? CblasNoTrans : CblasTrans,
                nl, nr, ni, 1.0, 
                A, 
                Layout == MemoryLayout::RowMajor ? (TransposeL == Transpose::NoTranspose ? ni : nl) :
                (TransposeL == Transpose::NoTranspose ? nl : ni),
                A, Layout == MemoryLayout::RowMajor ? 
                    (TransposeR == Transpose::NoTranspose ? nr : ni) :
                    (TransposeR == Transpose::NoTranspose ? ni : nr),
                0.0, C, Layout == MemoryLayout::RowMajor ? nr : nl
            );
        }
    }



    /** 
     * Set all elements of the array to zero.
     * TODO: implement a more efficient version (needs benchmarking)
     * @tparam T The type of the elements in the array.
     * @param size The number of elements in the array.
     * @param A Pointer to the array.
     */
    template<typename T>
    inline void zero(index_t size, T* A) {
        memset(A, 0, sizeof(T) * size);
    }



    /**
     * Computes the least integer n such that 2^n <= x.
     * @param x The input value.
     * @return The least integer n such that 2^n <= x.
     */
    template<std::unsigned_integral T>
    inline int ilog2(T x) {
        if (x == 0) {
            throw std::invalid_argument("ilog2 is only defined for positive integers.");
        }
        return std::bit_width(x) - 1;
    }



    /**
     * Computes the least integer n such that 2^n <= x.
     * @param x The input value.
     * @return The least integer n such that 2^n <= x.
     */
    template<std::signed_integral T>
    inline int ilog2(T x) {
        if (x <= 0) {
            throw std::invalid_argument("ilog2 is only defined for positive integers.");
        }
        return ilog2(static_cast<std::make_unsigned_t<T>>(x));
    }



    /**
     * Performs elementwise scalar multiplication on the array in-place.
     * @param size Number of elements in the array
     * @param A The array
     * @param b The scalar multiplier
     */
    template<typename T>
    inline void mul(index_t size, T* __restrict__ A, T b) {
#if defined(LIBARRAY_BLAS_ACCELERATE)
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>, 
            "mul is only implemented for double and float.");
        if constexpr (std::is_same_v<T, double>) {
            vDSP_vsmulD(A, 1, &b, A, 1, size);
        }
        else {
            vDSP_vsmul(A, 1, &b, A, 1, size);
        }
#elif defined(LIBARRAY_BLAS_MKL) || defined(LIBARRAY_BLAS_OPENBLAS) ||\
    defined(LIBARRAY_BLAS_GENERIC)
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>, 
            "mul is only implemented for double and float.");
        if constexpr (std::is_same_v<T, double>) {
            cblas_dscal(size, b, A, 1);
        }
        else {
            cblas_sscal(size, b, A, 1);
        }
#else
#error "mul not implemented!"
#endif // defined(LIBARRAY_BLAS_ACCELERATE)
    }



    /**
     * Performs elementwise multiplication of the arrays A and B and stores 
     * the result in C.
     * @param size Number of elements in the arrays
     * @param C The array to store the result
     * @param A The left-hand array
     * @param B The right-hand array
     */
    template<typename T>
    inline void mul(index_t size, T* __restrict__ C, const T* __restrict__ A, 
        const T* __restrict__ B) {
#if defined(LIBARRAY_BLAS_ACCELERATE)
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, float> || 
            std::is_same_v<T, DSPSplitComplex> || 
            std::is_same_v<T, DSPDoubleSplitComplex> ||
            std::is_same_v<T, DSPComplex> ||
            std::is_same_v<T, DSPDoubleComplex>,
            "mul is only implemented for double, float, DSPSplitComplex, "
            "DSPDoubleSplitComplex, and DSPComplex, and DSPDoubleComplex.");
        if constexpr (std::is_same_v<T, double>) {
            vDSP_vmulD(A, 1, B, 1, C, 1, size);
        }
        else if constexpr (std::is_same_v<T, float>) {
            vDSP_vmul(A, 1, B, 1, C, 1, size);
        }
        else if constexpr (std::is_same_v<T, DSPSplitComplex>) {
            vDSP_zvmul(A, 1, B, 1, C, 1, size, 1);
        }
        else if constexpr (std::is_same_v<T, DSPDoubleSplitComplex>) {
            vDSP_zvmulD(A, 1, B, 1, C, 1, size, 1);
        }
        else {
            // fall back to generic implementation
#pragma omp parallel for simd if (!omp_in_parallel()) schedule(static)
            for (index_t i = 0; i < size; ++i) {
                C[i] = {
                    A[i].real * B[i].real - A[i].imag * B[i].imag,
                    A[i].real * B[i].imag + A[i].imag * B[i].real
                };
            }
        }
#elif defined(LIBARRAY_BLAS_MKL)
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, float> ||
            std::is_same_v<T, MKL_Complex8> || 
            std::is_same_v<T, MKL_Complex16>,
            "mul is only implemented for double, float, MKL_Complex8, "
            "and MKL_Complex16.");
        if constexpr (std::is_same_v<T, double>) {
            vdMul(size, A, B, C);
        }
        else if constexpr (std::is_same_v<T, float>) {
            vsMul(size, A, B, C);
        }
        else if constexpr (std::is_same_v<T, MKL_Complex8>) {
            vcMul(size, A, B, C);
        }
        else if constexpr (std::is_same_v<T, MKL_Complex16>) {
            vzMul(size, A, B, C);
        }
#else
#error "mul not implemented!"
#endif // defined(LIBARRAY_BLAS_ACCELERATE)
    }


    /**
     * Performs elementwise multiplication of the arrays A and B and stores 
     * the result in A.
     * @param size Number of elements in the arrays
     * @param A The left-hand array and the array to store the result
     * @param B The right-hand array
     */
    template<typename T>
    inline void mul(index_t size, T* __restrict__ A, const T* __restrict__ B) {
#if defined(LIBARRAY_BLAS_ACCELERATE)
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, float> || 
            std::is_same_v<T, DSPSplitComplex> || 
            std::is_same_v<T, DSPDoubleSplitComplex> ||
            std::is_same_v<T, DSPComplex> ||
            std::is_same_v<T, DSPDoubleComplex>,
            "mul is only implemented for double, float, "
            "DSPComplex, DSPDoubleComplex, "
            "DSPSplitComplex, and DSPDoubleSplitComplex.");
        if constexpr (std::is_same_v<T, double>) {
            vDSP_vmulD(A, 1, B, 1, A, 1, size);
        }
        else if constexpr (std::is_same_v<T, float>) {
            vDSP_vmul(A, 1, B, 1, A, 1, size);
        }
        else if constexpr (std::is_same_v<T, DSPSplitComplex>) {
            vDSP_zvmul(A, 1, B, 1, A, 1, size, 1);
        }
        else if constexpr (std::is_same_v<T, DSPDoubleSplitComplex>) {
            vDSP_zvmulD(A, 1, B, 1, A, 1, size, 1);
        }
        else {
            // fall back to generic implementation
#pragma omp parallel for simd if (!omp_in_parallel()) schedule(static)
            for (index_t i = 0; i < size; ++i) {
                A[i] = {  
                    A[i].real * B[i].real - A[i].imag * B[i].imag,
                    A[i].real * B[i].imag + A[i].imag * B[i].real
                };
            }
        }
#elif defined(LIBARRAY_BLAS_MKL)
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, float> || 
            std::is_same_v<T, MKL_Complex8> || 
            std::is_same_v<T, MKL_Complex16>,
            "mul is only implemented for double, float, MKL_Complex8, "
            "and MKL_Complex16.");
        if constexpr (std::is_same_v<T, double>) {
            vdMul(size, A, B, A);
        }
        else if constexpr (std::is_same_v<T, float>) {
            vsMul(size, A, B, A);
        }
        else if constexpr (std::is_same_v<T, MKL_Complex8>) {
            vcMul(size, A, B, A);
        }
        else if constexpr (std::is_same_v<T, MKL_Complex16>) {
            vzMul(size, A, B, A);
        }
#else
#error "mul not implemented!"
#endif // defined(LIBARRAY_BLAS_ACCELERATE)
    }



    /**
     * Adds the product of two arrays A and B to the array C in-place.
     * @param size Number of elements in the arrays
     * @param C The array to store the result
     * @param A The left-hand array
     * @param B The right-hand array
     */
    template<typename T>
    inline void mul_add(index_t size, T* __restrict__ C, const T* __restrict__ A, 
        const T* __restrict__ B) {
#if defined(LIBARRAY_BLAS_ACCELERATE)
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, float> || 
            std::is_same_v<T, DSPSplitComplex> || 
            std::is_same_v<T, DSPDoubleSplitComplex>, 
            "mul_add is only implemented for double, float, DSPSplitComplex, "
            "and DSPDoubleSplitComplex.");
        if constexpr (std::is_same_v<T, double>) {
            vDSP_vmaD(A, 1, B, 1, C, 1, C, 1, size);
        }
        else if constexpr (std::is_same_v<T, float>) {
            vDSP_vma(A, 1, B, 1, C, 1, C, 1, size);
        }
        else if constexpr (std::is_same_v<T, DSPSplitComplex>) {
            vDSP_zvma(A, 1, B, 1, C, 1, C, 1, size);
        }
        else if constexpr (std::is_same_v<T, DSPDoubleSplitComplex>) {
            vDSP_zvmaD(A, 1, B, 1, C, 1, C, 1, size);
        }
        else {
            static_assert(false, "mul_add is only implemented for double, float, DSPSplitComplex, "
                "and DSPDoubleSplitComplex.");
        }
#elif defined(LIBARRAY_BLAS_MKL) || defined(LIBARRAY_BLAS_OPENBLAS)
        // there is no vector FMA primitive in VMA or OpenBLAS,
        // so we need to implement a loop ourselves
        // TODO: implement a more efficient version (requires benchmarking)
        // - Add parallel threshold
        // - Check AVX chunking
#pragma omp parallel for simd if (!omp_in_parallel()) schedule(static)
        for (index_t i = 0; i < size; ++i) {
            C[i] += A[i] * B[i];
        }
#else
#error "mul_add not implemented!"
#endif // defined(LIBARRAY_BLAS_ACCELERATE)
    }



    /**
     * Adds the array B into the array A in-place.
     * @param size Number of elements in the arrays
     * @param A The left-hand array and the array to store the result
     * @param B The right-hand array
     */
    template<typename T>
    inline void add(index_t size, T* __restrict__ A, const T* __restrict__ B) {
#if defined(LIBARRAY_BLAS_ACCELERATE)
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, float> || 
            std::is_same_v<T, DSPSplitComplex> || 
            std::is_same_v<T, DSPDoubleSplitComplex> ||
            std::is_same_v<T, DSPComplex> ||
            std::is_same_v<T, DSPDoubleComplex>,
            "add is only implemented for double, float, DSPSplitComplex, "
            "DSPDoubleSplitComplex, DSPComplex, and DSPDoubleComplex.");
        if constexpr (std::is_same_v<T, double>) {
            vDSP_vaddD(A, 1, B, 1, A, 1, size);
        }
        else if constexpr (std::is_same_v<T, float>) {
            vDSP_vadd(A, 1, B, 1, A, 1, size);
        }
        else if constexpr (std::is_same_v<T, DSPComplex>) {
            vDSP_vadd(reinterpret_cast<float*>(A), 1, 
            reinterpret_cast<const float*>(B), 1, 
            reinterpret_cast<float*>(A), 1, 2*size);
        }
        else if constexpr (std::is_same_v<T, DSPDoubleComplex>) {
            vDSP_vaddD(reinterpret_cast<double*>(A), 1, 
            reinterpret_cast<const double*>(B), 1, 
            reinterpret_cast<double*>(A), 1, 2*size);
        }
        else if constexpr (std::is_same_v<T, DSPSplitComplex>) {
            vDSP_zvadd(A, 1, B, 1, A, 1, size);
        }
        else if constexpr (std::is_same_v<T, DSPDoubleSplitComplex>) {
            vDSP_zvaddD(A, 1, B, 1, A, 1, size);
        }
        else {
            static_assert(false, "add is only implemented for double, float, DSPSplitComplex, "
                "and DSPDoubleSplitComplex.");
        }
#elif defined(LIBARRAY_BLAS_MKL) 
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, float> ||
            std::is_same_v<T, MKL_Complex8> ||
            std::is_same_v<T, MKL_Complex16>,
            "add is only implemented for double, float, MKL_Complex8, and MKL_Complex16.");
        if constexpr (std::is_same_v<T, double>) {
            vdAdd(size, A, B, A);
        }
        else if constexpr (std::is_same_v<T, float>) {
            vsAdd(size, A, B, A);
        }
        else if constexpr (std::is_same_v<T, MKL_Complex8>) {
            vcAdd(size, A, B, A);
        }
        else if constexpr (std::is_same_v<T, MKL_Complex16>) {
            vzAdd(size, A, B, A);
        }
        else {
            static_assert(false, "add is only implemented for double, float, MKL_Complex8, and MKL_Complex16.");
        }
#else
#error "add not implemented!"
#endif // defined(LIBARRAY_BLAS_ACCELERATE)
    }



    /**
     * Computes the median of an array.
     * **Note** The array is modified during the computation
     * TODO: make more efficient (requires benchmarking)
     * @tparam T The type of the elements in the array.
     * @param n The number of elements in the array.
     * @param A The array to compute the median of.
     */
    template<typename T>
    inline T median(index_t n, T* A) {
        auto mid = A + n / 2;
        std::nth_element(A, mid, A + n);
        if (n % 2 == 1)
            return *mid;

        T upper = *mid;
        return (*std::max_element(A, mid) + upper) / 2.0;

    }



    /**
     * Copies the source array to the destination array.
     * TODO: benchmark and check if the use of BLAS makes sense
     * @tparam T The type of the elements in the arrays.
     * @param n The number of elements to copy.
     * @param B Pointer to the destination array.
     * @param A Pointer to the source array.
     */
    template<typename T>
    inline void mov(index_t n, T* __restrict__ B, const T* __restrict__ A) {
        if constexpr (std::is_same_v<T,float>) {
            cblas_scopy(n, A, 1, B, 1);
        }
        else if constexpr (std::is_same_v<T,double>) {
            cblas_dcopy(n, A, 1, B, 1);
        }
        else if constexpr (std::is_same_v<T, complex_t<float>>) {
            cblas_ccopy(n, reinterpret_cast<const std::complex<float>*>(A), 
                1, reinterpret_cast<std::complex<float>*>(B), 1);
        }
        else if constexpr (std::is_same_v<T, complex_t<double>>) {
            cblas_zcopy(n, reinterpret_cast<const std::complex<double>*>(A), 
                1, reinterpret_cast<std::complex<double>*>(B), 1);
        }
#ifdef LIBARRAY_BLAS_ACCELERATE
        else if constexpr (std::is_same_v<T, split_complex_t<float>>) {
            vDSP_zvmov(A, 1, B, 1, n);
        }
        else if constexpr (std::is_same_v<T, split_complex_t<double>>) {
            vDSP_zvmovD(A, 1, B, 1, n);
        }
#endif // defined(LIBARRAY_BLAS_ACCELERATE)
        else {
            // fallback to generic implementation
#pragma omp parallel for simd if (!omp_in_parallel()) schedule(static)
            for (index_t i = 0; i < n; ++i) {
                B[i] = A[i];
            }
        }
    }
}

#endif // LIBARRAY_ARRAY_HPP
