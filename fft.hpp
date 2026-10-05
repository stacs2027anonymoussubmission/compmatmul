#ifndef COMPMATMUL_FFT_HPP
#define COMPMATMUL_FFT_HPP


/**
 * @file fft.hpp
 * This file contains helper functions for performing FFT/IFFT.
 */

#include "libarray/array.hpp"

#include <type_traits>
#include <cassert>

#ifdef COMPMATMUL_USE_ACCELERATE_FFT
#include <Accelerate/Accelerate.h>
#endif // COMPMATMUL_USE_ACCELERATE_FFT

#ifdef COMPMATMUL_USE_MKL_FFT
#include <mkl.h>
#endif // COMPMATMUL_USE_MKL_FFT

#ifdef COMPMATMUL_USE_FFTW_FFT
#include <fftw3.h>
#endif // COMPMATMUL_USE_FFTW_FFT



#ifdef COMPMATMUL_ENABLE_FFT
namespace compmatmul {
    using array::index_t;
    using array::ilog2;

#ifdef COMPMATMUL_USE_ACCELERATE_FFT
    /**
     * Performs FFT/IFFT using the Accelerate framework. 
     * This class is a wrapper around the vDSP library functions.
     * It supports both single and double precision floating point types. 
     * Only real-valued transforms are supported.
     * @tparam T The type of the input and output data (float or double)
     */
    template<typename T>
    class FFTTransformer {
    public:
        static_assert(std::is_same_v<T, float> ||
                     std::is_same_v<T, double>,
                      "FFT only supports float and double");


        using split_complex_t = std::conditional_t<std::is_same_v<T, float>, 
                DSPSplitComplex, DSPDoubleSplitComplex>;


        /**
         * The constructor initializes the FFT setup object for the given size n.
         * @param n The size of the FFT to be performed (must be a power of 2)
         */
        FFTTransformer(index_t n) :
            n(n), log2n(ilog2(static_cast<uint64_t>(n))) {
            if (n <= 0 || (n & (n - 1)) != 0) {
                throw std::invalid_argument("FFT size must be a positive power of 2");
            }
            if constexpr (std::is_same_v<T, float>) {
                fft_setup = vDSP_create_fftsetup(log2n, kFFTRadix2);
            } 
            else {
                fft_setup = vDSP_create_fftsetupD(log2n, kFFTRadix2);
            }
        }



        /**
         * The destructor deallocates the FFT setup object.
         */
        ~FFTTransformer() {
            if constexpr (std::is_same_v<T, float>) {
                vDSP_destroy_fftsetup(fft_setup);
            } 
            else {
                vDSP_destroy_fftsetupD(fft_setup);
            }
        }



        /**
         * Performs the forward FFT on the input array x and stores the 
         * result in y.
         * **Note** This uses Accelerate's split-complex format and does not
         * scale the output, so the output will be off by a factor of 2.
         * @param output The output array (complex) of size n/2
         * @param input The input array (real); the element x[2i] is stored in input.real[i] and x[2i+1] is stored in input.imag[i]
         * @param scratch A scratch space for temporary computations (array of length n/2, must be aligned to 64 bytes)
         */
        inline void fwd(split_complex_t* output, const split_complex_t* input, split_complex_t* scratch) {
            if constexpr (std::is_same_v<T, float>) {
                vDSP_fft_zropt(fft_setup, input, 1, output, 1, scratch, log2n, kFFTDirection_Forward);
            } 
            else {
                vDSP_fft_zroptD(fft_setup, input, 1, output, 1, scratch, log2n, kFFTDirection_Forward);
            }
        }
        
        
        
        /**
         * Performs the IFFT on the input array x and stores the 
         * result in y.
         * **Note** This uses Accelerate's split-complex format and does not
         * scale the output, so the output will be off by a factor of n. If
         * you previously applied the unscaled fwd, it's off by a factor of 2n.
         * @param output The output array (complex) of size n/2
         * @param input The input array (real); the element x[2i] is stored in input.real[i] and x[2i+1] is stored in input.imag[i]
         * @param scratch A scratch space for temporary computations (array of length n/2, must be aligned to 64 bytes)
         */
        inline void inv(split_complex_t* output, const split_complex_t* input, split_complex_t* scratch) {
            if constexpr (std::is_same_v<T, float>) {
                vDSP_fft_zropt(fft_setup, input, 1, output, 1, scratch, log2n, kFFTDirection_Inverse);
            } 
            else {
                vDSP_fft_zroptD(fft_setup, input, 1, output, 1, scratch, log2n, kFFTDirection_Inverse);
            }
        }



        /**
         * Performs the forward FFT on the input array x and stores the 
         * result in y.
         * **Note** This scales and maps the input into Accelerate-specific 
         * format, so it has subpar performance. Use the other fwd() function 
         * if you can provide the input in Accelerate format.
         * **Note** output[0] contains the real part of the DC component, and
         * output[1] contains the real part of the Nyquist frequency component.
         * @param output The output array (complex) of size n
         * @param input The input array (real) of size n
         * @param scratch A scratch space for temporary computations (array of length 3*n, must be aligned to 64 bytes)
         */
        inline void fwd(T* output, const T* input, T* scratch) {
            split_complex_t input_split;
            input_split.realp = scratch;
            input_split.imagp = scratch + n/2;
            split_complex_t output_split;
            output_split.realp = scratch + n;
            output_split.imagp = scratch + n + n/2;
            split_complex_t buffer;
            buffer.realp = scratch + 2*n;
            buffer.imagp = scratch + 2*n + n/2;

            if constexpr (std::is_same_v<T, float>) {
                vDSP_ctoz(reinterpret_cast<const DSPComplex*>(input), 2, &input_split, 1, n/2);
                vDSP_fft_zropt(fft_setup, &input_split, 1, &output_split, 1, &buffer, log2n, kFFTDirection_Forward);
                vDSP_ztoc(&output_split, 1, reinterpret_cast<DSPComplex*>(output), 2, n/2);
            } 
            else {
                vDSP_ctozD(reinterpret_cast<const DSPDoubleComplex*>(input), 2, &input_split, 1, n/2);
                vDSP_fft_zroptD(fft_setup, &input_split, 1, &output_split, 1, &buffer, log2n, kFFTDirection_Forward);
                vDSP_ztocD(&output_split, 1, reinterpret_cast<DSPDoubleComplex*>(output), 2, n/2);
            }
            array::mul(n, output, static_cast<T>(0.5));
        }



        /**
         * Performs the inverse FFT on the input array x and stores the 
         * result in y.
         * **Note** This scales and maps the input into Accelerate-specific 
         * format, so it has subpar performance. Use the other inv() function 
         * if you can provide the input in Accelerate format.
         * @param output The output array (complex) of size n/2+1
         * @param input The input array (real) of size n
         * @param scratch A scratch space for temporary computations (array of length 3*n, must be aligned to 64 bytes)
         */
        inline void inv(T* output, const T* input, T* scratch) {
            split_complex_t input_split;
            input_split.realp = scratch;
            input_split.imagp = scratch + n/2;
            split_complex_t output_split;
            output_split.realp = scratch + n;
            output_split.imagp = scratch + n + n/2;
            split_complex_t buffer;
            buffer.realp = scratch + 2*n;
            buffer.imagp = scratch + 2*n + n/2;

            if constexpr (std::is_same_v<T, float>) {
                vDSP_ctoz(reinterpret_cast<const DSPComplex*>(input), 2, &input_split, 1, n/2);
                vDSP_fft_zropt(fft_setup, &input_split, 1, &output_split, 1, &buffer, log2n, kFFTDirection_Inverse);
                vDSP_ztoc(&output_split, 1, reinterpret_cast<DSPComplex*>(output), 2, n/2);
            } 
            else {
                vDSP_ctozD(reinterpret_cast<const DSPDoubleComplex*>(input), 2, &input_split, 1, n/2);
                vDSP_fft_zroptD(fft_setup, &input_split, 1, &output_split, 1, &buffer, log2n, kFFTDirection_Inverse);
                vDSP_ztocD(&output_split, 1, reinterpret_cast<DSPDoubleComplex*>(output), 2, n/2);
            }
            array::mul(n, output, 1/static_cast<T>(n));
        }



        /**
         * Computes the size of the scratch space required for the FFT/IFFT operations.
         * @return The size of the scratch space in number of elements.
         */
        inline index_t scratch_size() const {
            return 3*n;
        }




        FFTTransformer(const FFTTransformer&) = delete;
        FFTTransformer& operator=(const FFTTransformer&) = delete;
        FFTTransformer(FFTTransformer&&) = delete;
        FFTTransformer& operator=(FFTTransformer&&) = delete;

    private:
        using fft_setup_t = typename std::conditional_t<std::is_same_v<T, float>, 
            FFTSetup, FFTSetupD>;
        index_t n; ///< The size of the FFT
        index_t log2n; ///< The log2 of the size of the FFT
        fft_setup_t fft_setup; ///< The FFT setup object
    };
#endif // COMPMATMUL_USE_ACCELERATE_FFT

#ifdef COMPMATMUL_USE_MKL_FFT
    /**
     * Performs FFT/IFFT using the Intel MKL library. 
     * This class is a wrapper around the MKL DFTI interface.
     * It supports both single and double precision floating point types. 
     * Only real-valued transforms are supported.
     * @tparam T The type of the input and output data (float or double)
     */
    template<typename T>
    class FFTTransformer {
    public:
        static_assert(std::is_same_v<T, float> ||
                     std::is_same_v<T, double>,
                      "FFT only supports float and double");

        using complex_t = array::complex_t<T>;


        /**
         * Initializes a real FFT of size n. The size must be a power of 2.
         * @param n The size of the FFT to be performed (must be a power of 2)
         */
        FFTTransformer(index_t n) : n(n) {
            if (n <= 0 || (n & (n - 1)) != 0) {
                throw std::invalid_argument("FFT size must be a positive power of 2");
            }

            if (DftiCreateDescriptor(&descriptor, 
                (std::is_same_v<T, float> ? DFTI_SINGLE : DFTI_DOUBLE), 
                DFTI_REAL, 
                1, // dimension of the transform
                n) != DFTI_NO_ERROR) {
                throw std::runtime_error("Failed to create DFTI descriptor");
            }

            if (DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE) != DFTI_NO_ERROR) {
                throw std::runtime_error("Failed to set DFTI_PLACEMENT");
            }
            if (DftiSetValue(descriptor, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX) != DFTI_NO_ERROR) {
                throw std::runtime_error("Failed to set DFTI_CONJUGATE_EVEN_STORAGE");
            }
            if (DftiSetValue(descriptor, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT) != DFTI_NO_ERROR) {
                throw std::runtime_error("Failed to set DFTI_PACKED_FORMAT");
            }
            if (DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0/n) != DFTI_NO_ERROR) {
                throw std::runtime_error("Failed to set DFTI_BACKWARD_SCALE");
            }
            if (DftiCommitDescriptor(descriptor) != DFTI_NO_ERROR) {
                throw std::runtime_error("Failed to commit DFTI descriptor");
            }
        }

        ~FFTTransformer() {
            DftiFreeDescriptor(&descriptor);
        }

        FFTTransformer(const FFTTransformer&) = delete;
        FFTTransformer& operator=(const FFTTransformer&) = delete;
        FFTTransformer(FFTTransformer&&) = delete;
        FFTTransformer& operator=(FFTTransformer&&) = delete;

        /**
         * Performs the forward FFT on the input array and stores the result in the output array.
         * @param output The output array (complex) of size n/2+1
         * @param input The input array (real) of size n
         */
        void fwd(complex_t* __restrict__ output, const T* __restrict__ input) {
            // MKL api is not const-correct, so we need to cast away constness
            DftiComputeForward(descriptor, const_cast<T*>(input), reinterpret_cast<T*>(output));
        }

        /**
         * Performs the inverse FFT on the input array and stores the result in the output array.
         * Note: The output will be scaled by 1/n, where n is the size of the FFT.
         * Thus, inv(fwd(x)) = x, up to numerical errors.
         * @param output The output array (real) of size n
         * @param input The input array (complex) of size n/2+1
         */
        void inv(T* __restrict__ output, const complex_t* __restrict__ input) {
            // MKL api is not const-correct, so we need to cast away constness
            DftiComputeBackward(descriptor, 
                const_cast<T*>(reinterpret_cast<const T*>(input)), output);
        }

    private:
        index_t n; ///< The size of the FFT
        DFTI_DESCRIPTOR_HANDLE descriptor; ///< The MKL DFTI descriptor
    };
#endif // COMPMATMUL_USE_MKL_FFT


#ifdef COMPMATMUL_USE_FFTW_FFT
    /**
     * Performs FFT/IFFT using the FFTW library. 
     * This class is a wrapper around the FFTW interface.
     * It supports both single and double precision floating point types. 
     * Only real-valued transforms are supported.
     * @tparam T The type of the input and output data (float or double)
     */
    template<typename T>
    class FFTTransformer {
    public:
        static_assert(std::is_same_v<T, float> ||
                     std::is_same_v<T, double>,
                      "FFT only supports float and double");

        using complex_t = array::complex_t<T>;

        /**
         * Initializes a real FFT of size n. The size must be a power of 2.
         * The scratch array must be of size 2*n+2, as it will be used to
         * construct the FFTW plan (n floats of input, n/2+1 complex floats of output). 
         * The scratch array must be aligned to 64 bytes.
         * @param n The size of the FFT to be performed (must be a power of 2)
         * @param scratch Pointer to the scratch array
         */
        FFTTransformer(index_t n, T* scratch) : 
            n(n), fwd_plan(nullptr), inv_plan(nullptr) {
            if (n <= 0 || (n & (n - 1)) != 0) {
                throw std::invalid_argument("FFT size must be a positive power of 2");
            }
            // The scratch array to be aligned to 64 bytes
            if (reinterpret_cast<uintptr_t>(scratch) % 64 != 0) {
                throw std::invalid_argument("Scratch array must be aligned to 64 bytes");
            }

            if constexpr (std::is_same_v<T, float>) {
                fwd_plan = fftwf_plan_dft_r2c_1d(n, scratch, 
                    reinterpret_cast<fftwf_complex*>(scratch+n), 
                    FFTW_MEASURE | FFTW_PRESERVE_INPUT);
                inv_plan = fftwf_plan_dft_c2r_1d(n, 
                    reinterpret_cast<fftwf_complex*>(scratch+n), scratch, 
                    FFTW_MEASURE | FFTW_DESTROY_INPUT);
            } 
            else if constexpr (std::is_same_v<T, double>) {
                fwd_plan = fftw_plan_dft_r2c_1d(n, scratch, 
                    reinterpret_cast<fftw_complex*>(scratch+n), 
                    FFTW_MEASURE | FFTW_PRESERVE_INPUT);
                inv_plan = fftw_plan_dft_c2r_1d(n, 
                    reinterpret_cast<fftw_complex*>(scratch+n), scratch, 
                    FFTW_MEASURE | FFTW_DESTROY_INPUT);
            }

            if (!fwd_plan || !inv_plan) {
                throw std::runtime_error("Failed to create FFTW plan");
            }
        }

        /**
         * Performs the forward FFT on the input array and stores the result 
         * in the output array.
         * The function shall not modify the input array.
         * @param output The output array (complex) of size n/2+1
         * @param input The input array (real) of size n
         */
        void fwd(complex_t* __restrict__ output, const T* __restrict__ input) {
            assert(reinterpret_cast<uintptr_t>(input) % 64 == 0);
            assert(reinterpret_cast<uintptr_t>(output) % 64 == 0);
            if constexpr (std::is_same_v<T, float>) {
                fftwf_execute_dft_r2c(fwd_plan, const_cast<T*>(input), 
                    reinterpret_cast<fftwf_complex*>(output));
            } 
            else if constexpr (std::is_same_v<T, double>) {
                fftw_execute_dft_r2c(fwd_plan, const_cast<T*>(input), 
                    reinterpret_cast<fftw_complex*>(output));
            }
        }

        /**
         * Performs the inverse FFT on the input array and stores the result 
         * in the output array.
         * Note: The output will be **not** be scaled, so to get the original 
         * input back, you need to scale the output by 1/n, where n is the 
         * size of the FFT.
         * Thus, inv(fwd(x)) = x, up to numerical errors.
         * The function may destroy the content of the input array, 
         * so it should not be used after this call.
         * @param output The output array (real) of size n
         * @param input The input array (complex) of size n/2+1
         */
        void inv(T* __restrict__ output, complex_t* __restrict__ input) {
            assert(reinterpret_cast<uintptr_t>(input) % 64 == 0);
            assert(reinterpret_cast<uintptr_t>(output) % 64 == 0);
            if constexpr (std::is_same_v<T, float>) {
                fftwf_execute_dft_c2r(inv_plan, 
                    reinterpret_cast<fftwf_complex*>(input), output);
            } 
            else if constexpr (std::is_same_v<T, double>) {
                fftw_execute_dft_c2r(inv_plan, 
                    reinterpret_cast<fftw_complex*>(input), output);
            }
        }

        ~FFTTransformer() {
            if constexpr (std::is_same_v<T, float>) {
                fftwf_destroy_plan(fwd_plan);
                fftwf_destroy_plan(inv_plan);
            } 
            else if constexpr (std::is_same_v<T, double>) {
                fftw_destroy_plan(fwd_plan);
                fftw_destroy_plan(inv_plan);
            }
        }

        FFTTransformer(const FFTTransformer&) = delete;
        FFTTransformer& operator=(const FFTTransformer&) = delete;
        FFTTransformer(FFTTransformer&&) = delete;
        FFTTransformer& operator=(FFTTransformer&&) = delete;

    private:
        using plan_t = typename std::conditional_t<std::is_same_v<T, float>, 
            fftwf_plan, fftw_plan>;

        index_t n; ///< The size of the FFT
        plan_t fwd_plan; ///< The FFTW plan for the forward transform
        plan_t inv_plan; ///< The FFTW plan for the inverse transform
    };
#endif // COMPMATMUL_USE_FFTW_FFT
}
#endif // COMPMATMUL_ENABLE_FFT

#endif // COMPMATMUL_FFT_HPP