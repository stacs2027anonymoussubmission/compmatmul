#ifndef COMPMATMUL_UTIL_HPP
#define COMPMATMUL_UTIL_HPP

#include "matmul.hpp"
#include <unistd.h>
#include <vector>
#include <chrono>
#include <string>

namespace compmatmul {
    std::string get_hostname() {
        char hostname[256];
        if (gethostname(hostname, sizeof(hostname)) != 0) {
            strcpy(hostname, "unknown");
        }
        hostname[sizeof(hostname) - 1] = '\0';
        return hostname;
    }

    class Timer {
    public:
        Timer() {
            stack.reserve(1000);
        }

        inline void push() {
            stack.push_back(std::chrono::high_resolution_clock::now());
        }

        inline double pop() {
            if (stack.empty()) {
                throw std::runtime_error("Timer stack is empty");
            }
            auto start = stack.back();
            stack.pop_back();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::nanoseconds elapsed = end - start;
            return elapsed.count() * 1e-9;
        }



    private:
        std::vector<std::chrono::high_resolution_clock::time_point> stack;
    };



    inline std::string get_blas_backend() {
#if defined(LIBARRAY_BLAS_OPENBLAS)
        return "openblas";
#elif defined(LIBARRAY_BLAS_MKL)
        return "mkl";
#elif defined(LIBARRAY_BLAS_ACCELERATE)
        return "accelerate";
#else
#error "Unknown BLAS backend"
        return "unknown";
#endif
    }



    inline std::string get_fft_backend() {
#if defined(COMPMATMUL_ENABLE_FFT)
#if defined(COMPMATMUL_USE_ACCELERATE_FFT)
        return "accelerate";
#elif defined(COMPMATMUL_USE_FFTW_FFT)
        return "fftw";
#elif defined(COMPMATMUL_USE_MKL_FFT)
        return "mkl";
#else
#error "Unknown FFT backend"
        return "unknown";
#endif
#else
        return "none";
#endif
    }




    inline std::string get_fwht_backend() {
#if defined(COMPMATMUL_ENABLE_FWHT)
#if defined(COMPMATMUL_USE_FXT_FWHT)
        return "fxt";
#elif defined(COMPMATMUL_USE_FFHT_FWHT)
        return "ffht";
#else
#error "Unknown FWHT backend"
        return "unknown";
#endif
#else
        return "none";
#endif
    }



    inline std::string to_string(Algorithm algorithm) {
        switch (algorithm) {
        case Algorithm::GEMM:
            return "gemm";
#ifdef COMPMATMUL_ENABLE_FFT
        case Algorithm::FFT:
            return "fft";
#endif // COMPMATMUL_ENABLE_FFT
#ifdef COMPMATMUL_ENABLE_FWHT
        case Algorithm::FWHT:
            return "fwht";
#endif // COMPMATMUL_ENABLE_FWHT
        default:
            return "unknown";
        }
    }



    inline std::string to_string(DType dtype) {
        switch (dtype) {
        case DType::Float32:
            return "float32";
        case DType::Float64:
            return "float64";
        default:
            return "unknown";
        }
    }



    inline std::string to_string(MemoryLayout layout) {
        switch (layout) {
        case MemoryLayout::RowMajor:
            return "row-major";
        case MemoryLayout::ColumnMajor:
            return "column-major";
        default:
            return "unknown";
        }
    }


    inline std::string to_string(Transpose transpose) {
        switch (transpose) {
        case Transpose::NoTranspose:
            return "no-transpose";
        case Transpose::Transpose:
            return "transpose";
        default:
            return "unknown";
        }
    }

    inline std::string to_string(HashFunction hash) {
        switch (hash) {
        case HashFunction::MultiplyAddShift:
            return "multiply-add-shift";
        default:
            return "unknown";
        }
    }

    inline std::string join(const std::vector<std::string>& vec, 
        const std::string& delimiter) {
        std::ostringstream oss;
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i != 0) {
                oss << delimiter;
            }
            oss << vec[i];
        }
        return oss.str();
    }

    class Row {
    public:
        explicit Row(const std::vector<std::string>& columns) : 
            columns(columns), values(columns.size()) {}

        std::string header() const {
            return join(columns, ",");
        }

        std::string to_string() const {
            return join(values, ",");
        }

        void set(const std::string& column, const std::string& value) {
            auto it = std::find(columns.begin(), columns.end(), column);
            if (it == columns.end()) {
                throw std::runtime_error("Column not found");
            }
            else {
                size_t index = std::distance(columns.begin(), it);
                values[index] = value;
            }
        }

    private:
        std::vector<std::string> columns;
        std::vector<std::string> values;
    };



    inline std::string get_timestamp() {
        auto now_time = std::chrono::floor<std::chrono::seconds>(
        std::chrono::system_clock::now());
        std::string timestamp = std::format("{:%Y-%m-%d %H:%M:%S}+00:00", now_time);
        return timestamp;
    }
}

#endif // COMPMATMUL_UTIL_HPP