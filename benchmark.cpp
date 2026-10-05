#include "util.hpp"
#include "matmul.hpp"
#include "tclap/CmdLine.h"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <print>

using array::Transpose;
using array::MemoryLayout;
using array::index_t;
using array::random::RNG_t;
using compmatmul::Algorithm;
using compmatmul::DType;
using compmatmul::HashFunction;
using compmatmul::MultiplyAddShiftHash;
#ifdef COMPMATMUL_ENABLE_FFT
using compmatmul::real_poly_mem_size;
#endif // COMPMATMUL_ENABLE_FFT
using compmatmul::get_blas_backend;
using compmatmul::get_fft_backend;
using compmatmul::get_fwht_backend;
using compmatmul::to_string;
using compmatmul::get_timestamp;
using compmatmul::get_hostname;
using compmatmul::Timer;
using compmatmul::Row;



static void benchmark(Algorithm algorithm, DType dtype, 
    index_t n_left, index_t n_inner, index_t n_right,
    int d, int b, 
    MemoryLayout layout, Transpose transpose_a, Transpose transpose_b, 
    int n_reps, uint64_t seed) {
    std::println(stderr, "benchmarking...");

    std::string blas_backend = get_blas_backend();
    std::string fft_backend = get_fft_backend();
    std::string fwht_backend = get_fwht_backend();

    std::println(stderr, "BLAS backend: {}", blas_backend);
    std::println(stderr, "FFT backend: {}", fft_backend);
    std::println(stderr, "FWHT backend: {}", fwht_backend);

    std::println(stderr, "algorithm: {}", to_string(algorithm));
    std::println(stderr, "data type: {}", to_string(dtype));

    HashFunction hash = HashFunction::MultiplyAddShift;
    std::println(stderr, "hash function: {}", to_string(hash));

    std::println(stderr, "dimensions: {} x {} x {}", n_left, n_inner, n_right);
    std::println(stderr, "d: {}", d);
    std::println(stderr, "b: {}", b);
    std::println(stderr, "layout: {}", to_string(layout));
    std::println(stderr, "transpose A: {}", to_string(transpose_a));
    std::println(stderr, "transpose B: {}", to_string(transpose_b));
    std::println(stderr, "repetitions: {}", n_reps);
    std::println(stderr, "seed: {}", seed);

    std::string hostname = get_hostname();

    std::string timestamp = get_timestamp();
    std::println(stderr, "timestamp: {}", timestamp);
    std::println(stderr, "hostname: {}", hostname);
    
    Timer timer;

    timer.push();
    void* A;
    void* B;
    void* C;
    void* p = nullptr;
    void* scratch = nullptr;

    RNG_t rng(seed);
    if (dtype == DType::Float32) {
        A = array::allocate<float>(n_left * n_inner);
        B = array::allocate<float>(n_inner * n_right);
        C = array::allocate<float>(n_left * n_right);

        if (A == nullptr || B == nullptr || C == nullptr) {
            throw std::runtime_error("Failed to allocate matrices");
        }

        rng.fill(n_left * n_inner, static_cast<float*>(A));
        rng.fill(n_inner * n_right, static_cast<float*>(B));

        if (d > 0) {
#ifdef COMPMATMUL_ENABLE_FFT
            if (algorithm == Algorithm::FFT) {
                p = array::allocate<float>(d*real_poly_mem_size<float>(b));
            }
#endif // COMPMATMUL_ENABLE_FFT
#ifdef COMPMATMUL_ENABLE_FWHT
            if (algorithm == Algorithm::FWHT) {
                p = array::allocate<float>(d*b);
            }
#endif // COMPMATMUL_ENABLE_FWHT

            if (p == nullptr) {
                throw std::runtime_error("Failed to allocate polynomial sketch");
            }
        }
    

#ifdef COMPMATMUL_ENABLE_FFT
        if (algorithm == Algorithm::FFT) {
            scratch = array::allocate<float>(
                compmatmul::matmul_compress_and_multiply_fft_scratch_size<float>(d, b)
            );
            if (scratch == nullptr) {
                throw std::runtime_error("Failed to allocate scratch space for FFT");
            }
        }
#endif // COMPMATMUL_ENABLE_FFT
#ifdef COMPMATMUL_ENABLE_FWHT
        if (algorithm == Algorithm::FWHT) {
            scratch = array::allocate<float>(
                compmatmul::matmul_compress_and_multiply_fwht_scratch_size(d, b)
            );
            if (scratch == nullptr) {
                throw std::runtime_error("Failed to allocate scratch space for FWHT");
            }
        }
#endif // COMPMATMUL_ENABLE_FWHT
    }
    else if (dtype == DType::Float64) {
        A = array::allocate<double>(n_left * n_inner);
        B = array::allocate<double>(n_inner * n_right);
        C = array::allocate<double>(n_left * n_right);
        if (A == nullptr || B == nullptr || C == nullptr) {
            throw std::runtime_error("Failed to allocate matrices");
        }
        rng.fill(n_left * n_inner, static_cast<double*>(A));
        rng.fill(n_inner * n_right, static_cast<double*>(B));

        if (d > 0) {
#ifdef COMPMATMUL_ENABLE_FFT
            if (algorithm == Algorithm::FFT) {
                p = array::allocate<double>(d*real_poly_mem_size<double>(b));
            }
#endif // COMPMATMUL_ENABLE_FFT
#ifdef COMPMATMUL_ENABLE_FWHT
            if (algorithm == Algorithm::FWHT) {
                p = array::allocate<double>(d*b);
            }
#endif // COMPMATMUL_ENABLE_FWHT
            if (p == nullptr) {
                throw std::runtime_error("Failed to allocate polynomial sketch");
            }
        }

#ifdef COMPMATMUL_ENABLE_FFT
        if (algorithm == Algorithm::FFT) {
            scratch = array::allocate<double>(
                compmatmul::matmul_compress_and_multiply_fft_scratch_size<double>(d, b)
            );
            if (scratch == nullptr) {
                throw std::runtime_error("Failed to allocate scratch space for FFT");
            }
        }
#endif // COMPMATMUL_ENABLE_FFT
#ifdef COMPMATMUL_ENABLE_FWHT
        if (algorithm == Algorithm::FWHT) {
            scratch = array::allocate<double>(
                compmatmul::matmul_compress_and_multiply_fwht_scratch_size(d, b)
            );
            if (scratch == nullptr) {
                throw std::runtime_error("Failed to allocate scratch space for FWHT");
            }
        }
#endif // COMPMATMUL_ENABLE_FWHT
    }
    else {
        throw std::runtime_error("Unsupported dtype");
    }
    
    double t = timer.pop();

    std::println(stderr, "time to allocate and fill matrices: {:f} seconds", t);

    MultiplyAddShiftHash::param_t* ha_params = nullptr;
    MultiplyAddShiftHash::param_t* sa_params = nullptr;
    MultiplyAddShiftHash::param_t* hb_params = nullptr;
    MultiplyAddShiftHash::param_t* sb_params = nullptr;

    timer.push();
    if (d > 0) {
        ha_params = array::allocate<MultiplyAddShiftHash::param_t>(
            MultiplyAddShiftHash::NUM_PARAMS * d);
        sa_params = array::allocate<MultiplyAddShiftHash::param_t>(
            MultiplyAddShiftHash::NUM_PARAMS * d);
        hb_params = array::allocate<MultiplyAddShiftHash::param_t>(
            MultiplyAddShiftHash::NUM_PARAMS * d);
        sb_params = array::allocate<MultiplyAddShiftHash::param_t>(
            MultiplyAddShiftHash::NUM_PARAMS * d);
        rng.fill(MultiplyAddShiftHash::NUM_PARAMS * d, ha_params);
        rng.fill(MultiplyAddShiftHash::NUM_PARAMS * d, sa_params);
        rng.fill(MultiplyAddShiftHash::NUM_PARAMS * d, hb_params);
        rng.fill(MultiplyAddShiftHash::NUM_PARAMS * d, sb_params);
    }
    t = timer.pop();
    std::println(stderr, "time to allocate and fill hash parameters: {:f} seconds", t);

    const std::vector<std::string> columns = {
        "timestamp", "hostname",
        "blas_backend", "fft_backend", "fwht_backend",
        "algorithm", "dtype", "hash",
        "n_left", "n_inner", "n_right",
        "d", "b", 
        "layout", "transpose_a", "transpose_b",
        "rep", "seed", "time"
    };

    for (int rep = 0; rep < n_reps; ++rep) {
        timer.push();
        matmul_static(algorithm, hash, dtype, 
            n_left, n_inner, n_right,
            C, A, B, p, scratch,
            ha_params, sa_params, hb_params, sb_params,
            d, b, layout, transpose_a, transpose_b);
        t = timer.pop();

        Row row(columns);
        row.set("timestamp", timestamp);
        row.set("hostname", hostname);
        row.set("blas_backend", blas_backend);
        row.set("fft_backend", fft_backend);
        row.set("fwht_backend", fwht_backend);
        row.set("algorithm", to_string(algorithm));
        row.set("dtype", to_string(dtype));
        row.set("hash", to_string(hash));
        row.set("n_left", std::to_string(n_left));
        row.set("n_inner", std::to_string(n_inner));
        row.set("n_right", std::to_string(n_right));
        row.set("d", std::to_string(d));
        row.set("b", std::to_string(b));
        row.set("layout", to_string(layout));
        row.set("transpose_a", to_string(transpose_a));
        row.set("transpose_b", to_string(transpose_b));
        row.set("rep", std::to_string(rep));
        row.set("seed", std::to_string(seed));
        row.set("time", std::to_string(t));

        if (rep == 0) {
            std::println(stdout, "{}", row.header());
        }
        std::println(stdout, "{}", row.to_string());
    }

    array::deallocate(A);
    array::deallocate(B);
    array::deallocate(C);
    array::deallocate(ha_params);
    array::deallocate(sa_params);
    array::deallocate(hb_params);
    array::deallocate(sb_params);
    array::deallocate(p);
    array::deallocate(scratch);
}

int main(int argc, char* argv[]) {
    try {
        TCLAP::CmdLine cmd("Benchmarking tool for compmatmul", ' ', "0.1.0");

        std::vector<std::string> algorithms { "gemm", 
#ifdef COMPMATMUL_ENABLE_FFT
            "fft", 
#endif // COMPMATMUL_ENABLE_FFT
#ifdef COMPMATMUL_ENABLE_FWHT
            "fwht" 
#endif // COMPMATMUL_ENABLE_FWHT
        };
        std::vector<std::string> dtypes { "float32", "float64" };

        TCLAP::ValuesConstraint<std::string> algorithm_constraint(algorithms);
        TCLAP::ValuesConstraint<std::string> dtype_constraint(dtypes);

        TCLAP::UnlabeledValueArg<std::string> algorithm_arg("algorithm", 
            "Algorithm", true, "", &algorithm_constraint);
        TCLAP::UnlabeledValueArg<std::string> dtype_arg("dtype", "Data type", true, "float64", &dtype_constraint);
        TCLAP::UnlabeledValueArg<int> n_left_arg("n_left", "Left-hand side dimension", true, -1, "integer");
        TCLAP::UnlabeledValueArg<int> n_inner_arg("n_inner", "Inner dimension", true, -1, "integer");
        TCLAP::UnlabeledValueArg<int> n_right_arg("n_right", "Right-hand side dimension", true, -1, "integer");
        cmd.add(algorithm_arg);
        cmd.add(dtype_arg);
        cmd.add(n_left_arg);
        cmd.add(n_inner_arg);
        cmd.add(n_right_arg);

        TCLAP::ValueArg<int> d_arg("d", "d", "Number of sketches", false, -1, "integer");
        TCLAP::ValueArg<int> b_arg("b", "b", "Length of polynomial", false, -1, "integer");
        cmd.add(d_arg);
        cmd.add(b_arg);

        TCLAP::ValueArg<int> n_reps_arg("r", "reps", "Number of repetitions", false, 1, "integer");
        cmd.add(n_reps_arg);

        TCLAP::SwitchArg transposea_arg("", "transpose-a", "Transpose A (left-hand side)");
        cmd.add(transposea_arg);
        TCLAP::SwitchArg transposeb_arg("", "transpose-b", "Transpose B (right-hand side)");
        cmd.add(transposeb_arg);

        TCLAP::ValueArg<uint64_t> seed_arg("s", "seed", "Random seed", false, 0, "integer");
        cmd.add(seed_arg);

        cmd.parse(argc, argv);

        MemoryLayout layout = MemoryLayout::RowMajor;
        Transpose transpose_a = transposea_arg ? Transpose::Transpose : Transpose::NoTranspose;
        Transpose transpose_b = transposeb_arg ? Transpose::Transpose : Transpose::NoTranspose;

        int n_reps = n_reps_arg.getValue();
        if (n_reps <= 0) {
            std::println(stderr, "{}: Number of repetitions must be positive", argv[0]);
            return EXIT_FAILURE;
        }

        index_t n_left, n_inner, n_right;
        n_left = n_left_arg.getValue();
        n_inner = n_inner_arg.getValue();
        n_right = n_right_arg.getValue();

        if (n_left <= 0 || n_inner <= 0 || n_right <= 0) {
            std::println(stderr, "{}: All dimensions must be positive", argv[0]);
            return EXIT_FAILURE;
        }

        Algorithm algorithm = Algorithm::GEMM;
#ifdef COMPMATMUL_ENABLE_FFT
        if (algorithm_arg.getValue() == "fft") {
            algorithm = Algorithm::FFT;
        } 
        else 
#endif // COMPMATMUL_ENABLE_FFT
#ifdef COMPMATMUL_ENABLE_FWHT
        if (algorithm_arg.getValue() == "fwht") {
            algorithm = Algorithm::FWHT;
        }
        else 
#endif // COMPMATMUL_ENABLE_FWHT
        if (algorithm_arg.getValue() == "gemm") {
            algorithm = Algorithm::GEMM;
        }
        else {
            std::println(stderr, "{}: Unknown algorithm: {}", argv[0], algorithm_arg.getValue());
            return EXIT_FAILURE;
        }

        DType dtype = DType::Float64;
        if (dtype_arg.getValue() == "float32") {
            dtype = DType::Float32;
        } 
        else if (dtype_arg.getValue() == "float64") {
            dtype = DType::Float64;
        }
        else {
            std::println(stderr, "{}: Unknown data type: {}", argv[0], dtype_arg.getValue());
            return EXIT_FAILURE;
        }

        int d = d_arg.getValue();
        int b = b_arg.getValue();
        if (
#if defined(COMPMATMUL_ENABLE_FFT) && defined(COMPMATMUL_ENABLE_FWHT)
            algorithm == Algorithm::FFT || algorithm == Algorithm::FWHT
#elif defined(COMPMATMUL_ENABLE_FFT)
            algorithm == Algorithm::FFT
#elif defined(COMPMATMUL_ENABLE_FWHT)
            algorithm == Algorithm::FWHT
#else
            false
#endif
        ) {
            if (d <= 0 || b <= 0 || (b & (b - 1)) != 0) {
                std::println(stderr, "{}: algorithm {} requires positive "
                    "d and b parameters, and b must be a power of 2", 
                    argv[0], to_string(algorithm));
                return EXIT_FAILURE;
            }
        }

        uint64_t seed;
        if (seed_arg.isSet()) {
            seed = seed_arg.getValue();
        }
        else {
            std::random_device rd;
            if constexpr (std::is_same_v<std::random_device::result_type,
                uint32_t>) {
                seed = rd();
                seed <<= 32;
                seed |= rd();
            }
            else if constexpr (std::is_same_v<std::random_device::result_type,
                uint64_t>) {
                seed = rd();
            }
            else {
                throw std::runtime_error("Unsupported random_device result type");
            }
        }

        benchmark(algorithm, dtype, n_left, n_inner, n_right, d, b, 
            layout, transpose_a, transpose_b, n_reps, seed);
    }
    catch (TCLAP::ArgException& e) {
        std::println(stderr, "{}: {} for arg {}", argv[0], e.error(), 
            e.argId());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}