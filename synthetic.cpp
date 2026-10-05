#include "util.hpp"
#include "matmul.hpp"
#include "tclap/CmdLine.h"
#include "cnpy/cnpy.h"
#include <fstream>
#include <functional>
#include <queue>
#include <flat_set>
#include <flat_map>
#include <print>

using compmatmul::DType;
using compmatmul::Algorithm;
using compmatmul::HashFunction;
using compmatmul::MemoryLayout;
using compmatmul::Transpose;
using compmatmul::index_t;
using compmatmul::to_string;
using compmatmul::Timer;
using compmatmul::MultiplyAddShiftHash;
using compmatmul::Row;
using array::random::RNG_t;

struct Elem {
    index_t i;
    index_t j;
    double value;
};

template<typename T>
std::pair<T*,T*> generate_covariance_matrices(index_t n, int k, double rho, uint64_t seed, std::vector<Elem>& ground_truth) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "T must be float or double");
    if (k > n || k <= 0) {
        throw std::invalid_argument("k must be positive and less than or equal to n");
    }
    ground_truth.clear();
    ground_truth.reserve(k);

    T* A = array::allocate<T>(n * n);
    T* B = array::allocate<T>(n * n);

    if (A == nullptr || B == nullptr) {
        throw std::runtime_error("Failed to allocate memory for matrices A and B");
    }

    RNG_t rng(seed);

    Timer timer;
    timer.push();
    rng.fill_normal(n * n, A, 0.0, 1.0 / std::sqrt(n));
    double time_A = timer.pop();
    std::println(stderr, "Generated matrix A of size {}x{} in {:f} seconds", n, n, time_A);

    #pragma omp parallel for
    for (index_t idx = 0; idx < n * n; ++idx) {
        if (std::isnan(A[idx]) || std::isinf(A[idx])) {
            throw std::runtime_error("Generated matrix A contains NaN or Inf");
        }
    }

    timer.push();
    rng.fill_normal(n * n, B, 0.0, 1.0 / std::sqrt(n));
    double time_B = timer.pop();
    std::println(stderr, "Generated matrix B of size {}x{} in {:f} seconds", n, n, time_B);

    #pragma omp parallel for
    for (index_t idx = 0; idx < n * n; ++idx) {
        if (std::isnan(B[idx]) || std::isinf(B[idx])) {
            throw std::runtime_error("Generated matrix B contains NaN or Inf");
        }
    }

    timer.push();
    std::vector<index_t> perm(n);
    for (index_t i = 0; i < n; ++i) {
        perm[i] = i;
    }
    rng.shuffle(n, &perm[0]);
    std::vector<index_t> planted_i(perm.begin(), perm.begin() + k);
    rng.shuffle(n, &perm[0]);
    std::vector<index_t> planted_j(perm.begin(), perm.begin() + k);
    std::vector<index_t> planted_idx(k);
    for (int l = 0; l < k; ++l) {
        planted_idx[l] = planted_j[l]*n + planted_i[l];
        ground_truth.push_back({planted_i[l], planted_j[l], 0.0});
    }
    double time_planted = timer.pop();
    std::println(stderr, "Selected {} planted indices in {:f} seconds", k, time_planted);

    double tau = std::sqrt(1 - rho * rho);
    timer.push();
    #pragma omp parallel for
    for (int l = 0; l < k; ++l) {
        index_t i = planted_i[l];
        index_t j = planted_j[l];
        double& v = ground_truth[l].value;
        for (index_t t = 0; t < n; ++t) {
            B[j * n + t] = tau * B[j * n + t] + rho * A[i + t * n ];
            v += A[i + t * n] * B[j * n + t];
        }
    }
    double time_add = timer.pop();
    std::println(stderr, "Added correlated components to B in {:f} seconds", time_add);

    return std::make_pair(A,B);
}

template<typename T>
std::pair<T*,T*> generate_diagonal_matrices(index_t n, int k, uint64_t seed, std::vector<Elem>& ground_truth) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "T must be float or double");
    if (k > n || k <= 0) {
        throw std::invalid_argument("k must be positive and less than or equal to n");
    }
    ground_truth.clear();
    ground_truth.reserve(k);
    T* A = array::allocate<T>(n * n);
    T* B = array::allocate<T>(n * n);
    if (A == nullptr || B == nullptr) {
        throw std::runtime_error("Failed to allocate memory for matrices A and B");
    }
    T* LU = array::allocate<T>(n * n);
    if (LU == nullptr) {
        throw std::runtime_error("Failed to allocate memory for matrix LU");
    }
    

    std::println(stderr, "Generating diagonal matrices of size {}x{} with {} planted elements", n, n, k);

    RNG_t rng(seed);

    Timer timer;

    std::println(stderr, "Generating random matrix A of size {}x{}", n, n);
    timer.push();
    rng.fill(n * n, A);
    double time_A = timer.pop();
    std::println(stderr, "Generated matrix A in {:f} seconds", time_A);

    std::println(stderr, "Generating random permuted diagonal matrix PD of size {}x{}", n, n);
    timer.push();
    std::vector<index_t> perm(n);
    for (index_t i = 0; i < n; ++i) {
        perm[i] = i;
    }
    rng.shuffle(n, &perm[0]);
    std::vector<index_t> planted(perm.begin(), perm.begin() + k);
    std::sort(planted.begin(), planted.end());
    rng.shuffle(n, &perm[0]);

    T zero_value = 0.0001;
    std::vector<T> values(k);
    rng.fill(k, &values[0]);
    for (index_t i = 0; i < k; ++i) {
        values[i] -= (values[i] < 0.5);
    }
    std::vector<T> diag(n, zero_value);
    for (index_t i = 0; i < k; ++i) {
        diag[planted[i]] = values[i];
    }

    for (index_t i = 0; i < n; ++i) {
        index_t j = perm[i];
        B[j * n + i] = diag[i]; // column-major layout due to lapack
        if (diag[i] != zero_value) {
            ground_truth.push_back({i, j, diag[i]});
        }
    }

    double time_B = timer.pop();
    std::println(stderr, "Generated matrix PD in {:f} seconds", time_B);

    std::println(stderr, "Creating a copy of A for LU factorization");
    timer.push();
    array::mov(n * n, LU, A);
    double time_LU_copy = timer.pop();
    std::println(stderr, "Copied matrix A to LU in {:f} seconds", time_LU_copy);

    #ifdef LIBARRAY_BLAS_ACCELERATE
    using lapack_int = __LAPACK_int;
    #endif
    lapack_int info = -1;
    std::vector<lapack_int> ipiv(n);
    lapack_int n_lapack = static_cast<lapack_int>(n);

    timer.push();
    if constexpr (std::is_same_v<T,double>) {
        dgetrf_(&n_lapack, &n_lapack, LU, &n_lapack, &ipiv[0], &info);
    }
    else if constexpr (std::is_same_v<T,float>) {
        sgetrf_(&n_lapack, &n_lapack, LU, &n_lapack, &ipiv[0], &info);
    }

    double time_LU_factorization = timer.pop();
    std::println(stderr, "LU factorization completed in {:f} seconds", time_LU_factorization);

    if (info == 0) {
        std::println(stderr, "LU factorization completed successfully");
    } 
    else if (info < 0) {
        throw std::runtime_error("LU factorization failed: illegal argument");
    } 
    else {
        throw std::runtime_error("LU factorization failed: matrix is singular");
    }

    timer.push();
    char trans = 'N';
    if constexpr (std::is_same_v<T,double>) {
        dgetrs_(&trans, &n_lapack, &n_lapack, LU, &n_lapack, &ipiv[0], B, &n_lapack, &info);
    }
    else if constexpr (std::is_same_v<T,float>) {
        sgetrs_(&trans, &n_lapack, &n_lapack, LU, &n_lapack, &ipiv[0], B, &n_lapack, &info);
    }
    double time_solve = timer.pop();
    std::println(stderr, "Solved linear system B = A^-1 PD in {:f} seconds", time_solve);

    if (info == 0) {
        std::println(stderr, "Solved linear system B = A^-1 PD successfully");
    } 
    else if (info < 0) {
        throw std::runtime_error("Solving linear system failed: illegal argument");
    } 
    else {
        throw std::runtime_error("Solving linear system failed: positive info (should not happen)");
    }

    array::deallocate(LU);

    return std::make_pair(A,B);
}

template<typename T>
void experiment(Algorithm algorithm, HashFunction hash,
    const T* A, const T* B, index_t n_left, index_t n_inner, index_t n_right,
    MemoryLayout layout, Transpose transpose_A, Transpose transpose_B,
    int d, int b, uint64_t seed, uint64_t data_seed, uint64_t rep_seed, int reps, 
    const std::vector<Elem>& ground_truth, double epsilon, double tau, double rho, int instance_num,
    const std::string& matrix_type) {

    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "T must be float or double");

    for (index_t idx = 0; idx < n_left * n_inner; ++idx) {
        if (std::isnan(A[idx]) || std::isinf(A[idx])) {
            throw std::runtime_error("Input matrix A contains NaN or Inf");
        }
    }

    for (index_t idx = 0; idx < n_inner * n_right; ++idx) {
        if (std::isnan(B[idx]) || std::isinf(B[idx])) {
            throw std::runtime_error("Input matrix B contains NaN or Inf");
        }
    }


    const std::vector<std::string> columns = {
        "timestamp", "hostname", "num_threads",
        "blas_backend", "fft_backend", "fwht_backend",
        "algorithm", "dtype", "hash", 
        "n_left", "n_inner", "n_right",
        "matrix_type", "k", 
        "d", "b", 
        "layout", "transpose_a", "transpose_b",
        "instance_num",
        "rep", 
        "seed",
        "data_seed", 
        "rep_seed",
        "epsilon",
        "tau", 
        "rho",
        "matmul_time",
        "tp", "fp", "fn",
        "precision", "recall", "f1",
        "tp_eps", "fp_eps", "fn_eps",
        "precision_eps", "recall_eps", "f1_eps"
    };

    DType dtype = std::is_same_v<T, float> ? DType::Float32 : DType::Float64;

    std::string timestamp = compmatmul::get_timestamp();
    std::println(stderr, "timestamp: {}", timestamp);

    std::string hostname = compmatmul::get_hostname();
    std::println(stderr, "hostname: {}", hostname);

    int num_threads;    
    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
        }
    }
    std::println(stderr, "omp threads: {}", num_threads);

    std::string blas_backend = compmatmul::get_blas_backend();
    std::string fft_backend = compmatmul::get_fft_backend();
    std::string fwht_backend = compmatmul::get_fwht_backend();
    std::println(stderr, "BLAS backend: {}", blas_backend);
    std::println(stderr, "FFT backend: {}", fft_backend);
    std::println(stderr, "FWHT backend: {}", fwht_backend);
    
    std::println(stderr, "algorithm: {}", to_string(algorithm));
    std::println(stderr, "data type: {}", to_string(dtype));
    std::println(stderr, "hash function: {}", to_string(hash));

    std::println(stderr, "dimensions: {} x {} x {}", n_left, n_inner, n_right);
    std::println(stderr, "d: {}", d);
    std::println(stderr, "b: {}", b);
    std::println(stderr, "layout: {}", to_string(layout));
    std::println(stderr, "transpose A: {}", to_string(transpose_A));
    std::println(stderr, "transpose B: {}", to_string(transpose_B));
    std::println(stderr, "instance num: {}", instance_num);

    std::println(stderr, "random seed: {}", seed);
    std::println(stderr, "data seed: {}", data_seed);
    std::println(stderr, "rep seed: {}", rep_seed);

    std::println(stderr, "epsilon: {}", epsilon);
    std::println(stderr, "tau: {}", tau);
    std::println(stderr, "rho: {}", rho);

    T* C = array::allocate<T>(n_left * n_right);

    if (C == nullptr) {
        throw std::runtime_error("Failed to allocate memory for output matrix C");
    }

    T* p = nullptr;
    T* scratch = nullptr;
    std::vector<MultiplyAddShiftHash::param_t*> ha_params(reps,nullptr);
    std::vector<MultiplyAddShiftHash::param_t*> sa_params(reps,nullptr);
    std::vector<MultiplyAddShiftHash::param_t*> hb_params(reps,nullptr);
    std::vector<MultiplyAddShiftHash::param_t*> sb_params(reps,nullptr);

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
        RNG_t rng(rep_seed);
        for (int r = 0; r < reps; ++r) {
            ha_params[r] = array::allocate<MultiplyAddShiftHash::param_t>(MultiplyAddShiftHash::NUM_PARAMS * d);
            sa_params[r] = array::allocate<MultiplyAddShiftHash::param_t>(MultiplyAddShiftHash::NUM_PARAMS * d);
            hb_params[r] = array::allocate<MultiplyAddShiftHash::param_t>(MultiplyAddShiftHash::NUM_PARAMS * d);
            sb_params[r] = array::allocate<MultiplyAddShiftHash::param_t>(MultiplyAddShiftHash::NUM_PARAMS * d);
            rng.fill(MultiplyAddShiftHash::NUM_PARAMS * d, ha_params[r]);
            rng.fill(MultiplyAddShiftHash::NUM_PARAMS * d, sa_params[r]);
            rng.fill(MultiplyAddShiftHash::NUM_PARAMS * d, hb_params[r]);
            rng.fill(MultiplyAddShiftHash::NUM_PARAMS * d, sb_params[r]);
        }
        p = array::allocate<T>(d * b);
        if (p == nullptr) {
            throw std::runtime_error("Failed to allocate memory for polynomial p");
        }
        #ifdef COMPMATMUL_ENABLE_FFT
        if (algorithm == Algorithm::FFT) {
            scratch = array::allocate<T>(compmatmul::matmul_compress_and_multiply_fft_scratch_size<T>(d, b));
        } 
        #endif // COMPMATMUL_ENABLE_FFT
        #ifdef COMPMATMUL_ENABLE_FWHT
        if (algorithm == Algorithm::FWHT) {
            scratch = array::allocate<T>(compmatmul::matmul_compress_and_multiply_fwht_scratch_size(d, b));
        }
        #endif // COMPMATMUL_ENABLE_FWHT
        if (scratch == nullptr) {
            throw std::runtime_error("Failed to allocate memory for scratch space");
        }
    }

    Timer timer;

    int k = ground_truth.size();

    std::println(stderr, "k: {}", k);

    std::flat_set<uint64_t> ground_truth_idx_set; // ground truth elements
    std::flat_map<uint64_t, T> ground_truth_val_set; // ground truth elements with values
    for (const auto& e : ground_truth) {
        uint64_t idx = static_cast<uint64_t>(e.i)  + static_cast<uint64_t>(e.j) * n_left;
        ground_truth_idx_set.insert(idx);
        ground_truth_val_set[idx] = e.value;
    }


    std::vector<int64_t> tps(num_threads);
    std::vector<int64_t> fps(num_threads);
    std::vector<int64_t> fns(num_threads);
    std::vector<int64_t> tp_epss(num_threads);
    std::vector<int64_t> fp_epss(num_threads);
    std::vector<int64_t> fn_epss(num_threads);
    for (int rep = 0; rep < reps; ++rep) {
        std::println(stderr, "rep: {}", rep);

        double matmul_time = 0.0;
        double top_k_time = 0.0;


        timer.push();

        matmul_static(algorithm, hash, dtype,
            n_left, n_inner, n_right,
            C, A, B,
            p, scratch,
            ha_params[rep], sa_params[rep], hb_params[rep], sb_params[rep],
            d, b, layout, transpose_A, transpose_B);

        double chunk_time = timer.pop();
        matmul_time += chunk_time;

        std::println(stderr, "matmul took {:f} seconds", chunk_time);

        for (index_t idx = 0; idx < n_left * n_right; ++idx) {
            if (std::isinf(C[idx])) {
                std::println(stderr, "C[{}] is Inf", idx);

                index_t i = idx % n_left;
                index_t j = idx / n_left;
                T v = 0.0;
                for (index_t k = 0; k < n_inner; ++k) {
                    v += A[i + k * n_left] * B[k + j * n_inner];
                }
                std::println(stderr, "C[{}] should be {}", idx, v);
                break;
            }
        }


        #pragma omp parallel for
        for (index_t idx = 0; idx < n_left * n_right; ++idx) {
            if (std::isnan(C[idx]) || std::isinf(C[idx])) {
                throw std::runtime_error("Output matrix C contains NaN or Inf");
            }
        }

        timer.push();
        #pragma omp parallel 
        {
            int thread_id = omp_get_thread_num();
            int64_t tp = 0;
            int64_t fp = 0;
            int64_t fn = 0;
            int64_t tp_eps = 0;
            int64_t fp_eps = 0;
            int64_t fn_eps = 0;
            #pragma omp for
            for (index_t idx = 0; idx < n_left * n_right; ++idx) {
                T v = C[idx];
                if (std::abs(v) >= tau) {
                    if (ground_truth_idx_set.contains(idx)) {
                        ++tp;
                        double gt_value = ground_truth_val_set[idx];
                        if (std::abs(v - gt_value) <= epsilon) {
                            ++tp_eps;
                        } 
                        else {
                            ++fn_eps;
                        }
                    }
                    else {
                        ++fp;
                        ++fp_eps;
                    }
                }
            }
            #pragma omp for
            for (const auto& e : ground_truth) {
                uint64_t idx = static_cast<uint64_t>(e.i)  + static_cast<uint64_t>(e.j) * n_left;
                if (std::abs(C[idx]) < tau) {
                    ++fn;
                    ++fn_eps;
                }
            }
            tps[thread_id] = tp;
            fps[thread_id] = fp;
            fns[thread_id] = fn;
            tp_epss[thread_id] = tp_eps;
            fp_epss[thread_id] = fp_eps;
            fn_epss[thread_id] = fn_eps;
        }
        int64_t tp = 0, fp = 0, fn = 0;
        int64_t tp_eps = 0, fp_eps = 0, fn_eps = 0;
        for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
            tp += tps[thread_id];
            fp += fps[thread_id];
            fn += fns[thread_id];
            tp_eps += tp_epss[thread_id];
            fp_eps += fp_epss[thread_id];
            fn_eps += fn_epss[thread_id];
        }
        top_k_time += timer.pop();

        std::println(stderr, "checking top results took {:f} seconds", top_k_time);
        std::println(stderr, "tp: {}, fp: {}, fn: {}", tp, fp, fn);
        std::println(stderr, "tp_eps: {}, fp_eps: {}, fn_eps: {}", tp_eps, fp_eps, fn_eps);




        double precision = (tp + fp) > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double recall = (tp + fn) > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
        double f1 = (tp+fp+fn) > 0 ? 2.0 * tp / (2.0 * tp + fp + fn) : 0.0;

        double precision_eps = (tp_eps + fp_eps) > 0 ? static_cast<double>(tp_eps) / (tp_eps + fp_eps) : 0.0;
        double recall_eps = (tp_eps + fn_eps) > 0 ? static_cast<double>(tp_eps) / (tp_eps + fn_eps) : 0.0;
        double f1_eps = (tp_eps+fp_eps+fn_eps) > 0 ? 2.0 * tp_eps / (2.0 * tp_eps + fp_eps + fn_eps) : 0.0;

        std::println(stderr, "precision: {}, recall: {}, f1: {}", precision, recall, f1);
        std::println(stderr, "precision_eps: {}, recall_eps: {}, f1_eps: {}", precision_eps, recall_eps, f1_eps);

        Row row(columns);
        row.set("timestamp", timestamp);
        row.set("hostname", hostname);
        row.set("num_threads", std::to_string(num_threads));
        row.set("blas_backend", blas_backend);
        row.set("fft_backend", fft_backend);
        row.set("fwht_backend", fwht_backend);
        row.set("algorithm", to_string(algorithm));
        row.set("dtype", to_string(dtype));
        row.set("hash", to_string(hash));
        row.set("n_left", std::to_string(n_left));
        row.set("n_inner", std::to_string(n_inner));
        row.set("n_right", std::to_string(n_right));
        row.set("matrix_type", matrix_type);
        row.set("k", std::to_string(k));
        row.set("d", std::to_string(d));
        row.set("b", std::to_string(b));
        row.set("layout", to_string(layout));
        row.set("transpose_a", to_string(transpose_A));
        row.set("transpose_b", to_string(transpose_B));
        row.set("instance_num", std::to_string(instance_num));
        row.set("rep", std::to_string(rep));
        row.set("seed", std::to_string(seed));
        row.set("data_seed", std::to_string(data_seed));
        row.set("rep_seed", std::to_string(rep_seed));
        row.set("epsilon", std::to_string(epsilon));
        row.set("tau", std::to_string(tau));
        row.set("rho", std::to_string(rho));
        row.set("matmul_time", std::to_string(matmul_time));
        row.set("tp", std::to_string(tp));
        row.set("fp", std::to_string(fp));
        row.set("fn", std::to_string(fn));
        row.set("precision", std::to_string(precision));
        row.set("recall", std::to_string(recall));
        row.set("f1", std::to_string(f1));
        row.set("tp_eps", std::to_string(tp_eps));
        row.set("fp_eps", std::to_string(fp_eps));
        row.set("fn_eps", std::to_string(fn_eps));
        row.set("precision_eps", std::to_string(precision_eps));
        row.set("recall_eps", std::to_string(recall_eps));
        row.set("f1_eps", std::to_string(f1_eps));

        if (rep == 0) {
            std::println(stdout, "{}", row.header());
        }
        std::println(stdout, "{}", row.to_string());
    }

    array::deallocate(C);
    array::deallocate(p);
    array::deallocate(scratch);
    for (int r = 0; r < reps; ++r) {
        array::deallocate(ha_params[r]);
        array::deallocate(sa_params[r]);
        array::deallocate(hb_params[r]);
        array::deallocate(sb_params[r]);
    }
}

int main(int argc, char* argv[]) {
    try {
        TCLAP::CmdLine cmd("Npy array multplication benchmarking tool for compmatmul", ' ', "0.1.0");

        // std::vector<std::string> memory_layouts { "row-major", "column-major" };
        // TCLAP::ValuesConstraint<std::string> memory_layout_constraint(memory_layouts);
        // TCLAP::ValueArg<std::string> memory_layout_arg("", "mem-layout", "Memory layout", false, "", &memory_layout_constraint);

        // TCLAP::SwitchArg transpose_a_arg("", "transpose-a", "Transpose A (left-hand side)");
        // TCLAP::SwitchArg transpose_b_arg("", "transpose-b", "Transpose B (right-hand side)");

        TCLAP::ValueArg<int> reps_arg("r", "reps", "Number of repetitions", false, 1, "int");

        TCLAP::ValueArg<int> d_arg("d", "d", "Number of sketches", false, -1, "int");
        TCLAP::ValueArg<int> b_arg("b", "b", "Length of polynomial", false, -1, "int");

        TCLAP::ValueArg<uint64_t> seed_arg("s", "seed", "Random seed", false, 0, "integer");
        TCLAP::ValueArg<double> epsilon_arg("e", "epsilon", "Epsilon for approximate top-k", false, 0.1, "double");
        TCLAP::ValueArg<double> tau_arg("t", "tau", "Tau for approximate top-k", false, 0.4, "double");
        TCLAP::ValueArg<double> rho_arg("p", "rho", "Rho for approximate covariance", false, 0.8, "double");

        std::vector<std::string> algorithms { "gemm", "fft", "fwht" };
        TCLAP::ValuesConstraint<std::string> algorithm_constraint(algorithms);
        TCLAP::UnlabeledValueArg<std::string> algorithm_arg("algorithm", "Algorithm", true, "", &algorithm_constraint);

        std::vector<std::string> dtypes { "float32", "float64" };
        TCLAP::ValuesConstraint<std::string> dtype_constraint(dtypes);
        TCLAP::UnlabeledValueArg<std::string> dtype_arg("dtype", "Data type", true, "", &dtype_constraint);

        TCLAP::UnlabeledValueArg<int> n_arg("n", "Number of rows and columns in A and B", true, 0, "int");

        std::vector<std::string> matrix_types { "diagonal", "covariance" };
        TCLAP::ValuesConstraint<std::string> matrix_type_constraint(matrix_types);
        TCLAP::UnlabeledValueArg<std::string> matrix_type_arg("matrix_type", "Matrix type", true, "", &matrix_type_constraint);

        TCLAP::UnlabeledValueArg<int> k_arg("k", "Number of k planted elements", true, 0, "int");
        TCLAP::UnlabeledValueArg<int> instance_num_arg("instance_num", "Instance number", true, 0, "int");

        cmd.add(reps_arg);
        cmd.add(d_arg);
        cmd.add(b_arg);
        cmd.add(seed_arg);
        // cmd.add(transpose_a_arg);
        // cmd.add(transpose_b_arg);
        cmd.add(epsilon_arg);
        cmd.add(tau_arg);
        cmd.add(rho_arg);
        cmd.add(algorithm_arg);
        cmd.add(dtype_arg);
        cmd.add(n_arg);
        cmd.add(matrix_type_arg);
        cmd.add(k_arg);
        cmd.add(instance_num_arg);
        cmd.parse(argc, argv);

        int reps = reps_arg.getValue();
        if (reps <= 0) {
            std::println(stderr, "Number of repetitions must be positive");
            return EXIT_FAILURE;
        }
        std::println(stderr, "Repetitions: {}", reps);

        Algorithm algorithm;
        std::string algorithm_str = algorithm_arg.getValue();
        if (algorithm_str == "gemm") {
            algorithm = Algorithm::GEMM;
        }
#ifdef COMPMATMUL_ENABLE_FFT
        else if (algorithm_str == "fft") {
            algorithm = Algorithm::FFT;
        }
#endif // COMPMATMUL_ENABLE_FFT
#ifdef COMPMATMUL_ENABLE_FWHT
        else if (algorithm_str == "fwht") {
            algorithm = Algorithm::FWHT;
        }
#endif // COMPMATMUL_ENABLE_FWHT
        else {
            std::println(stderr, "Unknown algorithm: {}", algorithm_str);
            return EXIT_FAILURE;
        }

        std::println(stderr, "Algorithm: {}", to_string(algorithm));


        int d = 0;
        int b = 0;
        if (algorithm != Algorithm::GEMM) {
            if (!d_arg.isSet()) {
                std::println(stderr, "Number of sketches (d) must be set for algorithm {}", to_string(algorithm));
                return EXIT_FAILURE;
            }
            d = d_arg.getValue();
            if (d <= 0) {
                std::println(stderr, "Number of sketches (d) must be positive");
                return EXIT_FAILURE;
            }
            if (!b_arg.isSet()) {
                std::println(stderr, "Length of polynomial (b) must be set for algorithm {}", to_string(algorithm));
                return EXIT_FAILURE;
            }
            b = b_arg.getValue();
            if (b <= 0 || (b & (b - 1)) != 0) {
                std::println(stderr, "Length of polynomial (b) must be a power of 2");
                return EXIT_FAILURE;
            }
        }

        std::println(stderr, "Number of sketches (d): {}", d);
        std::println(stderr, "Length of polynomial (b): {}", b);

        uint64_t seed;
        if (seed_arg.isSet()) {
            seed = seed_arg.getValue();
        }
        else {
            std::random_device rd;
            // should work if the return value is 32-bit or 64-bit
            seed = rd() & 0xFFFFFFFF;
            seed <<= 32;
            seed |= rd() & 0xFFFFFFFF;
        }
        std::println(stderr, "Random seed: {}", seed);

        Transpose transpose_a = Transpose::NoTranspose;
        Transpose transpose_b = Transpose::NoTranspose;

        std::println(stderr, "Transpose A: {}", to_string(transpose_a));
        std::println(stderr, "Transpose B: {}", to_string(transpose_b));

        double epsilon = epsilon_arg.getValue();
        if (epsilon < 0.0) {
            std::println(stderr, "Epsilon must be non-negative");
            return EXIT_FAILURE;
        }
        std::println(stderr, "Epsilon: {}", epsilon);

        double tau = tau_arg.getValue();
        if (tau < 0.0) {
            std::println(stderr, "Tau must be non-negative");
            return EXIT_FAILURE;
        }
        std::println(stderr, "Tau: {}", tau);

        double rho = rho_arg.getValue();
        if (rho < 0.0 || rho > 1.0) {
            std::println(stderr, "Rho must be in [0,1]");
            return EXIT_FAILURE;
        }
        std::println(stderr, "Rho: {}", rho);

        DType dtype;
        std::string dtype_str = dtype_arg.getValue();
        if (dtype_str == "float32") {
            dtype = DType::Float32;
        }
        else if (dtype_str == "float64") {
            dtype = DType::Float64;
        }
        else {
            std::println(stderr, "Unknown data type: {}", dtype_str);
            return EXIT_FAILURE;
        }

        std::println(stderr, "Data type: {}", to_string(dtype));

        index_t n = n_arg.getValue();
        if (n <= 0) {
            std::println(stderr, "Number of rows and columns must be positive");
            return EXIT_FAILURE;
        }

        index_t n_left, n_inner, n_right;
        n_left = n_inner = n_right = n;

        std::println(stderr, "Matrix dimensions: {}x{}x{}", n_left, n_inner, n_right);

        MemoryLayout layout = MemoryLayout::ColumnMajor; // default due to Lapack
        std::println(stderr, "Memory layout: {}", to_string(layout));

        int k = k_arg.getValue();
        if (k < 0) {
            std::println(stderr, "Number of planted elements (k) must be non-negative");
            return EXIT_FAILURE;
        }
        if (k > n) {
            std::println(stderr, "Number of planted elements (k) must be less than or equal to n");
            return EXIT_FAILURE;
        }
        std::println(stderr, "Number of planted elements (k): {}", k);

        int instance_num = instance_num_arg.getValue();
        if (instance_num < 0) {
            std::println(stderr, "Instance number must be non-negative");
            return EXIT_FAILURE;
        }
        std::println(stderr, "Instance number: {}", instance_num);


        RNG_t rng(seed_arg.isSet() ? std::optional<uint64_t>(seed_arg.getValue()) : std::nullopt);
        uint64_t data_seed = rng.next<uint64_t>();
        uint64_t experiment_seed = rng.next<uint64_t>();

        std::println(stderr, "Data seed: {}", data_seed);
        std::println(stderr, "Experiment seed: {}", experiment_seed);

        void* A = nullptr;
        void* B = nullptr;
        std::vector<Elem> ground_truth;
        std::string matrix_type = matrix_type_arg.getValue();
        if (matrix_type == "diagonal") {
            if (dtype == DType::Float32) {
                std::tie(A, B) = generate_diagonal_matrices<float>(n, k, data_seed, ground_truth);
            } 
            else if (dtype == DType::Float64) {
                std::tie(A, B) = generate_diagonal_matrices<double>(n, k, data_seed, ground_truth);
            } 
            else {
                std::println(stderr, "Unsupported dtype: {}", to_string(dtype));
                return EXIT_FAILURE;
            }
        }
        else if (matrix_type_arg.getValue() == "covariance") {
            if (dtype == DType::Float32) {
                std::tie(A, B) = generate_covariance_matrices<float>(n, k, rho, data_seed, ground_truth);
            } 
            else if (dtype == DType::Float64) {
                std::tie(A, B) = generate_covariance_matrices<double>(n, k, rho, data_seed, ground_truth);
            } 
            else {
                std::println(stderr, "Unsupported dtype: {}", to_string(dtype));
                return EXIT_FAILURE;
            }
        }
        else {
            std::println(stderr, "Unknown matrix type: {}", matrix_type_arg.getValue());
            return EXIT_FAILURE;
        }


        if (dtype == DType::Float32) {
            experiment<float>(algorithm, HashFunction::MultiplyAddShift,
                reinterpret_cast<float*>(A), reinterpret_cast<float*>(B),
                n_left, n_inner, n_right,
                layout, transpose_a, transpose_b,
                d, b, seed, data_seed, experiment_seed, reps,
                ground_truth, epsilon, tau, rho, instance_num, matrix_type);
        } 
        else if (dtype == DType::Float64) {
            experiment<double>(algorithm, HashFunction::MultiplyAddShift,
                reinterpret_cast<double*>(A), reinterpret_cast<double*>(B),
                n_left, n_inner, n_right,
                layout, transpose_a, transpose_b,
                d, b, seed, data_seed, experiment_seed, reps,
                ground_truth, epsilon, tau, rho, instance_num, matrix_type);
        } 
        else {
            std::println(stderr, "Unsupported dtype: {}", to_string(dtype));
            return EXIT_FAILURE;
        }

        array::deallocate(A);
        array::deallocate(B);

    }
    catch (TCLAP::ArgException &e) {
        std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}