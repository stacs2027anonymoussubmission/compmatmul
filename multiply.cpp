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
void experiment(Algorithm algorithm, HashFunction hash,
    const T* A, const T* B, index_t n_left, index_t n_inner, index_t n_right,
    MemoryLayout layout, Transpose transpose_A, Transpose transpose_B,
    // index_t chunk_size, 
    int d, int b, uint64_t seed, int reps, 
    std::vector<Elem>* ground_truth, double epsilon,
    bool use_abs) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "T must be float or double");

    const std::vector<std::string> columns = {
        "timestamp", "hostname", "num_threads",
        "blas_backend", "fft_backend", "fwht_backend",
        "algorithm", "dtype", "hash", "k",
        "n_left", "n_inner", "n_right",
        "d", "b", 
        "layout", "transpose_a", "transpose_b",
        "rep", 
        "seed", 
        "matmul_time",
        "top_k_time",
        "total_time",
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

    std::println(stderr, "random seed: {}", seed);

    // if (chunk_size == 0) {
    //     chunk_size = std::max(n_left,n_right);
    // }

    // T* C = array::allocate<T>(chunk_size * chunk_size);
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
        RNG_t rng(seed);
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

    int k = ground_truth ? ground_truth->size() : 0;

    // std::println(stderr, "chunk size: {}", chunk_size);

    std::flat_set<uint64_t> ground_truth_idx_set; // ground truth elements
    std::flat_map<uint64_t, T> ground_truth_val_set; // ground truth elements with values
    if (ground_truth) {
        for (const auto& e : *ground_truth) {
            uint64_t idx = static_cast<uint64_t>(e.i) * n_right + static_cast<uint64_t>(e.j);
            ground_truth_idx_set.insert(idx);
            ground_truth_val_set[idx] = e.value;
        }
    }


    for (int rep = 0; rep < reps; ++rep) {
        std::println(stderr, "rep: {}", rep);

        double matmul_time = 0.0;
        double top_k_time = 0.0;

        std::vector<std::priority_queue<Elem, std::vector<Elem>, std::function<bool(const Elem&, const Elem&)>>> pqs(num_threads,
            std::priority_queue<Elem, std::vector<Elem>, std::function<bool(const Elem&, const Elem&)>>([](const Elem& a, const Elem& b) { return a.value > b.value; }));

        // for (index_t i0 = 0; i0 < n_left; i0 += chunk_size) {
        //     index_t i1 = std::min(i0 + chunk_size, n_left);
        //     index_t n_left_chunk = i1 - i0;
        //     const T* A_chunk = A + i0 * n_inner;
        //     for (index_t j0 = 0; j0 < n_right; j0 += chunk_size) {
        //         index_t j1 = std::min(j0 + chunk_size, n_right);
        //         index_t n_right_chunk = j1 - j0;
        //         const T* B_chunk = B + j0 * n_inner;

        timer.push();
                // compmatmul::matmul(algorithm, hash, dtype,
                //     n_left_chunk, n_inner, n_right_chunk,
                //     C, A_chunk, B_chunk,
                //     d, b, seed,
                //     layout, transpose_A, transpose_B);
        matmul_static(algorithm, hash, dtype,
            n_left, n_inner, n_right,
            C, A, B,
            p, scratch,
            ha_params[rep], sa_params[rep], hb_params[rep], sb_params[rep],
            d, b, layout, transpose_A, transpose_B);
        double chunk_time = timer.pop();
        matmul_time += chunk_time;

        if (k < 1) {
            continue;
        }

        if (use_abs) {
            std::println(stderr, "Using absolute values of C for top-k computation");
            std::println(stderr, "Computing absolute values of C");
            timer.push();
            #pragma omp parallel for
            for (index_t idx = 0; idx < n_left * n_right; ++idx) {
                C[idx] = std::abs(C[idx]);
            }
            double abs_time = timer.pop();
            std::println(stderr, "Computing absolute values of C took {:f} seconds", abs_time);
        }

        timer.push();
        #pragma omp parallel 
        {
            int thread_id = omp_get_thread_num();
            auto& pq = pqs[thread_id];
            #pragma omp for
            for (index_t i = 0; i < n_left; ++i) {
                for (index_t j = 0; j < n_right; ++j) {
                    T v = C[i * n_right + j];
                    if (pq.size() < static_cast<size_t>(k)) {
                        pq.emplace(i, j, v);
                    } 
                    else if (v > pq.top().value) {
                        pq.pop();
                        pq.emplace(i, j, v);
                    }
                }
            }
        }
        double chunk_top_k_time = timer.pop();
        top_k_time += chunk_top_k_time;
        //     }
        // }

        timer.push();
        for (int thread_id = 1; thread_id < num_threads; ++thread_id) {
            auto& pq = pqs[thread_id];
            while (!pq.empty()) {
                pqs[0].push(pq.top());
                pqs[0].pop();
                pq.pop();
            }
        }
        double merge_time = timer.pop();
        top_k_time += merge_time;

        int tp = 0, fp = 0, fn = 0;
        int tp_eps = 0, fp_eps = 0, fn_eps = 0;

        while(!pqs[0].empty()) {
            const auto& e = pqs[0].top();
            uint64_t idx = static_cast<uint64_t>(e.i) * n_right + static_cast<uint64_t>(e.j);
            if (ground_truth_idx_set.contains(idx)) {
                ++tp;
                double gt_value = ground_truth_val_set[idx];
                if (std::abs((e.value - gt_value)/gt_value) <= epsilon) {
                    ++tp_eps;
                }
                else {
                    ++fp_eps;
                }
            } 
            else {
                ++fp;
                ++fp_eps;
            }
            pqs[0].pop();
        }
        // false negatives are those ground truth elements that were not found in the top-k
        fn += ground_truth_idx_set.size() - tp;
        fn_eps += ground_truth_idx_set.size() - tp_eps;
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

        double total_time = matmul_time + top_k_time;
        std::println(stderr, "time to multiply: {:f} seconds", matmul_time);
        std::println(stderr, "time to compute top-k: {:f} seconds", top_k_time);
        std::println(stderr, "total time: {:f} seconds", total_time);

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
        row.set("k", std::to_string(k));
        row.set("n_left", std::to_string(n_left));
        row.set("n_inner", std::to_string(n_inner));
        row.set("n_right", std::to_string(n_right));
        row.set("d", std::to_string(d));
        row.set("b", std::to_string(b));
        row.set("layout", to_string(layout));
        row.set("transpose_a", to_string(transpose_A));
        row.set("transpose_b", to_string(transpose_B));
        row.set("rep", std::to_string(rep));
        row.set("seed", std::to_string(seed));
        row.set("matmul_time", std::to_string(matmul_time));
        row.set("top_k_time", std::to_string(top_k_time));
        row.set("total_time", std::to_string(total_time));
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

        // timer.push();
        // compmatmul::matmul(algorithm, hash, dtype,
        //     n_left, n_inner, n_right,
        //     C, A, B,
        //     d, b, seed,
        //     layout, transpose_A, transpose_B);
        // double matmul_time = timer.pop();
        // std::println(stderr, "time to multiply: {:f} seconds", matmul_time);

        // if (ground_truth) {
        //     timer.push();
        //     std::vector<Elem> top_k = compmatmul::top_k(C, n_left, n_right, *ground_truth);
        //     double top_k_time = timer.pop();
        //     std::println(stderr, "time to compute top-k: {:f} seconds", top_k_time);
        // }
    }

    //   inline void matmul(Algorithm algorithm, HashFunction hash,
    //     DType dtype,
    //     index_t n_left, index_t n_inner, index_t n_right,
    //     void* __restrict__ C, const void* __restrict__ A, const void* __restrict__ B,
    //     std::optional<int> d = std::nullopt, 
    //     std::optional<int> b = std::nullopt,
    //     std::optional<uint64_t> seed = std::nullopt,
    //     MemoryLayout layout = MemoryLayout::RowMajor,
    //     Transpose transpose_A = Transpose::NoTranspose,
    //     Transpose transpose_B = Transpose::NoTranspose)

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

        // TCLAP::ValueArg<index_t> chunk_size_arg("c", "chunk-size", "Chunk size for processing", false, 0, "int");

        TCLAP::SwitchArg abs_arg("a", "abs", "Use absolute values of C for top-k");

        TCLAP::ValueArg<uint64_t> seed_arg("s", "seed", "Random seed", false, 0, "integer");
        TCLAP::ValueArg<std::string> ground_truth_file_arg("g", "ground-truth", "Ground truth file (optional)", false, "", "string");
        TCLAP::ValueArg<double> epsilon_arg("e", "epsilon", "Epsilon for approximate top-k", false, 0.1, "double");

        std::vector<std::string> algorithms { "gemm", "fft", "fwht" };
        TCLAP::ValuesConstraint<std::string> algorithm_constraint(algorithms);
        TCLAP::UnlabeledValueArg<std::string> algorithm_arg("algorithm", "Algorithm", true, "", &algorithm_constraint);

        TCLAP::UnlabeledValueArg<std::string> input_a_arg("A", "Left-hand input file", true, "", "string");
        TCLAP::UnlabeledValueArg<std::string> input_b_arg("B", "Right-hand input file", true, "", "string");
        cmd.add(reps_arg);
        cmd.add(d_arg);
        cmd.add(b_arg);
        // cmd.add(chunk_size_arg);
        cmd.add(seed_arg);
        cmd.add(ground_truth_file_arg);
        cmd.add(abs_arg);
        cmd.add(epsilon_arg);
        cmd.add(algorithm_arg);
        cmd.add(input_a_arg);
        cmd.add(input_b_arg);
        cmd.parse(argc, argv);

        std::string input_a_fn = input_a_arg.getValue();
        std::println(stderr, "Loading file {}...", input_a_fn);
        cnpy::NpyArray arr_a = cnpy::npy_load(input_a_fn);

        // dtype and shape is inferred from the NpyArray object
        DType dtype;
        if (arr_a.word_size == 4) {
            dtype = DType::Float32;
        } 
        else if (arr_a.word_size == 8) {
            dtype = DType::Float64;
        } 
        else {
            std::println(stderr, "Unsupported dtype with word size {} in file {}", arr_a.word_size, input_a_fn);
            return EXIT_FAILURE;
        }

        if (arr_a.shape.size() != 2) {
            std::println(stderr, "Input file {} is not a 2D array", input_a_fn);
            return EXIT_FAILURE;
        }

        index_t n_left = arr_a.shape[0];
        index_t n_inner = arr_a.shape[1];

        Transpose transpose_a = Transpose::NoTranspose;
        if (arr_a.fortran_order) {
            transpose_a = Transpose::Transpose;
        }

        std::println(stderr, "A is {}x{} with dtype {} and transpose {}", n_left, n_inner, to_string(dtype), to_string(transpose_a));

        std::string input_b_fn = input_b_arg.getValue();
        std::println(stderr, "Loading file {}...", input_b_fn);
        cnpy::NpyArray arr_b = cnpy::npy_load(input_b_fn);

        if (arr_b.shape.size() != 2) {
            std::println(stderr, "Input file {} is not a 2D array", input_b_fn);
            return EXIT_FAILURE;
        }

        if (arr_b.shape[0] != static_cast<size_t>(n_inner)) {
            std::println(stderr, "Input file {} has incompatible shape with A: {}x{} vs {}x{}", input_b_fn, n_left, n_inner, arr_b.shape[0], arr_b.shape[1]);
            return EXIT_FAILURE;
        }

        if (arr_b.word_size != arr_a.word_size) {
            std::println(stderr, "Input file {} has incompatible dtype with A (expected {})", input_b_fn, to_string(dtype));
            return EXIT_FAILURE;
        }
        Transpose transpose_b = Transpose::NoTranspose;
        if (arr_b.fortran_order) {
            transpose_b = Transpose::Transpose;
        }

        index_t n_right = arr_b.shape[1];
        std::println(stderr, "B is {}x{} with dtype {} and transpose {}", n_inner, n_right, to_string(dtype), to_string(transpose_b));

        MemoryLayout layout = MemoryLayout::RowMajor; // default

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

        std::unique_ptr<std::vector<Elem>> ground_truth = nullptr;
        if (ground_truth_file_arg.isSet()) {
            std::string ground_truth_file = ground_truth_file_arg.getValue();
            std::println(stderr, "Loading ground truth file {}...", ground_truth_file);

            ground_truth = std::make_unique<std::vector<Elem>>();

            std::ifstream gt_file(ground_truth_file);
            std::string line;
            while (std::getline(gt_file, line)) {
                // std::println(stderr, "{}", line);
                std::istringstream iss(line);
                index_t i, j;
                double value;
                char comma;
                if (!(iss >> value >> comma >> i >> comma >> j)) {
                    std::println(stderr, "Error parsing ground truth file {}: {}", ground_truth_file, line);
                    return EXIT_FAILURE;
                }
                ground_truth->push_back({i, j, value});
            }
        }

        double epsilon = epsilon_arg.getValue();

        if (epsilon < 0.0) {
            std::println(stderr, "Epsilon must be non-negative");
            return EXIT_FAILURE;
        }

        std::println(stderr, "Epsilon: {}", epsilon);

        bool use_abs = abs_arg.getValue();
        std::println(stderr, "Use absolute values of C for top-k: {}", use_abs);

        // index_t chunk_size = 0;
        // if (chunk_size_arg.isSet()) {
        //     chunk_size = chunk_size_arg.getValue();
        //     if (chunk_size < 0) {
        //         std::println(stderr, "Chunk size must be non-negative");
        //         return EXIT_FAILURE;
        //     }
        // }
        // std::println(stderr, "Chunk size: {}", chunk_size);

        // if (layout != MemoryLayout::RowMajor || transpose_a != Transpose::NoTranspose || transpose_b != Transpose::Transpose) {
        //     std::println(stderr, "Currently only row-major + no-transpose for A + transpose for B is supported");
        //     return EXIT_FAILURE;
        // }

        // k is inferred from ground truth

        if (dtype == DType::Float32) {
            experiment<float>(algorithm, HashFunction::MultiplyAddShift,
                arr_a.data<float>(), arr_b.data<float>(),
                n_left, n_inner, n_right,
                layout, transpose_a, transpose_b,
                // chunk_size, 
                d, b, seed, reps,
                ground_truth.get(), epsilon, use_abs);
        } 
        else if (dtype == DType::Float64) {
            experiment<double>(algorithm, HashFunction::MultiplyAddShift,
                arr_a.data<double>(), arr_b.data<double>(),
                n_left, n_inner, n_right,
                layout, transpose_a, transpose_b,
                // chunk_size, 
                d, b, seed, reps, 
                ground_truth.get(), epsilon, use_abs);
        } 
        else {
            std::println(stderr, "Unsupported dtype: {}", to_string(dtype));
            return EXIT_FAILURE;
        }
    } 
    catch (TCLAP::ArgException &e) {
        std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}