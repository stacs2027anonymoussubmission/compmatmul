#include "matmul_fwht.hpp"
#include "matmul_fft.hpp"
#include "matmul.hpp"
#include "libarray/array.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using compmatmul::Algorithm;
using compmatmul::DType;
using compmatmul::MemoryLayout;
using compmatmul::Transpose;
using compmatmul::HashFunction;
#ifdef COMPMATMUL_ENABLE_FFT
using compmatmul::matmul_compress_and_multiply_fft_scratch_size;
#endif // COMPMATMUL_ENABLE_FFT
#ifdef COMPMATMUL_ENABLE_FWHT
using compmatmul::matmul_compress_and_multiply_fwht_scratch_size;
#endif // COMPMATMUL_ENABLE_FWHT
using compmatmul::MultiplyAddShiftHash;
using array::index_t;

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](char c) { return std::tolower(c); });
    return s;
}

static py::array matmul_wrapper(
    Algorithm algorithm,
    const py::array& A, const py::array& B,
    std::optional<std::string> hash_name,
    std::optional<int> d,
    std::optional<int> b,
    std::optional<uint64_t> seed,
    std::optional<std::string> layout_str = std::nullopt,
    std::optional<std::string> transpose_A_str = std::nullopt,
    std::optional<std::string> transpose_B_str = std::nullopt
    ) {
    if (A.ndim() != 2 || B.ndim() != 2) { 
        throw py::value_error("Both A and B must be 2D arrays");
    }
    if (!A.dtype().is(B.dtype())) {
        throw py::type_error("A and B must have the same dtype");
    }
    DType dtype;
    if (A.dtype().is(py::dtype::of<double>())) {
        dtype = DType::Float64;
    }
    else if (A.dtype().is(py::dtype::of<float>())) {
        dtype = DType::Float32;
    }
    else {
        throw py::type_error("Unsupported dtype. Only float32 and float64 are supported.");
    }

    if (!(A.flags() & py::array::c_style) && !(A.flags() & py::array::f_style)) {
        throw py::value_error("A must be contiguous (C or F)");
    }
    if (!(B.flags() & py::array::c_style) && !(B.flags() & py::array::f_style)) {
        throw py::value_error("B must be contiguous (C or F)");
    }   

    MemoryLayout layout = MemoryLayout::RowMajor; // default layout

    if (layout_str.has_value()) {
        auto s = to_lower(layout_str.value());
        if (s == "rowmajor" || s == "c" || s == "c_contiguous") {
            layout = MemoryLayout::RowMajor;
        }
        else if (s == "columnmajor" || s == "f" || s == "f_contiguous") {
            layout = MemoryLayout::ColumnMajor;
        }
        else {
            throw py::value_error("Unsupported memory layout. Only "
                "'RowMajor'/'C' and 'ColumnMajor'/'F' are supported.");
        }
    }
    else {
        // infer from A and B
        if ((A.flags() & py::array::c_style) && 
            (B.flags() & py::array::c_style)) {
            layout = MemoryLayout::RowMajor;
        }
        else if ((A.flags() & py::array::f_style) && 
            (B.flags() & py::array::f_style)) {
            layout = MemoryLayout::ColumnMajor;
        }
        else if ((A.flags() & py::array::c_style) && 
            (B.flags() & py::array::f_style)) {
            layout = MemoryLayout::RowMajor;
        }
        else if ((A.flags() & py::array::f_style) && 
            (B.flags() & py::array::c_style)) {
            layout = MemoryLayout::RowMajor;
        }
        else {
            throw py::value_error("Cannot infer memory layout from A and B. "
                "Please specify explicitly. The arrays must be contigous.");
        }
    }

    Transpose transpose_A = Transpose::NoTranspose; // default transpose for A
    Transpose transpose_B = Transpose::NoTranspose; // default transpose for B

    auto parse_transpose = [](const std::string& t) -> Transpose {
        auto s = to_lower(t);
        if (s == "no" || s == "notranspose" || s == "n") {
            return Transpose::NoTranspose;
        }
        else if (s == "yes" || s == "transpose" || s == "t") {
            return Transpose::Transpose;
        }
        else {
            throw py::value_error("Unsupported transpose option. Only "
                "'NoTranspose'/'No'/'N' and 'Transpose'/'Yes'/'T' are "
                "supported.");
        }
    };

    if (transpose_A_str.has_value()) {
        transpose_A = parse_transpose(transpose_A_str.value());
    }
    if (transpose_B_str.has_value()) {
        transpose_B = parse_transpose(transpose_B_str.value());
    }

    HashFunction hash = HashFunction::MultiplyAddShift; // default hash function
    if (hash_name.has_value()) {
        if (hash_name.value() == "MultiplyAddShift") {
            hash = HashFunction::MultiplyAddShift;
        }
        else {
            throw py::value_error("Unsupported hash function. Only 'MultiplyAddShift' is supported.");
        }
    }

    if (algorithm != Algorithm::GEMM && (!d.has_value() || !b.has_value())) {
        throw py::value_error("Parameters d and b must be provided for non-GEMM algorithms");
    }

    if (A.ndim() != 2) {
        throw py::value_error("A must be a 2D array");
    }
    if (B.ndim() != 2) {
        throw py::value_error("B must be a 2D array");
    }

    index_t n_left, n_inner, n_right;
    if (transpose_A == Transpose::NoTranspose) {
        n_left = A.shape(0);
        n_inner = A.shape(1);
    } 
    else {
        n_left = A.shape(1);
        n_inner = A.shape(0);
    }
    if (transpose_B == Transpose::NoTranspose) {
        n_right = B.shape(1);
    } 
    else {
        n_right = B.shape(0);
    }

    if ((transpose_B == Transpose::NoTranspose && n_inner != B.shape(0)) ||
        (transpose_B == Transpose::Transpose && n_inner != B.shape(1))) {
        throw py::value_error("Inner dimensions of A and B must match for matrix multiplication");
    }



    void* C_data = nullptr;

    if (dtype == DType::Float32) {
        C_data = array::allocate<float>(n_left * n_right);
    }
    else if (dtype == DType::Float64) {
        C_data = array::allocate<double>(n_left * n_right);
    }
    else {
        throw py::type_error("Unsupported dtype. Only float32 and float64 are supported.");
    }

    py::capsule owner(C_data, [](void* p) {
        array::deallocate(p);
    });

    index_t dt_size = (dtype == DType::Float32) ? sizeof(float) : sizeof(double);
    py::array C = py::array(A.dtype(), {n_left, n_right}, 
        {layout == MemoryLayout::RowMajor ? dt_size * n_right : dt_size, 
         layout == MemoryLayout::RowMajor ? dt_size : dt_size * n_left}, C_data, owner);

    matmul(algorithm, hash, dtype, n_left, n_inner, n_right,
        C.mutable_data(), A.data(), B.data(),
        d, b, seed, layout, transpose_A, transpose_B);

    return C;
}

static py::array matmul_gemm(const py::array& A, const py::array& B,
    std::optional<std::string> layout = std::nullopt,
    std::optional<std::string> transpose_A = std::nullopt,
    std::optional<std::string> transpose_B = std::nullopt
) {
    return matmul_wrapper(Algorithm::GEMM, A, B, std::nullopt, std::nullopt, 
        std::nullopt, std::nullopt, layout, transpose_A, transpose_B);
}



#ifdef COMPMATMUL_ENABLE_FFT
static py::array matmul_fft(const py::array& A, const py::array& B, 
    int d, int b, 
    std::optional<std::string> hash_name = std::nullopt, 
    std::optional<uint64_t> seed = std::nullopt,
    std::optional<std::string> layout = std::nullopt,
    std::optional<std::string> transpose_A = std::nullopt,
    std::optional<std::string> transpose_B = std::nullopt
) {
    return matmul_wrapper(Algorithm::FFT, A, B, hash_name, d, b, seed, 
        layout, transpose_A, transpose_B);
}
#endif // COMPMATMUL_ENABLE_FFT



#ifdef COMPMATMUL_ENABLE_FWHT
static py::array matmul_fwht(const py::array& A, const py::array& B, 
    int d, int b, 
    std::optional<std::string> hash_name = std::nullopt, 
    std::optional<uint64_t> seed = std::nullopt,
    std::optional<std::string> layout = std::nullopt,
    std::optional<std::string> transpose_A = std::nullopt,
    std::optional<std::string> transpose_B = std::nullopt
) {
    return matmul_wrapper(Algorithm::FWHT, A, B, hash_name, d, b, seed, 
        layout, transpose_A, transpose_B);
}
#endif // COMPMATMUL_ENABLE_FWHT



/**
 * This is a convenience class that stores data for a GEMM multiplication so
 * that arrays need not be reallocated.
 */
template<Algorithm algorithm>
class MatmulObject {
public:
    MatmulObject(index_t n_left, index_t n_inner, index_t n_right, py::object dtype_obj, 
        std::optional<std::string> layout_str = std::nullopt) :
        MatmulObject(n_left, n_inner, n_right, dtype_obj, -1, -1, std::nullopt, std::nullopt, layout_str) 
        {
    }



    MatmulObject(index_t n_left, index_t n_inner, index_t n_right, 
        py::object dtype_obj, int d, int b, 
        std::optional<uint64_t> seed_opt = std::nullopt,
        std::optional<std::string> hash_name = std::nullopt,
        std::optional<std::string> layout_str = std::nullopt) :
        n_left(n_left), n_inner(n_inner), n_right(n_right),
        d(d), b(b) {

       if (layout_str.has_value()) {
            auto s = to_lower(layout_str.value());
            if (s == "rowmajor" || s == "c" || s == "c_contiguous") {
                this->layout = MemoryLayout::RowMajor;
            }
            else if (s == "columnmajor" || s == "f" || s == "f_contiguous") {
                this->layout = MemoryLayout::ColumnMajor;
            }
            else {
                throw py::value_error("Unsupported memory layout. Only "
                    "'RowMajor'/'C' and 'ColumnMajor'/'F' are supported.");
            }
        }


        py::dtype dt = py::dtype::from_args(dtype_obj);
        if (dt.is(py::dtype::of<double>())) {
            dtype = DType::Float64;
            C_data = array::allocate<double>(n_left * n_right);
        }
        else if (dt.is(py::dtype::of<float>())) {
            dtype = DType::Float32;
            C_data = array::allocate<float>(n_left * n_right);
        }
        else {
            throw py::type_error("Unsupported dtype. Only float32 and float64 are supported.");
        }

        index_t dt_size = (dtype == DType::Float32) ? sizeof(float) : sizeof(double);
        C = py::array(dt, {n_left, n_right},
            {layout == MemoryLayout::RowMajor ? dt_size * n_right : dt_size,
             layout == MemoryLayout::RowMajor ? dt_size : dt_size * n_left}, C_data);


        if (hash_name.has_value()) {
            std::string s = to_lower(hash_name.value());
            if (s == "multiplyaddshift") {
                hash = HashFunction::MultiplyAddShift;
            }
            else {
                throw py::value_error("Unsupported hash function. Only 'MultiplyAddShift' is supported.");
            }
        }
        if (algorithm != Algorithm::GEMM && d < 1) {
            throw py::value_error("Parameter d must be >= 1");
        }
        if (algorithm != Algorithm::GEMM && ((b <= 1) || ((b & (b - 1)) != 0))) { 
            throw py::value_error("Parameter b must be > 1 and a power of two");
        }

        if (hash == HashFunction::MultiplyAddShift) {
            ha_params = array::allocate<MultiplyAddShiftHash::param_t>(MultiplyAddShiftHash::NUM_PARAMS * d);
            sa_params = array::allocate<MultiplyAddShiftHash::param_t>(MultiplyAddShiftHash::NUM_PARAMS * d);
            hb_params = array::allocate<MultiplyAddShiftHash::param_t>(MultiplyAddShiftHash::NUM_PARAMS * d);
            sb_params = array::allocate<MultiplyAddShiftHash::param_t>(MultiplyAddShiftHash::NUM_PARAMS * d);
        }
        else {
            throw py::value_error("Unsupported hash function. Only 'MultiplyAddShift' is supported.");
        }

#if defined(COMPMATMUL_ENABLE_FFT) || defined(COMPMATMUL_ENABLE_FWHT)
        if constexpr (algorithm != Algorithm::GEMM) {
           if (dtype == DType::Float32) {
                    p = array::allocate<float>(d * b);
                }
                else if (dtype == DType::Float64) {
                    p = array::allocate<double>(d * b);
                }
        }
#endif // defined(COMPMATMUL_ENABLE_FFT) || defined(COMPMATMUL_ENABLE_FWHT)

#ifdef COMPMATMUL_ENABLE_FFT
        if constexpr (algorithm == Algorithm::FFT) {
            if (dtype == DType::Float32) {
                scratch = array::allocate<std::complex<float>>(
                    matmul_compress_and_multiply_fft_scratch_size<float>(d, b)
                );
            }
            else if (dtype == DType::Float64) {
                scratch = array::allocate<std::complex<double>>(
                    matmul_compress_and_multiply_fft_scratch_size<double>(d, b)
                );
            }
        }
        else 
#endif // defined(COMPMATMUL_ENABLE_FFT)
#ifdef COMPMATMUL_ENABLE_FWHT
        if constexpr (algorithm == Algorithm::FWHT) {
            if (dtype == DType::Float32) {
                scratch = array::allocate<float>(
                    matmul_compress_and_multiply_fwht_scratch_size(d, b)
                );
            }
            else if (dtype == DType::Float64) {
                scratch = array::allocate<double>(
                    matmul_compress_and_multiply_fwht_scratch_size(d, b)
                );
            }
        }
        else
#endif // defined(COMPMATMUL_ENABLE_FWHT)
        {
            scratch = nullptr;
        }

        uint64_t seed;
        if (seed_opt.has_value()) {
            seed = seed_opt.value();
        }
        else {
            std::random_device rd;
            seed = rd();
        }

        array::random::RNG_t rng(seed);

        if (hash == HashFunction::MultiplyAddShift) {
            MultiplyAddShiftHash::draw(d, 
                reinterpret_cast<MultiplyAddShiftHash::param_t*>(ha_params), 
                rng.next<uint64_t>());
            MultiplyAddShiftHash::draw(d, 
                reinterpret_cast<MultiplyAddShiftHash::param_t*>(sa_params), 
                rng.next<uint64_t>());
            MultiplyAddShiftHash::draw(d, 
                reinterpret_cast<MultiplyAddShiftHash::param_t*>(hb_params), 
                rng.next<uint64_t>());
            MultiplyAddShiftHash::draw(d, 
                reinterpret_cast<MultiplyAddShiftHash::param_t*>(sb_params), 
                rng.next<uint64_t>());
        }
        else {
            throw py::value_error("Unsupported hash function. Only 'MultiplyAddShift' is supported.");
        }
    }


    ~MatmulObject() {
        array::deallocate(C_data);
        array::deallocate(ha_params);
        array::deallocate(sa_params);
        array::deallocate(hb_params);
        array::deallocate(sb_params);
        array::deallocate(scratch);
        array::deallocate(p);
    }

    MatmulObject(const MatmulObject&) = delete;
    MatmulObject& operator=(const MatmulObject&) = delete;
    MatmulObject(MatmulObject&&) = delete;
    MatmulObject& operator=(MatmulObject&&) = delete;

    void multiply(const py::array& A, const py::array& B,
        std::optional<std::string> transpose_A_str = std::nullopt,
        std::optional<std::string> transpose_B_str = std::nullopt) {

        if (A.ndim() != 2 || B.ndim() != 2) {
            throw py::value_error("Both A and B must be 2D arrays");
        }

        if (!A.dtype().is(C.dtype())) {
            throw py::type_error("A has invalid dtype");
        }

        if (!B.dtype().is(C.dtype())) {
            throw py::type_error("B has invalid dtype");
        }

        Transpose transpose_A = Transpose::NoTranspose; // default transpose for A
        Transpose transpose_B = Transpose::NoTranspose; // default transpose for B

        if (transpose_A_str.has_value()) {
            auto s = to_lower(transpose_A_str.value());
            if (s == "no" || s == "notranspose" || s == "n") {
                transpose_A = Transpose::NoTranspose;
            }
            else if (s == "yes" || s == "transpose" || s == "t") {
                transpose_A = Transpose::Transpose;
            }
            else {
                throw py::value_error("Unsupported transpose option for A. Only "
                    "'NoTranspose'/'No'/'N' and 'Transpose'/'Yes'/'T' are supported.");
            }
        }
        if (transpose_B_str.has_value()) {
            auto s = to_lower(transpose_B_str.value());
            if (s == "no" || s == "notranspose" || s == "n") {
                transpose_B = Transpose::NoTranspose;
            }
            else if (s == "yes" || s == "transpose" || s == "t") {
                transpose_B = Transpose::Transpose;
            }
            else {
                throw py::value_error("Unsupported transpose option for B. Only "
                    "'NoTranspose'/'No'/'N' and 'Transpose'/'Yes'/'T' are supported.");
            }
        }

        if ((transpose_A == Transpose::NoTranspose && n_left != A.shape(0)) ||
            (transpose_A == Transpose::Transpose && n_left != A.shape(1))) {
            throw py::value_error("A has invalid shape");
        }
        if ((transpose_B == Transpose::NoTranspose && n_right != B.shape(1)) ||
            (transpose_B == Transpose::Transpose && n_right != B.shape(0))) {
            throw py::value_error("B has invalid shape");
        }

        if (layout == MemoryLayout::RowMajor) {
            if (!(A.flags() & py::array::c_style)) {
                throw py::value_error("A must be C-contiguous");
            }
            if (!(B.flags() & py::array::c_style)) {
                throw py::value_error("B must be C-contiguous");
            }
        }
        else if (layout == MemoryLayout::ColumnMajor) {
            if (!(A.flags() & py::array::f_style)) {
                throw py::value_error("A must be F-contiguous");
            }
            if (!(B.flags() & py::array::f_style)) {
                throw py::value_error("B must be F-contiguous");
            }
        }

        matmul_static(algorithm, hash, dtype, n_left, n_inner, n_right,
                C.mutable_data(), A.data(), B.data(), p,
                scratch, ha_params, sa_params, hb_params, sb_params,
                d, b, 
                layout, transpose_A, transpose_B);
    }
    // void compute_cache() {
    //     auto cache = cache_.mutable_unchecked<2>();
    //     for (std::size_t i = 0; i < rows_; ++i) {
    //         for (std::size_t j = 0; j < cols_; ++j) {
    //             cache(i, j) = static_cast<double>(i + j);
    //         }
    //     }
    // }

    // py::array_t<double> cache() const {
    //     return cache_;
    // }

    // std::size_t rows() const {
    //     return rows_;
    // }

    // std::size_t cols() const {
    //     return cols_;
    // }

    py::array get_C() const {
        return C;
    }

private:
    index_t n_left;
    index_t n_inner;
    index_t n_right;
    py::array C;
    DType dtype;
    MemoryLayout layout = MemoryLayout::RowMajor;
    void* C_data = nullptr;
    HashFunction hash = HashFunction::MultiplyAddShift; // default hash function
    int d = -1;
    int b = -1;
    void* ha_params = nullptr;
    void* sa_params = nullptr;
    void* hb_params = nullptr;
    void* sb_params = nullptr;
    void* scratch = nullptr;
    void* p = nullptr;
};

PYBIND11_MODULE(compmatmul, m) {
    m.doc() = "compmatmul module";
    py::class_<MatmulObject<Algorithm::GEMM>>(m, "MatmulGEMM")
        .def(py::init<index_t, index_t, index_t, py::object, 
            std::optional<std::string>>(), 
            py::arg("n_left"), py::arg("n_inner"), py::arg("n_right"), 
            py::arg("dtype"), py::arg("layout") = std::nullopt)
        .def_property_readonly("C", &MatmulObject<Algorithm::GEMM>::get_C)
        .def("multiply", &MatmulObject<Algorithm::GEMM>::multiply, 
            py::arg("A"), py::arg("B"), 
            py::arg("transpose_A") = std::nullopt, 
            py::arg("transpose_B") = std::nullopt);
#ifdef COMPMATMUL_ENABLE_FFT
    py::class_<MatmulObject<Algorithm::FFT>>(m, "MatmulFFT")
        .def(py::init<index_t, index_t, index_t, py::object, int, int, 
            std::optional<uint64_t>, 
            std::optional<std::string>, std::optional<std::string>>(), 
            py::arg("n_left"), py::arg("n_inner"), py::arg("n_right"), 
            py::arg("dtype"), py::arg("d"), py::arg("b"),
            py::arg("seed") = std::nullopt,
            py::arg("hash_name") = std::nullopt,
            py::arg("layout") = std::nullopt)
        .def_property_readonly("C", &MatmulObject<Algorithm::FFT>::get_C)
        .def("multiply", &MatmulObject<Algorithm::FFT>::multiply,
            py::arg("A"), py::arg("B"),
            py::arg("transpose_A") = std::nullopt,
            py::arg("transpose_B") = std::nullopt);
#endif // COMPMATMUL_ENABLE_FFT
#ifdef COMPMATMUL_ENABLE_FWHT
    py::class_<MatmulObject<Algorithm::FWHT>>(m, "MatmulFWHT")
        .def(py::init<index_t, index_t, index_t, py::object, int, int, 
            std::optional<uint64_t>, 
            std::optional<std::string>, std::optional<std::string>>(), 
            py::arg("n_left"), py::arg("n_inner"), py::arg("n_right"), 
            py::arg("dtype"), py::arg("d"), py::arg("b"),
            py::arg("seed") = std::nullopt,
            py::arg("hash_name") = std::nullopt,
            py::arg("layout") = std::nullopt)
        .def_property_readonly("C", &MatmulObject<Algorithm::FWHT>::get_C)
        .def("multiply", &MatmulObject<Algorithm::FWHT>::multiply,
            py::arg("A"), py::arg("B"),
            py::arg("transpose_A") = std::nullopt,
            py::arg("transpose_B") = std::nullopt);
#endif // COMPMATMUL_ENABLE_FWHT
    m.def("matmul_gemm", &matmul_gemm, py::arg("A"), py::arg("B"), 
        py::arg("layout") = std::nullopt, 
        py::arg("transpose_A") = std::nullopt, 
        py::arg("transpose_B") = std::nullopt,
        "Perform matrix multiplication of two 2D arrays A and B");
#ifdef COMPMATMUL_ENABLE_FFT
    m.def("matmul_fft", &matmul_fft, py::arg("A"), py::arg("B"), py::arg("d"), py::arg("b"),
        py::arg("hash_name") = std::nullopt, py::arg("seed") = std::nullopt,
        py::arg("layout") = std::nullopt, py::arg("transpose_A") = std::nullopt, 
        py::arg("transpose_B") = std::nullopt,
        "Perform FFT-based matrix multiplication of two 2D arrays A and B");
#endif // COMPMATMUL_ENABLE_FFT
#ifdef COMPMATMUL_ENABLE_FWHT
    m.def("matmul_fwht", &matmul_fwht, py::arg("A"), py::arg("B"), py::arg("d"), py::arg("b"),
        py::arg("hash_name") = std::nullopt, py::arg("seed") = std::nullopt,
        py::arg("layout") = std::nullopt, py::arg("transpose_A") = std::nullopt, 
        py::arg("transpose_B") = std::nullopt,
        "Perform FWHT-based matrix multiplication of two 2D arrays A and B");
#endif // COMPMATMUL_ENABLE_FWHT
}

