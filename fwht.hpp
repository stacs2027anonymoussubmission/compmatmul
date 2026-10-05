#ifndef COMPMATMUL_FWHT_HPP
#define COMPMATMUL_FWHT_HPP

#ifdef COMPMATMUL_ENABLE_FWHT

#if defined(COMPMATMUL_USE_FXT_FWHT) + defined(COMPMATMUL_USE_FFHT_FWHT) != 1
#error "Exactly one FWHT backend must be defined"
#endif

#include "libarray/array.hpp"

#ifdef COMPMATMUL_USE_FXT_FWHT
#include <walsh/walshwak.h>
#endif // COMPMATMUL_USE_FXT_FWHT

#ifdef COMPMATMUL_USE_FFHT_FWHT
#include "FFHT/fht_header_only.h"
#endif // COMPMATMUL_USE_FFHT_FWHT

namespace compmatmul {
    using array::index_t;
    using array::ilog2;

    /**
     * Performs FWHT using the specified backend.
     */
    template<typename T>
    class FWHTTransformer {
    public:
        /**
         * Initializes a FWHT of size n. The size must be a power of 2.
         * @param n The size of the FWHT to be performed (must be a power of 2)
         */
        explicit FWHTTransformer(index_t n) : n(n) {
            if (n <= 0 || (n & (n - 1)) != 0) {
                throw std::invalid_argument("FWHT size must be a positive power of 2");
            }
            ldn = ilog2(n);
        }

        /**
         * Performs the FWHT on the input array.
         * **Note.** This is not normalized, so you need to scale the output 
         * by 1/sqrt(n).
         * @param x The input array (modified in place)
         */
        inline void operator()(T* x) {
#if defined(COMPMATMUL_USE_FXT_FWHT)
            walsh_wak(x, ldn);
#elif defined(COMPMATMUL_USE_FFHT_FWHT)
            fht(x, ldn);
#else
#error "FWHT not implemented for this backend"
#endif // defined(COMPMATMUL_USE_FXT_FWHT)
        }



    private:
        index_t n; ///< The size of the FWHT
        unsigned long ldn; ///< The log2 of the size of the FWHT
    };
}

#endif // COMPMATMUL_ENABLE_FWHT

#endif // COMPMATMUL_FWHT_HPP
