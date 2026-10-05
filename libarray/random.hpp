#ifndef LIBARRAY_RANDOM_HPP
#define LIBARRAY_RANDOM_HPP

#include "array.hpp"

#ifdef __SSE2__
#include <x86intrin.h>
#include <wmmintrin.h>
#endif // __SSE2__

#ifdef __aarch64__
#include <arm_neon.h>
#endif // __aarch64__

#include <cstdint>
#include <type_traits>
#include <cassert>
#include <random>
#include <numbers>

#if defined(__BYTE_ORDER__) && \
    (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "This library only supports little-endian architectures."
#endif

#if defined(__aarch64__) && !defined(__ARM_FEATURE_CRYPTO)
#warning "ARM crypto AES instructions are required for best performance. " \
    "If you are compiling for Apple Silicon, add +crypto to your -march flag."
#endif


/**
 * @file random.hpp
 * @brief Random number generation functions.
 * 
 * This file contains implementations of Vigna's random number generation functions that are
 * faster than the standard library's random number generators. Also, provides an option to 
 * use AES-NI hardware accelerated random number generation through the Random123 library.
 */



namespace array {
    namespace random {
        /**
         * Performs the Box-Muller transform to generate two independent standard 
         * normally distributed random numbers from two uniformly distributed random numbers.
         * @param z0 The first output normally distributed random number.
         * @param z1 The second output normally distributed random number.
         * @param u1 The first input uniformly distributed random number in (0, 1).
         * @param u2 The second input uniformly distributed random number in (0, 1).
         */
        template<typename T>
        inline void box_muller(T& z0, T& z1, T u1, T u2) {
            if (u1 == 0) {
                u1 = std::numeric_limits<T>::min(); // avoid log(0)
            }
            T R = std::sqrt(T{-2.0} * std::log(u1));
            T Theta = 2.0 * std::numbers::pi_v<T> * u2;
            z0 = R * std::cos(Theta);
            z1 = R * std::sin(Theta);
        }



        /**
         * Standard left-rotate function for 64-bit unsigned integers.
         * @param x The value to rotate.
         * @param k The number of bits to rotate.
         * @return The rotated value.
         */
        inline uint64_t rotl(const uint64_t x, int k) {
            return (x << k) | (x >> (64 - k));
        }



        /**
         * SplitMix64 by Sebastiano Vigna (vigna@acm.org)
         * Adapted from https://prng.di.unimi.it/splitmix64.c
         * 
         * @param x Current state (to be updated)
         * @return A pseudo-random 64-bit unsigned integer
         */
        inline uint64_t splitmix64(uint64_t& x) {
        	uint64_t z = (x += 0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
            z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
            return z ^ (z >> 31);
        }



        /**
         * xoshiro256++ by David Blackman and Sebastiano Vigna
         * Adapted from https://prng.di.unimi.it/xoshiro256plusplus.c
         * 
         * @param s Current state (array of 4 uint64_t)
         * @return A pseudo-random 64-bit unsigned integer
         */
        inline uint64_t xoshiro256plusplus(uint64_t* s) {
	        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];

        	const uint64_t t = s[1] << 17;

            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];

            s[2] ^= t;

            s[3] = rotl(s[3], 45);

            return result;
        }



        /**
         * xoshiro256+ by David Blackman and Sebastiano Vigna
         * Adapted from https://prng.di.unimi.it/xoshiro256plus.c
         * 
         * @param s Current state (array of 4 uint64_t)
         * @return A pseudo-random 64-bit unsigned integer
         */
        inline uint64_t xoshiro256plus(uint64_t* s) {
            const uint64_t result = s[0] + s[3];

	        const uint64_t t = s[1] << 17;

            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];

            s[2] ^= t;

            s[3] = rotl(s[3], 45);

            return result;
        }



#if defined(__SSE2__) && defined(__AES__)
        /**
         * Implements the ARS (Advanced Randomization System) algorithm 
         * using AES-NI instructions, following Salmon, Moraes, Dror, 
         * Shaw (2011).
         * 
         * @param rounds Number of rounds to perform (typically 5)
         * @param counter 128-bit counter value (as __m128i)
         * @param key 128-bit key value (as __m128i)
         * @return A pseudo-random 128-bit value (as __m128i)
         */
        __m128i ars(int rounds, __m128i counter, __m128i key) {
            const __m128i key_constants = 
                _mm_set_epi64x(0xBB67AE8584CAA73B,  // sqrt(3) - 1.0 
                               0x9E3779B97F4A7C15); // golden ratio 
            __m128i state = _mm_xor_si128(counter, key);
            for (int i = 1; i < rounds; ++i) {
                key = _mm_add_epi64(key, key_constants);
                state = _mm_aesenc_si128(state, key);
            }
            key = _mm_add_epi64(key, key_constants);
            state = _mm_aesenclast_si128(state, key);
            return state;
        }



        /**
         * Treats an XMM value as a 128-bit unsigned integer and increments it
         * by one. This is useful for incrementing a 128-bit counter in the 
         * ARS algorithm.
         * 
         * @param c The 128-bit counter value (as __m128i)
         * @return The updated 128-bit counter value (as __m128i)
         */
        inline __m128i increment128(__m128i c) {
            uint64_t lo = _mm_extract_epi64(c, 0);
            uint64_t hi = _mm_extract_epi64(c, 1);
            ++lo;
            if (__builtin_expect(lo == 0, 0)) {
                ++hi;
            }
            return _mm_set_epi64x(hi, lo);
        }



        /**
         * Treats an XMM value as a 128-bit unsigned integer and adds a 
         * 64-bit unsigned integer to it. This is useful for incrementing 
         * a 128-bit counter in the ARS algorithm.
         * 
         * @param c The 128-bit counter value (as __m128i)
         * @param d The 64-bit value to add
         * @return The updated 128-bit counter value (as __m128i)
         */
        inline __m128i add128(__m128i c, uint64_t d) {
            alignas(16) uint64_t x[2];
            _mm_store_si128(reinterpret_cast<__m128i*>(x), c);
            unsigned char carry = _addcarry_u64(0, x[0], d, reinterpret_cast<unsigned long long*>(x));
            if (__builtin_expect(carry,0)) {
                ++x[1];
            }
            return _mm_load_si128(reinterpret_cast<__m128i*>(x));
        }
#endif // defined(__SSE2__) && defined(__AES__)



#ifdef __AVX512F__
        /**
         * Increments four 128-bit counters (stored in an array of __m512i) by one, treating
         * each 128-bit value as a pair of 64-bit integers.
         * @param counters An array of four 128-bit counter values (as __m512i)
         * @return The updated 128-bit counter values (as __m512i)
         */
        inline __m512i increment512(__m512i counters) {
            const __m512i ones_lo = _mm512_set_epi64(0, 1, 0, 1, 0, 1, 0, 1);
            const __m512i ones_hi = _mm512_set_epi64(1, 0, 1, 0, 1, 0, 1, 0);
            const __m512i all_ones = _mm512_set1_epi64(static_cast<int64_t>(-1));

            __mmask8 carry_from_low = _mm512_cmpeq_epu64_mask(counters, all_ones) & 0x55;
            counters = _mm512_add_epi64(counters, ones_lo);
            if (__builtin_expect(carry_from_low != 0, 0)) {
                __mmask8 carry_to_high = carry_from_low << 1;
                counters = _mm512_mask_add_epi64(counters, carry_to_high, counters, ones_hi);
            }
            return counters;
        }



        /**
         * Adds a 64-bit unsigned integer to four 128-bit counters (stored in an array of __m512i) .
         * @param counters An array of four 128-bit counter values (as __m512i)
         * @param a The 64-bit value to add
         * @return The updated 128-bit counter values (as __m512i)
         */
        inline __m512i add512(__m512i counters, uint64_t a) {
            __m512i old = counters;
            __m512i addend = _mm512_set_epi64(0, a, 0, a, 0, a, 0, a);
            counters = _mm512_add_epi64(counters, addend);
            __mmask8 carry_from_low = _mm512_cmplt_epu64_mask(counters, old);
            if (__builtin_expect(carry_from_low != 0, 0)) {
                __mmask8 carry_to_high = carry_from_low << 1;
                const __m512i ones = _mm512_set1_epi64(1);
                counters = _mm512_mask_add_epi64(counters, carry_to_high, counters, ones);
            }
            return counters;
        }



        /**
         * Turns out it can be faster to increment stuff in a pipeline 
         * because the masked instructions are sometimes slower.
         * @param counters An array of four 128-bit counter values (as __m512i)
         * @return The updated 128-bit counter values (as __m512i)
         */
        inline __m512i increment4x128(__m512i counters) {
            __m128i* c = reinterpret_cast<__m128i*>(&counters);
            __m512i d;
            __m128i* e = reinterpret_cast<__m128i*>(&d);
            for (int i = 0; i < 4; ++i) {
                e[i] = increment128(c[i]);
            }
            return d;
        }
#endif // __AVX512F__



#if defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)
        /**
         * Treats a 128-bit vector value as a 128-bit unsigned integer and 
         * increments it by one. This is useful for incrementing a 128-bit 
         * counter in the ARS algorithm.
         * 
         * @param c The 128-bit counter value (as uint8x16_t)
         */
        inline uint8x16_t increment128(uint8x16_t c) {
            uint64x2_t w = vreinterpretq_u64_u8(c);
            uint64_t lo = vgetq_lane_u64(w, 0);
            uint64_t hi = vgetq_lane_u64(w, 1);
            ++lo;
            if (__builtin_expect(lo == 0, 0)) {
                ++hi;
            }
            return vreinterpretq_u8_u64(uint64x2_t {lo, hi});
        }



        /**
         * Treates a 128-bit vector value as a 128-bit unsigned integer and 
         * adds a 64-bit unsigned integer to it.
         * @param c The 128-bit counter value (as uint8x16_t)
         * @param d The 64-bit value to add
         */
        inline uint8x16_t add128(uint8x16_t c, uint64_t d) {
            uint64x2_t w = vreinterpretq_u64_u8(c);
            uint64_t lo = vgetq_lane_u64(w, 0);
            uint64_t hi = vgetq_lane_u64(w, 1);
            lo += d;
            if (__builtin_expect(lo < d, 0)) {
                ++hi;
            }
            return vreinterpretq_u8_u64(uint64x2_t {lo, hi});
        }



        /**
         * Implements the ARS (Advanced Randomization System) algorithm 
         * using AES-NI instructions, following Salmon, Moraes, Dror, 
         * Shaw (2011).
         * 
         * @param rounds Number of rounds to perform (typically 5)
         * @param counter 128-bit counter value (as uint8x16_t)
         * @param key 128-bit key value (as uint8x16_t)
         * @return A pseudo-random 128-bit value (as uint8x16_t)
         */
        uint8x16_t ars(int rounds, uint8x16_t counter, uint8x16_t key) {
            const uint64x2_t key_constants =  {
                0x9E3779B97F4A7C15, // golden ratio 
                0xBB67AE8584CAA73B  // sqrt(3) - 1.0 
            }; 
            uint8x16_t state = counter;
            for (int i = 0; i < rounds - 1; ++i) {
                // arm does also the xor
                state = vaeseq_u8(state, key);
                state = vaesmcq_u8(state);
                key = vreinterpretq_u8_u64(
                    vaddq_u64(vreinterpretq_u64_u8(key), key_constants)
                );
            }

            state = vaeseq_u8(state, key);
            key = vreinterpretq_u8_u64(
                vaddq_u64(vreinterpretq_u64_u8(key), key_constants)
            );
            state = veorq_u8(state, key);

            return state;
        }        
#endif // __aarch64__

        /**
         * XoShiro256PlusPlusRNG implements the xoshiro256++ random number 
         * generator as a C++ class. The generator is initialized with 
         * SplitMix64 and provides as a convenience the possibility to generate
         * random numbers in arrays of uint64_t. Also, it is possible to 
         * generate uniformly random numbers in [0,1) as double-precision 
         * floats. The generator is not thread-safe.
         */
        class XoShiRo256PlusPlusRNG {
        public:
        /**
         * Constructs a new XoShiro256PlusPlusRNG with the given seed.
         * @param seed The initial seed for the random number generator.
         */
        explicit XoShiRo256PlusPlusRNG(
            std::optional<uint64_t> seed = std::nullopt) :
            buffer_index(BUFFER_SIZE) {
            uint64_t seed_val = seed.value_or(std::random_device{}());
            for (int i = 0; i < 4; ++i) {
                state[i] = splitmix64(seed_val);
            }
        }



        /**
         * Generates a random number of type T. The type T can be either 
         * uint64_t or double. If T is uint64_t, the returned random number 
         * will be uniformly distributed in the range {0, 2^64 - 1}. If T is 
         * double, the returned random number will be uniformly distributed in 
         * the range [0,1).
         * @return A uniformly distributed random number of type T.
         */
        template<typename T>
        inline T next() {
            void* ptr;
            if (buffer_index > BUFFER_SIZE - static_cast<int>(sizeof(T))) {
                *reinterpret_cast<uint64_t*>(buffer) = 
                    xoshiro256plusplus(state);
                buffer_index = 0;
            }
            ptr = &buffer[buffer_index];
            buffer_index += sizeof(T);

            if constexpr (std::is_same_v<T, uint64_t>) {
                return *reinterpret_cast<uint64_t*>(ptr);
            } 
            else if constexpr (std::is_same_v<T, uint32_t>) {
                return *reinterpret_cast<uint32_t*>(ptr);
            }
            else if constexpr (std::is_same_v<T, double>) {
                return (*reinterpret_cast<uint64_t*>(ptr) >> 11) * 0x1.0p-53;
            }
            else if constexpr (std::is_same_v<T, float>) {
                return (*reinterpret_cast<uint32_t*>(ptr) >> 8) * 0x1.0p-24f;
            }
            else {
                static_assert(false, "Invalid type for next().");
                return 0;
            }
        }



        /**
         * Generates an array of random numbers of type T.
         * @param size The number of random numbers to generate.
         * @param A The array to store the generated random numbers.
         */
        template<typename T>
        inline void fill(index_t size, T* A) {
            static_assert(std::is_same_v<T, uint64_t> || 
                std::is_same_v<T, double> || std::is_same_v<T, uint32_t> || 
                std::is_same_v<T, float>, 
                    "XoShiRo256PlusPlusRNG can only generate uint64_t, "
                    "double, uint32_t, or float.");

            const index_t ALIGNMENT = sizeof(uint64_t);
            const index_t ELEMENTS_PER_ITERATION = sizeof(uint64_t) / sizeof(T);
            index_t i = 0;
            index_t alignment_offset = (ALIGNMENT - 
                reinterpret_cast<uintptr_t>(A) % ALIGNMENT) % ALIGNMENT / 
                sizeof(T);
            for (; i < std::min(alignment_offset, size); ++i) {
                // Fill until A is 64-bit aligned
                A[i] = next<T>();
            }
            index_t iterations = (size - i) / ELEMENTS_PER_ITERATION;
            for (index_t j = 0; j < iterations; ++j) {
                index_t k = alignment_offset + j * ELEMENTS_PER_ITERATION;
                uint64_t x = xoshiro256plusplus(state);
                if constexpr (std::is_same_v<T, uint64_t>) {
                    A[k] = x;
                }
                else if constexpr (std::is_same_v<T, uint32_t>) {
                    *reinterpret_cast<uint64_t*>(A + k) = x;
                }
                else if constexpr (std::is_same_v<T, double>) {
                    A[k] = (x >> 11) * 0x1.0p-53;
                }
                else if constexpr (std::is_same_v<T, float>) {
                    A[k] = ((x & 0xffffffff) >> 8) * 0x1.0p-24f;
                    A[k + 1] = (x >> 40) * 0x1.0p-24f;
                }
            }
            i += iterations * ELEMENTS_PER_ITERATION;
            for (; i < size; ++i) {
                // Fill the remaining elements
                A[i] = next<T>();
            }
        }

        private:
            static constexpr int BUFFER_SIZE = sizeof(uint64_t);

            uint64_t state[4];
            int buffer_index;
            uint8_t buffer[BUFFER_SIZE];
        };



#if defined(__SSE2__) && defined(__AES__) || defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)
        /**
         * ARS5RNG implements the ARS5 random number in a simplified 
         * manner without using AVX-512 instructions.
         * The generator is initialized with 
         * SplitMix64 and provides as a convenience the possibility to generate
         * random numbers in arrays of uint64_t. Also, it is possible to 
         * generate uniformly random numbers in [0,1) as double-precision 
         * floats. The generator is not thread-safe.
         */
        class ARS5RNG {
        public:
        /**
         * Constructs a new ARS5RNG with the given seed.
         * @param seed The initial seed for the random number generator.
         */
        explicit ARS5RNG(std::optional<uint64_t> seed = std::nullopt) :
            buffer_index(BUFFER_SIZE) {
            alignas(64) uint64_t x[2];
            alignas(64) uint64_t y[2];
            uint64_t seed_val = seed.value_or(std::random_device{}());
            for (int j = 0; j < 2; ++j) 
                x[j] = splitmix64(seed_val);
            for (int j = 0; j < 2; ++j) 
                y[j] = splitmix64(seed_val);
#if defined(__AVX512F__) && defined(__VAES__)
            round_keys[0] =  _mm512_broadcast_i64x2(*reinterpret_cast<__m128i*>(x));
            const __m512i key_constants = 
                _mm512_set_epi64(0xBB67AE8584CAA73B,
                                 0x9E3779B97F4A7C15,
                                 0xBB67AE8584CAA73B,
                                 0x9E3779B97F4A7C15,
                                 0xBB67AE8584CAA73B,
                                 0x9E3779B97F4A7C15,
                                 0xBB67AE8584CAA73B,
                                 0x9E3779B97F4A7C15);
            for (int j = 1; j < 6; ++j) {
                round_keys[j] = _mm512_add_epi64(round_keys[j-1], key_constants);
            }
            alignas(64) __m128i counters[4];
            for (int i = 0; i < 4; ++i) {
                counters[i] = _mm_loadu_si128(reinterpret_cast<__m128i*>(y));
            }
            counters[1] = add128(counters[1], 1);
            counters[2] = add128(counters[2], 2);
            counters[3] = add128(counters[3], 3);
            counter = _mm512_loadu_si512(reinterpret_cast<__m512i*>(counters));
#elif defined(__SSE2__) && defined(__AES__)
            round_keys[0] = _mm_loadu_si128(reinterpret_cast<__m128i*>(x));
            const __m128i key_constants = 
                _mm_set_epi64x(0xBB67AE8584CAA73B,  // sqrt(3) - 1.0 
                               0x9E3779B97F4A7C15); // golden ratio 
            for (int j = 1; j < 6; ++j) {
                round_keys[j] = _mm_add_epi64(round_keys[j-1], key_constants);
            }
            counter = _mm_loadu_si128(reinterpret_cast<__m128i*>(y));
#elif defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)
            round_keys[0] = vreinterpretq_u8_u64(vld1q_u64(x));
            const uint64x2_t key_constants {
                0x9E3779B97F4A7C15, // golden ratio 
                0xBB67AE8584CAA73B, // sqrt(3) - 1.0 
            };
                               
            for (int j = 1; j < 6; ++j) {
                round_keys[j] = vreinterpretq_u8_u64(vaddq_u64(
                    vreinterpretq_u64_u8(round_keys[j-1]), key_constants
                ));
            }
            counter = vreinterpretq_u8_u64(vld1q_u64(y));
#else
#error "ARS5RNG requires either AVX-512 with VAES, SSE2 with AES support, or ARM with crypto support."
#endif
        }



        /**
         * Generates a random number of type T. The type T can be either 
         * uint64_t or double. If T is uint64_t, the returned random number 
         * will be uniformly distributed in the range {0, 2^64 - 1}. If T is 
         * double, the returned random number will be uniformly distributed in 
         * the range [0,1).
         * @return A uniformly distributed random number of type T.
         */
        template<typename T>
        inline T next() {
            static_assert(std::is_same_v<T, uint64_t> || 
                std::is_same_v<T, double> ||
                std::is_same_v<T, uint32_t> ||
                std::is_same_v<T, float>,
                "ARS5RNG can only generate uint64_t, double, uint32_t, "
                "or float.");


#if defined(__AVX512F__) && defined(__VAES__)
            if (BUFFER_SIZE - buffer_index < static_cast<int>(sizeof(T))) {
                simd_t state = _mm512_xor_si512(counter, round_keys[0]);
                for (int r = 1; r < 5; ++r) {
                    state = _mm512_aesenc_epi128(state, round_keys[r]);
                }
                state = _mm512_aesenclast_epi128(state, round_keys[5]);
                counter = add512(counter, 4);
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(buffer), state);
                buffer_index = 0;
            }
#elif defined(__SSE2__) && defined(__AES__)
            if (BUFFER_SIZE - buffer_index < static_cast<int>(sizeof(T))) {
                simd_t state = _mm_xor_si128(counter, round_keys[0]);
                for (int r = 1; r < 5; ++r) {
                    state = _mm_aesenc_si128(state, round_keys[r]);
                }
                state = _mm_aesenclast_si128(state, round_keys[5]);
                counter = increment128(counter);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), state);
                buffer_index = 0;
            }
#elif defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)
            if (BUFFER_SIZE - buffer_index < static_cast<int>(sizeof(T))) {
                simd_t state = counter;
                for (int r = 0; r < 4; ++r) {
                    state = vaeseq_u8(state, round_keys[r]);
                    state = vaesmcq_u8(state);
                }
                state = vaeseq_u8(state, round_keys[4]);
                state = veorq_u8(state, round_keys[5]);
                counter = increment128(counter);
                vst1q_u8(buffer, state);
                buffer_index = 0;
            }
#else
#error "ARS5RNG requires either AVX-512 with VAES, SSE2 with AES support, "
    "or ARM with crypto support."
#endif

            using uint_t = std::conditional_t<sizeof(T) == sizeof(uint64_t),
                uint64_t,
                std::conditional_t<sizeof(T) == sizeof(uint32_t),
                    uint32_t,void>>;

            uint_t x = *reinterpret_cast<uint_t*>(&buffer[buffer_index]);
            buffer_index += sizeof(T);

            if constexpr (std::is_same_v<T, uint64_t>) {
                return x;
            } 
            else if constexpr (std::is_same_v<T, uint32_t>) {
                return x;
            }
            else if constexpr (std::is_same_v<T, double>) {
                return (x >> 11) * 0x1.0p-53;
            }
            else if constexpr (std::is_same_v<T, float>) {
                return (x >> 8) * 0x1.0p-24;
            }
            else {
                static_assert(false, "Invalid type for next().");
                return 0;
            }
        }



        /**
         * Generates an array of random numbers of type T.
         * @param size The number of random numbers to generate.
         * @param A The array to store the generated random numbers.
         */
        template<typename T>
        inline void fill(index_t size, T* A) {
            static_assert(std::is_same_v<T, uint64_t> || 
                std::is_same_v<T, double> ||
                std::is_same_v<T, uint32_t> ||
                std::is_same_v<T, float>,
                          "ARS5RNG can only generate uint64_t, double, "
                          "uint32_t, or float.");
#if defined(__AVX512F__) && defined(__VAES__)
            const index_t ALIGNMENT = 64;
#elif defined(__SSE2__) && defined(__AES__) || defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)
            const index_t ALIGNMENT = 16;
#endif
            const index_t ELEMENTS_PER_ITERATION = sizeof(simd_t) / sizeof(T);

            index_t i = 0;
            index_t alignment_offset = (ALIGNMENT - 
                reinterpret_cast<uintptr_t>(A) % ALIGNMENT) % ALIGNMENT / 
                sizeof(T);
            for (; i < std::min(alignment_offset, size); ++i) {
                // Fill until A is aligned
                A[i] = next<T>();
            }
            simd_t counter0 = counter;
            index_t iterations = (size - i) / ELEMENTS_PER_ITERATION;

#pragma omp parallel for if (!omp_in_parallel()) schedule(static)
            for (index_t j = 0; j < iterations; ++j) {
                // the main loop
                index_t k = alignment_offset + j * ELEMENTS_PER_ITERATION;

#if defined(__AVX512F__) && defined(__VAES__)
                __m512i counterj = add512(counter0, 4*j);
                __m512i state = _mm512_xor_si512(counterj, round_keys[0]);
                for (int r = 1; r < 5; ++r) {
                    state = _mm512_aesenc_epi128(state, round_keys[r]);
                }
                state = _mm512_aesenclast_epi128(state, round_keys[5]);
                if constexpr (std::is_same_v<T,uint64_t> || 
                    std::is_same_v<T,uint32_t>) {
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(A+k), state);
                }
                else if constexpr (std::is_same_v<T,double>) {
                    const __m512d scale = _mm512_set1_pd(0x1.0p-53);
                    state = _mm512_srli_epi64(state, 11);
                    __m512d d = _mm512_cvtepu64_pd(state);
                    d = _mm512_mul_pd(d, scale);
                    _mm512_storeu_pd(A+k, d);
                }
                else if constexpr (std::is_same_v<T,float>) {
                    const __m512 scale = _mm512_set1_ps(0x1.0p-24f);
                    state = _mm512_srli_epi32(state, 8);
                    __m512 f = _mm512_cvtepu32_ps(state);
                    f = _mm512_mul_ps(f, scale);
                    _mm512_storeu_ps(A+k, f);
                }
                else {
                    static_assert(false, "ARS5RNG can only generate uint64_t, double, "
                        "uint32_t, or float.");
                }
#elif defined(__SSE2__) && defined(__AES__)
                __m128i counterj = add128(counter0, j);
                __m128i state = _mm_xor_si128(counterj, round_keys[0]);
                for (int r = 1; r < 5; ++r) {
                    state = _mm_aesenc_si128(state, round_keys[r]);
                }
                state = _mm_aesenclast_si128(state, round_keys[5]);

                if constexpr (std::is_same_v<T,uint64_t> || 
                    std::is_same_v<T,uint32_t>) {
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(A+k), state);
                }
                else if constexpr (std::is_same_v<T,double>) {
                    const __m128d scale = _mm_set1_pd(0x1.0p-53);
                    state = _mm_srli_epi64(state, 11);
                    uint64_t x[2];
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(x), state);
                    __m128d d = _mm_set_pd(static_cast<double>(x[1]), static_cast<double>(x[0]));
                    d = _mm_mul_pd(d, scale);
                    _mm_storeu_pd(A+k, d);
                }
                else if constexpr (std::is_same_v<T,float>) {
                    const __m128 scale = _mm_set1_ps(0x1.0p-24f);
                    state = _mm_srli_epi32(state, 8);
                    uint32_t x[4];
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(x), state);
                    __m128 f = _mm_set_ps(static_cast<float>(x[3]), static_cast<float>(x[2]),
                                          static_cast<float>(x[1]), static_cast<float>(x[0]));
                    f = _mm_mul_ps(f, scale);
                    _mm_storeu_ps(A+k, f);
                }
                else {
                    static_assert(false, "ARS5RNG can only generate uint64_t, "
                        "double, uint32_t, or float.");
                }
#elif defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)
                simd_t counterj = add128(counter0, j);
                simd_t state = counterj;
                for (int r = 0; r < 4; ++r) {
                    state = vaeseq_u8(state, round_keys[r]);
                    state = vaesmcq_u8(state);
                }
                state = vaeseq_u8(state, round_keys[4]);
                state = veorq_u8(state, round_keys[5]);
                if constexpr (std::is_same_v<T,uint64_t>) {
                    // simply store the bit pattern as is
                    vst1q_u64(A + k, vreinterpretq_u64_u8(state));
                }
                else if constexpr(std::is_same_v<T,uint32_t>) {
                    // simply store the bit pattern as is
                    vst1q_u32(A + k, vreinterpretq_u32_u8(state));
                }
                else if constexpr (std::is_same_v<T,double>) {
                    // store 2 doubles, scaled into [0,1)
                    const float64x2_t scale = vdupq_n_f64(0x1.0p-53);
                    float64x2_t d = vcvtq_f64_u64(
                        vshrq_n_u64(vreinterpretq_u64_u8(state), 11)
                    );
                    d = vmulq_f64(d, scale);
                    vst1q_f64(A + k, d);
                }
                else if constexpr (std::is_same_v<T,float>) {
                    // store 4 floats, scaled into [0,1)
                    const float32x4_t scale = vdupq_n_f32(0x1.0p-24f);
                    float32x4_t f = vcvtq_f32_u32(
                        vshrq_n_u32(vreinterpretq_u32_u8(state), 8)
                    );
                    f = vmulq_f32(f, scale);
                    vst1q_f32(A + k, f);
                }
                else {
                    static_assert(false, "ARS5RNG can only generate uint64_t, "
                        "double, uint32_t, or float.");
                }
#endif
            }
#if defined(__AVX512F__) && defined(__VAES__)
            counter = add512(counter0, 4*iterations);
#elif defined(__SSE2__) && defined(__AES__) || \
    defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)
            counter = add128(counter0, iterations);
#endif
            i += iterations * ELEMENTS_PER_ITERATION;
            for (; i < size; ++i) {
                // fill the remaining elements
                A[i] = next<T>();
            }
        }



        /**
         * TODO: make more efficient, this is here only to provide 
         * a minimal functionality
         */
        template<typename T>
        inline T uniform_int(T a, T b) {
            if (a > b) {
                throw std::invalid_argument("Lower bound must be less than or equal to upper bound.");
            }
            uint64_t diff = static_cast<uint64_t>(b - a);
            uint64_t bitmask = ~(0xffffffffffffffffULL << (ilog2(diff) + 1));
            uint64_t x;
            do {
                x = next<uint64_t>() & bitmask;
            } 
            while (x > diff);
            return a + static_cast<T>(x);
        }



        /**
         * Perform a Knuth shuffle on an array of size `size` using the provided random number generator.
         * TODO: make more efficient, this is here only to provide a minimal functionality
         * @param size The number of elements in the array.
         * @param A The array to shuffle.
         */
        template<typename T>
        inline void shuffle(index_t size, T* A) {
            for (index_t i = size - 1; i > 0; --i) {
                index_t j = uniform_int<index_t>(0, i);
                std::swap(A[i], A[j]);
            }
        }



        /**
         * Fills an array with normally distributed random numbers using the Box-Muller transform.
         * TODO: make more efficient, this is here only to provide a minimal functionality
         * @param size The number of elements in the array.
         * @param A The array to fill with normally distributed random numbers.
         * @param mean The mean of the normal distribution.
         * @param stdev The standard deviation of the normal distribution.
         */
        template<typename T>
        inline void fill_normal(index_t size, T* A, 
            double mean = 0.0, double stdev = 1.0) {
            static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                "fill_normal can only generate double or float.");
            fill(size, A); // Fill with uniform random numbers first
            #pragma omp parallel for if (!omp_in_parallel()) schedule(static)
            for (index_t i = 0; i < size/2*2; i += 2) {
                box_muller(A[i], A[i + 1], A[i], A[i + 1]);
                A[i] = A[i] * stdev + mean;
                A[i + 1] = A[i + 1] * stdev + mean;
            }
            // optionally handle the last element if size is odd
            if (size % 2 != 0) {
                T u1 = A[size - 1];
                T u2 = next<T>();
                box_muller(A[size - 1], u2, u1, u2);
                A[size - 1] = A[size - 1] * stdev + mean;
            }
        }


        private:
#if defined(__AVX512F__) && defined(__VAES__)
            using simd_t = __m512i;
            
#elif defined(__SSE2__) && defined(__AES__) 
            using simd_t = __m128i;
#elif defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)
            using simd_t = uint8x16_t;
#else
            #error "ARS5RNG requires either AVX-512 with VAES, SSE2 with AES support, or ARM with crypto support."
#endif
            static constexpr int BUFFER_SIZE = sizeof(simd_t); // bytes

            simd_t round_keys[6];
            simd_t counter;
            alignas(64) uint8_t buffer[BUFFER_SIZE];
            int buffer_index; // byte pointer
        };
#endif // defined(__SSE2__) && defined(__AES__) || defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)


#if defined(__SSE2__) && defined(__AES__) || defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)
using RNG_t = ARS5RNG;
#else
using RNG_t = XoShiRo256PlusPlusRNG;
#endif



        /**
         * A uniform integer distribution.
         * TODO: make more efficient, this is here only to provide 
         * a minimal functionality
         */
        template<typename T, typename RNG>
        class UniformIntDistribution {
        public:
            explicit UniformIntDistribution(T a, T b, RNG& rng) : a(a), b(b), rng(rng) {
                if (a > b) {
                    throw std::invalid_argument("Lower bound must be less than or equal to upper bound.");
                }
                diff = b - a;
                bitmask = ~(0xffffffffffffffffULL << (ilog2(diff) + 1));
            }

            T operator()() {
                uint64_t x;
                do {
                    x = rng.template next<uint64_t>() & bitmask;
                } 
                while (x > diff);
                return a + static_cast<T>(x);
            }

        private:
            T a; ///< The lower bound of the uniform distribution (inclusive)
            T b; ///< The upper bound of the uniform distribution (inclusive)
            uint64_t diff;
            RNG& rng;
            uint64_t bitmask;
        };
    }
}

#endif // LIBARRAY_RANDOM_HPP