#ifndef LIBCOMPMATMUL_HASHING_HPP
#define LIBCOMPMATMUL_HASHING_HPP

#include "libarray/random.hpp"
#include "libarray/array.hpp"
#include <cstdint>

/**
 * @file hashing.hpp
 * This file includes the hash functions used in the library, mainly following
 * Mikkel Thorup: "High Speed Hashing for Integers and Strings" (2020). arXiv: 1504.06804
 */

namespace compmatmul {
    using array::index_t;
 
    /**
     * A simple multiply-add-shift hash
     * That is, draw random 64-bit unsigned integers a, b, and compute the
     * hash of an integer x as (a * x + b) >> (64 - r), where r is the number 
     * of bits in the hash value.
     * 
     * The class exists mainly for convenience to provide easy access to 
     * required types and constants when used together with templates.
     */
    class MultiplyAddShiftHash {
    public:
        using hash_t = std::uint64_t; ///< Hash value type
        using key_t = std::uint64_t; ///< Key type (input type)
        using param_t = std::uint64_t; ///< Parameter type (random values a, b)
        using seed_t = std::uint64_t; ///< Seed type for random number generation
        static constexpr index_t HASH_BITS = 64; ///< Number of bits in the hash value
        static constexpr index_t NUM_PARAMS = 2; ///< Number of parameters (a, b)



        /**
         * Compute the hash value of a key using the multiply-add-shift method.
         * Essentially, hash the key into 2^r buckets, that is, the hash value 
         * is in the range {0, 1, ..., 2^r - 1}.
         * @param key The key to be hashed.
         * @param params The parameters (a, b) used in the hash function.
         * @param r The number of bits in the hash value.
         * @return The computed hash value.
         */
        inline static hash_t bucket_hash_pow2(const key_t& key, 
            const param_t* params, index_t r) {
            param_t a = params[0];
            param_t b = params[1];
            hash_t hash_value = (a * key + b) >> (HASH_BITS - r);
            return hash_value;
        }



        /**
         * Compute the sign of the hash value of a key using the multiply-add-shift method.
         * The sign is determined by the most significant bit of the hash value.
         * @param key The key to be hashed.
         * @param params The parameters (a, b) used in the hash function.
         * @return 1 if the hash value is odd, -1 if it is even
         */
        inline static int sign_hash(const key_t& key, const param_t* params) {
            param_t a = params[0];
            param_t b = params[1];
            hash_t hash_value = (a * key + b);
            return (hash_value >> (HASH_BITS - 1)) ? 1 : -1;
        }



        /**
         * Draw one set of random parameters (a,b).
         * Note: This is very inefficient, please use the draw() function if 
         * you want to draw a larger number of such parameters.
         */
        inline static void draw_one(param_t* params, seed_t seed) {
            array::random::RNG_t rng(seed);
            rng.fill(NUM_PARAMS, params);
        }



        /**
         * Draw a number of sets of random parameters (a,b).
         * @param num_params The number of sets of parameters to draw.
         * @param params The array to store the drawn parameters. It should have
         *        size at least num_params * NUM_PARAMS.
         * @param seed The seed for the random number generator.
         */
        inline static void draw(index_t num_params, param_t* params, seed_t seed) {
            array::random::RNG_t rng(seed);
            rng.fill(num_params * NUM_PARAMS, params);
        }
    };
}

#endif //LIBCOMPMATMUL_HASHING_HPP