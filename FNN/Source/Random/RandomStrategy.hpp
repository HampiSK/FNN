#pragma once

#include <chrono>
#include <cstdint>

#include "RandomStrategyInterface.hpp"

namespace fnn
{
    /**
     * @class FastRandomStrategy
     * @brief Provides a fast random number generation strategy for initializing weights
     *
     * Utilizes a high-resolution clock for seed initialization to generate random weights within a specified range.
     */
    class FastRandomStrategy final : public IRandomStrategy
    {
    public:
        ~FastRandomStrategy() override = default;

        /**
         * @brief Generates a random weight within the specified range
         * @param min [in] Minimum value for the weight, default is 0
         * @param max [in] Maximum value for the weight, default is 1
         * @return A random float between min and max
         */
        float GetWeight(const float min = 0, const float max = 1) override;

    private:
        uint64_t m_seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()); ///< Seed for the random number generator
    };
}
