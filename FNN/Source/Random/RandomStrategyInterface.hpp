#pragma once

namespace fnn
{
    /**
     * @interface IRandomStrategy
     * @brief Defines an interface for random number generation strategies
     *
     * Allowing for different randomization strategies to be applied
     */
    class IRandomStrategy
    {
    public:
        virtual ~IRandomStrategy() = default;

        /**
         * @brief Generates a random weight within a specified range
         * @param min [in] Minimum bound for the generated weight
         * @param max [in] Maximum bound for the generated weight
         * @return A random weight as a float between min and max
         */
        virtual float GetWeight(const float min, const float max) = 0;
    };
}
