#include "RandomStrategy.hpp"

using namespace fnn;

float FastRandomStrategy::GetWeight(const float min, const float max)
{
    constexpr uint64_t a = 1664525;    // LCG multiplier
    constexpr uint64_t c = 1013904223; // LCG increment
    constexpr uint64_t m = 2147483647; // Modulus (2^31 - 1)

    // Update the seed
    m_seed = (a * m_seed + c) % m;

    // Scale and then adjust to min, max
    return static_cast<float>(m_seed) / static_cast<float>(m) * (max - min) + min;
}
