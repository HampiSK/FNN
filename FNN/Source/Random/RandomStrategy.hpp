#pragma once

#include <chrono>
#include <cstdint>

#include "RandomStrategyInterface.hpp"

namespace fnn
{
    class FastRandomStrategy final : public IRandomStrategy
    {
    public:
        ~FastRandomStrategy() override = default;

        float GetWeight(const float min = 0, const float max = 1) override;

    private:
        // Initial seed
        uint64_t m_seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    };
}
