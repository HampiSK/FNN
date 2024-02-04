#pragma once

namespace fnn
{
    class IRandomStrategy
    {
    public:
        virtual ~IRandomStrategy() = default;

        virtual float GetWeight(const float min, const float max) = 0;
    };
}