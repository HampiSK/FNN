#pragma once

namespace fnn
{
    class INeuronFunctionStrategy
    {
    public:
        virtual ~INeuronFunctionStrategy() = default;

        virtual float Activation(const float input) = 0;
        virtual float Derivation(const float activationOutput) = 0;
    };
}
