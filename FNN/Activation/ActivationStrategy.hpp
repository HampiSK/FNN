#pragma once

#include "ActivationStrategyInterface.hpp"

namespace fnn
{
    // Concrete activation functions

    class EmptyActivationStrategy final : public INeuronFunctionStrategy
    {
    public:
        ~EmptyActivationStrategy() override = default;

        float Activation(const float input) override;
        float Derivation(const float activationOutput) override;
    };

    class ReLUStrategy final : public INeuronFunctionStrategy
    {
    public:
        const float m_threshold = 0.5f;

        explicit ReLUStrategy(const float threshold = 0.5f);
        ~ReLUStrategy() override = default;

        float Activation(const float input) override;
        float Derivation(const float activationOutput) override;
    };

    class SigmoidStrategy final : public INeuronFunctionStrategy
    {
    public:
        ~SigmoidStrategy() override = default;

        float Activation(const float input) override;
        float Derivation(const float activationOutput) override;
    };

    class TanhStrategy final : public INeuronFunctionStrategy
    {
    public:
        ~TanhStrategy() override = default;

        float Activation(const float input) override;
        float Derivation(const float activationOutput) override;
    };

    class LinearStrategy final : public INeuronFunctionStrategy
    {
    public:
        ~LinearStrategy() override = default;

        float Activation(const float input) override;
        float Derivation(const float activationOutput) override;
    };

}
