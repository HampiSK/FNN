#pragma once

#include <vector>
#include <memory>

#include "../Edge/Edge.hpp"
#include "../Activation/ActivationStrategyInterface.hpp"
#include "NeuronStrategyInterface.hpp"

namespace fnn
{
    // Concrete neuron strategies

    class NeuronErrorStrategy final : public INeuronErrorStrategy
    {
    public:
        ~NeuronErrorStrategy() override = default;

        float CalculateError(const std::vector<Edge> &tailEdges, const std::vector<Edge> &headEdges, const float error) override;
        float CalculateError(const float target, const float actual) override;
    private:
        float CalculateErrorPortion(const float weightSum, const float connectedWeight, const float error) const;
    };

    class NeuronValueStrategy final : public INeuronValueStrategy
    {
    public:
        ~NeuronValueStrategy() override = default;

        float CalculateValue(const std::vector<Edge> &headEdges, const std::shared_ptr<INeuronFunctionStrategy> activationFunction) override;
    };

    class NeuronWeightStrategy final : public INeuronWeightStrategy
    {
    public:
        ~NeuronWeightStrategy() override = default;

        void UpdateConectedWeights(std::vector<Edge> &headEdges, const float learningRate, const float error) override;
    };
}
