#pragma once

#include <vector>
#include <memory>

#include "../Edge/Edge.hpp"
#include "../Activation/ActivationStrategyInterface.hpp"

namespace fnn
{
    class INeuronErrorStrategy
    {
    public:
        virtual ~INeuronErrorStrategy() = default;

        // Used for input/hidden neurons
        virtual float CalculateError(const std::vector<Edge> &tailEdges, const std::vector<Edge> &headEdges, const float error) = 0;

        // Used for output neurons
        virtual float CalculateError(const float target, const float actual) = 0;
    };

    class INeuronValueStrategy
    {
    public:
        virtual ~INeuronValueStrategy() = default;

        virtual float CalculateValue(const std::vector<Edge> &headEdges, const std::shared_ptr<INeuronFunctionStrategy> activationFunction) = 0;
    };

    class INeuronWeightStrategy
    {
    public:
        virtual ~INeuronWeightStrategy() = default;

        virtual void UpdateConectedWeights(std::vector<Edge> &headEdges, const float learningRate, const float error) = 0;
    };
}
