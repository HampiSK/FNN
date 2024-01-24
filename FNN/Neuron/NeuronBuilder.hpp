#pragma once

#include <memory>

#include "Neuron.hpp"
#include "NeuronStrategyInterface.hpp"

namespace fnn
{
    class NeuronBuilder final
    {
    public:
        static NeuronBuilder create();

        NeuronBuilder AsType(const NeuronType type);

        NeuronBuilder HasTarget(const float target = 0.0f);
        NeuronBuilder HasLearningRate(const float learningRate = 0.2f);
        NeuronBuilder HasHeadConnection(const size_t size = 0);
        NeuronBuilder HasTailConnection(const size_t size = 0);

        NeuronBuilder WithActivationFunction(const std::shared_ptr<INeuronFunctionStrategy> activationFunction);
        NeuronBuilder WithErrorCalculation(const std::shared_ptr<INeuronErrorStrategy> errorCalculation);
        NeuronBuilder WithValueCalculation(const std::shared_ptr<INeuronValueStrategy> valueCalculation);
        NeuronBuilder WithWeightCalculation(const std::shared_ptr<INeuronWeightStrategy> weightCalculation);

        std::shared_ptr<Neuron> Build();

    private:
        std::shared_ptr<Neuron> m_neuron = std::make_shared<Neuron>();
    };
}
