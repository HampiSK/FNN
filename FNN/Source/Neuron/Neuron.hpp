#pragma once

#include <memory>
#include <optional>
#include <vector>
#include <string>

#include "../Edge/Edge.hpp"
#include "NeuronStrategyInterface.hpp"

namespace fnn
{
    enum class NeuronType
    {
        Unknown = -1,
        Input,
        Hidden,
        Output,
    };

    std::string neuronTypeToString(const NeuronType type);

    struct Neuron final
    {
    public:
        NeuronType m_neuronType = NeuronType::Unknown;
        float m_value = 0;
        float m_error = 0;

        std::optional<float> m_target;
        std::optional<float> m_learningRate;
        std::optional<std::vector<Edge>> m_headConnections;
        std::optional<std::vector<Edge>> m_tailConnections;

        std::optional<std::shared_ptr<INeuronFunctionStrategy>> m_activationFunction;
        std::optional<std::shared_ptr<INeuronErrorStrategy>> m_errorCalculation;
        std::optional<std::shared_ptr<INeuronValueStrategy>> m_valueCalculation;
        std::optional<std::shared_ptr<INeuronWeightStrategy>> m_weightCalculation;
    };
}

