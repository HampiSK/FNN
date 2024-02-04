#pragma once

#include <memory>
#include <optional>
#include <vector>
#include <string>

#include "../Edge/Edge.hpp"
#include "NeuronStrategyInterface.hpp"

namespace fnn
{
    /**
     * @enum NeuronType
     * @brief Enumerates the types of neurons in a neural network
     *
     * Defines the role of a neuron within the network: as an input, hidden, or output neuron, or undefined.
     */
    enum class NeuronType
    {
        Unknown = -1, ///< Represents an undefined or uninitialized neuron
        Input,        ///< Represents an input neuron
        Hidden,       ///< Represents a hidden neuron within the network
        Output,       ///< Represents an output neuron
    };

    namespace utility
    {
        /**
         * @brief Converts a NeuronType to its string representation
         * @param type [in] The NeuronType to convert
         * @return A string representing the NeuronType
         */
        std::string neuronTypeToString(const NeuronType type);
    }

    /**
     * @struct Neuron
     * @brief Represents a neuron within a neural network
     *
     * Encapsulates the properties and state of a neuron
     */
    struct Neuron final
    {
    public:
        NeuronType m_neuronType = NeuronType::Unknown; ///< Type of the neuron, defaulting to Unknown
        float m_value = 0; ///< Current value of the neuron
        float m_error = 0; ///< Current error of the neuron

        std::optional<float> m_target; ///< Target value for output neurons, if applicable
        std::optional<float> m_learningRate; ///< Learning rate for updating the neuron's weight, if applicable

        std::optional<std::vector<Edge>> m_headConnections; ///< Incoming connections from other neurons
        std::optional<std::vector<Edge>> m_tailConnections; ///< Outgoing connections to other neurons

        std::optional<std::shared_ptr<INeuronFunctionStrategy>> m_activationFunction; ///< Activation function strategy for this neuron
        std::optional<std::shared_ptr<INeuronErrorStrategy>> m_errorCalculation; ///< Error calculation strategy for this neuron
        std::optional<std::shared_ptr<INeuronValueStrategy>> m_valueCalculation; ///< Value calculation strategy for this neuron
        std::optional<std::shared_ptr<INeuronWeightStrategy>> m_weightCalculation; ///< Weight calculation strategy for this neuron
    };
}
