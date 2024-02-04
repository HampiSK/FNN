#pragma once

#include <vector>
#include <memory>

#include "../Edge/Edge.hpp"
#include "../Activation/ActivationStrategyInterface.hpp"
#include "NeuronStrategyInterface.hpp"

namespace fnn
{
    /// Concrete neuron strategies

    /**
     * @class NeuronErrorStrategy
     * @brief Implements error calculation strategies for neurons
     */
    class NeuronErrorStrategy final : public INeuronErrorStrategy
    {
    public:
        ~NeuronErrorStrategy() override = default;

        /**
         * @brief Calculates error based on neuron connections
         * @param tailEdges [in] Edges from output neurons
         * @param headEdges [in] Edges to input neurons
         * @param error [in] Current error
         * @return Calculated error
         */
        float CalculateError(const std::vector<Edge> &tailEdges, const std::vector<Edge> &headEdges, const float error) override;

        /**
         * @brief Calculates error based on target and actual values, applied only for output layer
         * @param target [in] Target value
         * @param actual [in] Actual neuron output
         * @return Calculated error
         */
        float CalculateError(const float target, const float actual) override;
    private:
        /**
         * @brief Helper to calculate error portion
         * @param weightSum [in] Sum of weights
         * @param connectedWeight [in] Weight of connected edge
         * @param error [in] Current error
         * @return Error portion
         */
        float CalculateErrorPortion(const float weightSum, const float connectedWeight, const float error) const;
    };

    /**
     * @class NeuronValueStrategy
     * @brief Implements value calculation for neurons
     */
    class NeuronValueStrategy final : public INeuronValueStrategy
    {
    public:
        ~NeuronValueStrategy() override = default;

        /**
         * @brief Calculates neuron value
         * @param headEdges [in] Edges to input neurons
         * @param activationFunction [in] Activation function
         * @return Calculated value
         */
        float CalculateValue(const std::vector<Edge> &headEdges, const std::shared_ptr<INeuronFunctionStrategy> activationFunction) override;
    };

    /**
     * @class NeuronWeightStrategy
     * @brief Implements weight update strategies for neurons
     */
    class NeuronWeightStrategy final : public INeuronWeightStrategy
    {
    public:
        ~NeuronWeightStrategy() override = default;

        /**
         * @brief Updates weights of connected edges
         * @param headEdges [in, out] Edges to input neurons
         * @param learningRate [in] Learning rate
         * @param error [in] Error value
         */
        void UpdateConectedWeights(std::vector<Edge> &headEdges, const float learningRate, const float error) override;
    };
}
