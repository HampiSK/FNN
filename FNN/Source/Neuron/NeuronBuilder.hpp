#pragma once

#include <memory>

#include "Neuron.hpp"
#include "NeuronStrategyInterface.hpp"

namespace fnn
{
    /**
     * @class NeuronBuilder
     * @brief Facilitates the construction of Neuron objects using the Builder pattern
     *
     * Provides a fluent API to set various properties of a Neuron object and finally build it
     */
    class NeuronBuilder final
    {
    public:
        /**
         * @brief Creates a NeuronBuilder instance
         * @return A new NeuronBuilder object
         */
        static NeuronBuilder Create();

        /**
         * @brief Initializes a NeuronBuilder with a specific NeuronType properties
         * @param type [in] The type of neuron to create
         * @return A NeuronBuilder object with the specified neuron type set
         */
        static NeuronBuilder CreateAsType(const NeuronType type);


        /**
         * @brief Sets the neuron type
         * @param type [in] The NeuronType to set
         * @return The NeuronBuilder instance for chaining
         */
        NeuronBuilder AsType(const NeuronType type);


        /**
         * @brief Sets the target value for the neuron, default is 0.0
         * @param target [in] The target value
         * @return The NeuronBuilder instance for chaining
         */
        NeuronBuilder HasTarget(const float target = 0.0f);

        /**
         * @brief Sets the learning rate for the neuron, default is 0.2
         * @param learningRate [in] The learning rate
         * @return The NeuronBuilder instance for chaining
         */
        NeuronBuilder HasLearningRate(const float learningRate = 0.2f);

        /**
         * @brief Initializes head connections with a specified size, default is 0
         * @param size [in] The number of head connections
         * @return The NeuronBuilder instance for chaining
         */
        NeuronBuilder HasHeadConnection(const size_t size = 0);

        /**
         * @brief Initializes tail connections with a specified size, default is 0
         * @param size [in] The number of tail connections
         * @return The NeuronBuilder instance for chaining
         */
        NeuronBuilder HasTailConnection(const size_t size = 0);


        /**
         * @brief Sets the activation function strategy for the neuron
         * @param activationFunction [in] The activation function strategy
         * @return The NeuronBuilder instance for chaining
         */
        NeuronBuilder WithActivationFunction(const std::shared_ptr<INeuronFunctionStrategy> activationFunction);

        /**
         * @brief Sets the error calculation strategy for the neuron
         * @param errorCalculation [in] The error calculation strategy
         * @return The NeuronBuilder instance for chaining
         */
        NeuronBuilder WithErrorCalculation(const std::shared_ptr<INeuronErrorStrategy> errorCalculation);

        /**
         * @brief Sets the value calculation strategy for the neuron
         * @param valueCalculation [in] The value calculation strategy
         * @return The NeuronBuilder instance for chaining
         */
        NeuronBuilder WithValueCalculation(const std::shared_ptr<INeuronValueStrategy> valueCalculation);

        /**
         * @brief Sets the weight calculation strategy for the neuron
         * @param weightCalculation [in] The weight calculation strategy
         * @return The NeuronBuilder instance for chaining
         */
        NeuronBuilder WithWeightCalculation(const std::shared_ptr<INeuronWeightStrategy> weightCalculation);

        /**
         * @brief Builds the Neuron object with the specified properties and transfers ownership
         * @return A shared pointer to the constructed Neuron
         */
        std::shared_ptr<Neuron> Build();

    private:
        std::shared_ptr<Neuron> m_neuron = std::make_shared<Neuron>(); ///< Shared pointer to the neuron being built

        /**
         * @brief Hiding constructor, use Create() or CreateAsType() instead
         */
        NeuronBuilder() = default;

        /**
         * @brief Hiding constructor, use Create() or CreateAsType() instead
         */
        NeuronBuilder(NeuronBuilder&) = default;

        /**
         * @brief Hiding constructor, use Create() or CreateAsType() instead
         */
        NeuronBuilder(NeuronBuilder&&) = default;
    };
}
