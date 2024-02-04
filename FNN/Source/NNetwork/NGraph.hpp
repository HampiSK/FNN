#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <initializer_list>

#include "../Neuron/Neuron.hpp"
#include "../Neuron/NeuronStrategyInterface.hpp"
#include "../Random/RandomStrategyInterface.hpp"
#include "../Random/RandomStrategy.hpp"

namespace fnn
{
    /**
     * @class NGraph
     * @brief Represents a neural network graph
     *
     * Manages neurons and their connections, facilitating operations like adding neurons, connecting them, and applying functions across the network
     */
    class NGraph final
    {
    public:
        std::unordered_map<size_t, std::shared_ptr<Neuron>> m_matrix; ///< Adjacency list representing neuron connections
        std::unordered_set<size_t> m_inputs; ///< Keys of input neurons
        std::unordered_set<size_t> m_outputs; ///< Keys of output neurons
        std::shared_ptr<IRandomStrategy> m_randomStrategy; ///< Strategy for random number generation used in the graph


        /**
         * @brief Constructor initializing the neural graph
         * @param layerSizes [in] Sizes of each layer in the graph
         * @param randomStrategy [in] Strategy for random number generation, defaults to FastRandomStrategy
         */
        explicit NGraph(const std::initializer_list<size_t> &layerSizes, const std::shared_ptr<IRandomStrategy> randomStrategy = std::make_shared<FastRandomStrategy>());


        /**
         * @brief Returns the total number of neurons in the graph
         * @return Size of the neural graph
         */
        size_t Size() const;

        /**
         * @brief Retrieves a neuron by its key
         * @param neuronKey [in] Key of the neuron to retrieve
         * @return Shared pointer to the requested neuron
         */
        std::shared_ptr<Neuron> GetNeuron(const size_t neuronKey) const;

        /**
         * @brief Adds a neuron to the graph
         * @param neuronKey [in] Key to assign to the neuron
         * @param neuron [in] Shared pointer to the neuron to add
         */
        void AddNeuron(const size_t neuronKey, const std::shared_ptr<Neuron> neuron);


        /**
         * @brief Connects a source neuron to a destination neuron's head
         * @param sourceKey [in] Key of the source neuron
         * @param destinationKey [in] Key of the destination neuron
         * @return True if connection was successful, false otherwise
         */
        bool AddSourceToDestinationHead(const size_t sourceKey, const size_t destinationKey);

        /**
         * @brief Connects a source neuron to a destination neuron's tail
         * @param sourceKey [in] Key of the source neuron
         * @param destinationKey [in] Key of the destination neuron
         * @return True if connection was successful, false otherwise
         */
        bool AddSourceToDestinationTail(const size_t sourceKey, const size_t destinationKey);


        /**
         * @brief Applies an activation function to all neurons in the graph
         * @param activationFunction [in] Activation function to apply
         */
        void MapFunction(const std::shared_ptr<INeuronFunctionStrategy> activationFunction);

        /**
         * @brief Applies an activation function to all neurons in a specific layer
         * @param activationFunction [in] Activation function to apply
         * @param layer [in] Layer to which the function should be applied
         */
        void MapFunction(const std::shared_ptr<INeuronFunctionStrategy> activationFunction, const size_t layer);

        /**
         * @brief Sets a learning rate for all neurons in the graph
         * @param learningRate [in] Learning rate to set
         */
        void MapLearningRate(const float learningRate);

        /**
         * @brief Sets a learning rate for all neurons in a specific layer
         * @param learningRate [in] Learning rate to set
         * @param layer [in] Layer to which the learning rate should be applied
         */
        void MapLearningRate(const float learningRate, const size_t layer);

    private:
        /**
         * @brief Connects neurons across specified layers
         * @param layerSizes [in] Sizes of the layers to connect
         */
        void ConnectLayers(const std::initializer_list<size_t> &layerSizes);
    };
}
