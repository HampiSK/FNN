#pragma once

#include <memory>
#include <vector>
#include <initializer_list>

#include "NGraph.hpp"
#include "../Edge/Edge.hpp"
#include "../Random/RandomStrategyInterface.hpp"
#include "../Random/RandomStrategyInterface.hpp"
#include "../Random/RandomStrategy.hpp"

namespace fnn
{
    /**
     * @class NNetwork
     * @brief Represents a neural network encapsulating a graph of neurons
     *
     * Facilitates building and training of the neural network using forward and backward propagation
     * Supports fitting to training data and making predictions on new data
     */
    class NNetwork final
    {
    public:
        std::shared_ptr<NGraph> m_network; ///< Graph structure representing the neural network


        NNetwork();
        /**
         * @brief Initializes the neural network with specified layer sizes and random strategy
         * @param layerSizes [in] Sizes of the neural network layers
         * @param randomStrategy [in] Strategy for random number generation, defaults to FastRandomStrategy
         */
        explicit NNetwork(const std::initializer_list<size_t> &layerSizes, const std::shared_ptr<IRandomStrategy> randomStrategy = std::make_shared<FastRandomStrategy>());


        /**
         * @brief Trains the neural network on given training data for a number of epochs
         * @param trainX [in] Input features for training
         * @param trainY [in] Target outputs for training
         * @param epochs [in] Number of training iterations, default is 10
         * @return True if training is successful, false otherwise
         */
        bool Fit(const std::vector<std::vector<float>> &trainX, const std::vector<std::vector<float>> &trainY, const size_t epochs = 10);

        /**
         * @brief Predicts the output for given input data
         * @param testX [in] Input features for prediction
         * @param output [out] Predicted outputs
         * @return True if prediction is successful, false otherwise
         */
        bool Predict(const std::vector<std::vector<float>> &testX, std::vector<std::vector<float>> &output);

    private:
        // Methods for internal use in the training and prediction processes

        // TODO: When I start hating my self, implement option to allow maximum number of allowed cycles

        /**
        * @brief Checks for cycles in the forward direction of the graph
        * @return True if a cycle is detected, false otherwise
        */
        bool HasCycleForward() const;

        /**
         * @brief Checks for cycles in the backward direction of the graph
         * @return True if a cycle is detected, false otherwise
         */
        bool HasCycleBackward() const;


        /**
         * @brief Propagates inputs forward through the network
         * @param x [in] Single set of input features
         * @return True if propagation is successful, false otherwise
         */
        bool ForwardPropagate(const std::vector<float> &x);
        /**
         * @brief Propagates errors backward through the network
         * @param y [in] Single set of target outputs
         * @return True if error propagation is successful, false otherwise
         */
        bool BackwardPropagateError(const std::vector<float> &y);

        /**
         * @brief Updates weights in the network based on back-propagated errors
         * @return True if weight update is successful, false otherwise
         */
        bool BackwardPropagateWeights();

        // Helper functions for setting up and traversing the network

        /**
         * @brief Sets inputs to the network and discovers connections for forward propagation
         * @param inputX [in] Single set of input features
         * @param nextLayer [out] Set of neurons in the next layer to be processed
         * @return True if set-up is successful, false otherwise
         */
        bool SetInputsAndDiscoverConnections(const std::vector<float> &inputX, std::unordered_set<std::shared_ptr<Neuron>> &nextLayer);

        /**
         * @brief Sets errors in the output layer and discovers connections for backward error propagation
         * @param target [in] Single set of target outputs
         * @param nextLayer [out] Set of neurons in the next layer to be processed for error propagation
         * @return True if errors are set successfully, false otherwise
         */
        bool SetErrorsAndDiscoverConnections(const std::vector<float> &target, std::unordered_set<std::shared_ptr<Neuron>> &nextLayer);

        /**
         * @brief Updates weights in the network and discovers connections for the next layer
         * @param nextLayer [out] Set of neurons in the next layer to be processed for weight update
         * @return True if weights are updated successfully, false otherwise
         */
        bool SetWeightsAndDiscoverConnections(std::unordered_set<std::shared_ptr<Neuron>> &nextLayer);
    };
}
