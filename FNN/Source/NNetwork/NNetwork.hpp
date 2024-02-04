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
    class NNetwork final
    {
    public:
        std::shared_ptr<NGraph> m_network;

        NNetwork();
        explicit NNetwork(const std::initializer_list<size_t> &layerSizes, const std::shared_ptr<IRandomStrategy> randomStrategy = std::make_shared<FastRandomStrategy>());

        bool Fit(const std::vector<std::vector<float>> &trainX, const std::vector<std::vector<float>> &trainY, const size_t epochs = 10);
        bool Predict(const std::vector<std::vector<float>> &testX, std::vector<std::vector<float>> &output);

    private:
        // TODO: When I start hating my self, implement option to allow maximum number of allowed cycles
        bool HasCycleForward() const;
        bool HasCycleBackward() const;

        bool ForwardPropagate(const std::vector<float> &x);
        bool BackwardPropagateError(const std::vector<float> &y);
        bool BackwardPropagateWeights();

        bool SetInputsAndDiscoverConnections(const std::vector<float> &inputX, std::unordered_set<std::shared_ptr<Neuron>> &nextLayer);
        bool SetErrorsAndDiscoverConnections(const std::vector<float> &target, std::unordered_set<std::shared_ptr<Neuron>> &nextLayer);
        bool SetWeightsAndDiscoverConnections(std::unordered_set<std::shared_ptr<Neuron>> &nextLayer);
    };
}
