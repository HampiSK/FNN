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
    class NGraph final
    {
    public:
        std::unordered_map<size_t, std::shared_ptr<Neuron>> m_matrix; // Adjacency list
        std::unordered_set<size_t> m_inputs; // Holding keys of input neurons
        std::unordered_set<size_t> m_outputs; // Holding keys of output neurons
        std::shared_ptr<IRandomStrategy> m_randomStrategy;

        explicit NGraph(const std::initializer_list<size_t> &layerSizes, const std::shared_ptr<IRandomStrategy> randomStrategy = std::make_shared<FastRandomStrategy>());

        size_t Size() const;

        std::shared_ptr<Neuron> GetNeuron(const size_t neuronKey) const;
        void AddNeuron(const size_t neuronKey, const std::shared_ptr<Neuron> neuron);

        bool AddSourceToDestinationHead(const size_t sourceKey, const size_t destinationKey);
        bool AddSourceToDestinationTail(const size_t sourceKey, const size_t destinationKey);

        void MapFunction(const std::shared_ptr<INeuronFunctionStrategy> activationFunction);
        void MapFunction(const std::shared_ptr<INeuronFunctionStrategy> activationFunction, const size_t layer);

        void MapLearningRate(const float learningRate);
        void MapLearningRate(const float learningRate, const size_t layer);

    private:
        void ConnectLayers(const std::initializer_list<size_t> &layerSizes);
    };
}
