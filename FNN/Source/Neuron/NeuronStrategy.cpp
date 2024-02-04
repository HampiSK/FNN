#include "neuronStrategy.hpp"

#include <numeric>

#include "Neuron.hpp"
#include "../Activation/ActivationStrategy.hpp"
#include "../Edge/Edge.hpp"


using namespace fnn;


float NeuronErrorStrategy::CalculateErrorPortion(const float weightSum, const float connectedWeight, const float error) const
{
    // Check for division by zero
    return weightSum == 0.0f ? 0.0f : connectedWeight / weightSum * error;
}

float NeuronErrorStrategy::CalculateError(const std::vector<Edge> &tailEdges, const std::vector<Edge> &headEdges, const float error)
{
    // Pre-calculate weightSum
    const float weightSum = std::accumulate(
        headEdges.begin(), headEdges.end(),
        0.0f,
        [] (float sum, const auto &edge) { return sum + edge.m_weight; }
    );

    // Calculate error sum
    return std::accumulate(
        tailEdges.begin(), tailEdges.end(),
        0.0f,
        [this, weightSum, error] (float acc, const Edge &edge)
        {
            return acc + CalculateErrorPortion(weightSum, edge.m_weight, error);
        }
    );
}

float NeuronErrorStrategy::CalculateError(const float target, const float actual)
{
    // Calculate actual
    return target - actual;
}

float NeuronValueStrategy::CalculateValue(const std::vector<Edge> &headEdges, const std::shared_ptr<INeuronFunctionStrategy> activationFunction)
{
    if (headEdges.empty() || activationFunction == nullptr)
    {
        return 0.0f;
    }

    const float total = std::transform_reduce(
        headEdges.begin(), headEdges.end(),
        0.0f,
        std::plus<>(),
        [] (const Edge &edge) { return edge.m_head->m_value * edge.m_weight; }
    );

    return activationFunction->Activation(total);
}

void NeuronWeightStrategy::UpdateConectedWeights(std::vector<Edge> &headEdges, const float learningRate, const float error)
{
    for (auto &edge : headEdges)
    {
        edge.m_weight -= learningRate * error * edge.m_head->m_value;
    }
}
