#include "NGraph.hpp"

#include <ranges>

#include "../Neuron/Neuron.hpp"
#include "../Neuron/NeuronBuilder.hpp"
#include "../Neuron/NeuronStrategy.hpp"
#include "../Activation/ActivationStrategy.hpp"

using namespace fnn;

NGraph::NGraph(const std::initializer_list<size_t> &layerSizes, const std::shared_ptr<IRandomStrategy> randomStrategy)
    : m_randomStrategy(randomStrategy)
{
    size_t neuronID = 0;
    for (auto layerIt = layerSizes.begin(); layerIt != layerSizes.end(); ++layerIt)
    {
        const size_t layerSize = *layerIt;

        // Skip layers with zero neurons
        if (layerSize == 0)
        {
            continue;
        }

        const bool isInputLayer = (layerIt == layerSizes.begin());
        const bool isOutputLayer = (std::next(layerIt) == layerSizes.end());

        // Create neurons
        for (size_t j = 0; j < layerSize; ++j)
        {
            if (isInputLayer)
            {
                AddNeuron(neuronID, NeuronBuilder::CreateAsType(NeuronType::Input).Build());
                m_inputs.insert(neuronID);
            }
            else if (isOutputLayer)
            {
                AddNeuron(neuronID,NeuronBuilder::CreateAsType(NeuronType::Output).Build()
                );
                m_outputs.insert(neuronID);
            }
            else
            {
                AddNeuron(neuronID,
                    NeuronBuilder::CreateAsType(NeuronType::Hidden).Build()
                );
            }
            ++neuronID;
        }
    }
    ConnectLayers(layerSizes);
}

size_t NGraph::Size() const
{
    return m_matrix.size();
}

std::shared_ptr<Neuron> NGraph::GetNeuron(const size_t neuronKey) const
{
    const auto it = m_matrix.find(neuronKey);
    if (it != m_matrix.end())
    {
        return it->second;
    }
    else
    {
        return nullptr;
    }
}

void NGraph::AddNeuron(const size_t neuronKey, const std::shared_ptr<Neuron> neuron)
{
    // Handle invalid neuron
    if (neuron == nullptr)
    {
        return;
    }

    m_matrix[neuronKey] = neuron;
    if (neuron->m_neuronType == NeuronType::Input)
    {
        m_inputs.insert(neuronKey);
    }
    else if (neuron->m_neuronType == NeuronType::Output)
    {
        m_outputs.insert(neuronKey);
    }
}

bool NGraph::AddSourceToDestinationHead(const size_t sourceKey, const size_t destinationKey)
{
    const auto sourceNeuron = GetNeuron(sourceKey);
    const auto destinationNeuron = GetNeuron(destinationKey);

    if (sourceNeuron == nullptr || destinationNeuron == nullptr)
    {
        return false;
    }
    if (!destinationNeuron->m_headConnections.has_value())
    {
        return false;
    }
 
    const auto newEdge = Edge(m_randomStrategy->GetWeight(0.0f, 1.0f), sourceNeuron, destinationNeuron);
    auto &headConnections = destinationNeuron->m_headConnections.value();

    // Check if the edge already exists in the destination's head connections
    if (auto existingEdge = std::ranges::find(headConnections.begin(), headConnections.end(), newEdge); existingEdge == headConnections.end())
    {
        // If the edge doesn't exist, add it
        headConnections.push_back(newEdge);
    }
    else
    {
        // If it exists, update the edge
        existingEdge->m_tail = destinationNeuron;
        existingEdge->m_head = sourceNeuron;
    }
    return true;
}

bool NGraph::AddSourceToDestinationTail(const size_t sourceKey, const size_t destinationKey)
{
    const auto sourceNeuron = GetNeuron(sourceKey);
    const auto destinationNeuron = GetNeuron(destinationKey);

    if (sourceNeuron == nullptr || destinationNeuron == nullptr)
    {
        return false;
    }
    if (! destinationNeuron->m_tailConnections.has_value())
    {
        return false;
    }

    const auto newEdge = Edge(m_randomStrategy->GetWeight(0.0f, 1.0f), destinationNeuron, sourceNeuron);
    auto &tailConnections = destinationNeuron->m_tailConnections.value();

    // Check if the edge already exists in the source's tail connections
    if (auto existingEdge = std::ranges::find(tailConnections.begin(), tailConnections.end(), newEdge); existingEdge == tailConnections.end())
    {
        // If the edge doesn't exist, add it
        tailConnections.push_back(newEdge);
    }
    else
    {
        // If it exists, update the edge
        existingEdge->m_tail = sourceNeuron;
        existingEdge->m_head = destinationNeuron;
    }
    return true;
}

void NGraph::MapFunction(const std::shared_ptr<INeuronFunctionStrategy> activationFunction)
{
    // Handle invalid activation function
    if (activationFunction == nullptr)
    {
        return;
    }

    for (const auto &[neuronID, neuron] : m_matrix)
    {
        if (neuron->m_activationFunction.has_value())
        {
            neuron->m_activationFunction = activationFunction;
        }
    }
}

void NGraph::MapFunction(const std::shared_ptr<INeuronFunctionStrategy> activationFunction, const size_t layer)
{
    // Handle invalid activation function
    if (activationFunction == nullptr)
    {
        return;
    }

    // Using unordered set to skip already included neurons
    std::unordered_set<std::shared_ptr<Neuron>> currentLayer;
    std::unordered_set<std::shared_ptr<Neuron>> nextLayer;

    for (const auto neuronKey : m_inputs)
    {
        const auto &neuron = GetNeuron(neuronKey);
        if (neuron == nullptr)
        {
            continue;
        }
        // When first layer update neurons directly and exit
        if (layer == 0)
        {
            neuron->m_activationFunction = activationFunction;
        }
        else
        {
            currentLayer.insert(neuron);
        }
    }

    // Iterate over until correct layer was found or it is possible to do so
    for (size_t i = 0; ! currentLayer.empty(); ++i)
    {
        nextLayer.clear();

        for (const auto &neuron : currentLayer)
        {
            if (neuron == nullptr)
            {
                continue;
            }

            // Found correct layer
            if (i == layer)
            {
                neuron->m_activationFunction = activationFunction;
                continue;
            }

            // Add tail neuron's connections to nextLayer
            for (const auto &tailEdge : neuron->m_tailConnections.value())
            {
                nextLayer.insert(tailEdge.m_tail);
            }

        }
        std::swap(currentLayer, nextLayer);
    }
}


void NGraph::MapLearningRate(const float learningRate)
{
    for (const auto &[neuronID, neuron] : m_matrix)
    {
        if (neuron->m_learningRate.has_value())
        {
            neuron->m_learningRate = learningRate;
        }
    }
}

void NGraph::MapLearningRate(const float learningRate, const size_t layer)
{
    // Using unordered set to skip already included neurons
    std::unordered_set<std::shared_ptr<Neuron>> currentLayer;
    std::unordered_set<std::shared_ptr<Neuron>> nextLayer;

    for (const auto neuronKey : m_inputs)
    {
        const auto& neuron = GetNeuron(neuronKey);
        if (neuron == nullptr)
        {
            continue;
        }
        // When first layer update neurons directly and exit
        if (layer == 0)
        {
            neuron->m_learningRate = learningRate;
        }
        else
        {
            currentLayer.insert(neuron);
        }
    }

    // Iterate over until correct layer was found or it is possible to do so
    for (size_t i = 0; !currentLayer.empty(); ++i)
    {
        nextLayer.clear();

        for (const auto& neuron : currentLayer)
        {
            if (neuron == nullptr)
            {
                continue;
            }

            // Found correct layer
            if (i == layer)
            {
                neuron->m_learningRate = learningRate;
                continue;
            }

            // Add tail neuron's connections to nextLayer
            for (const auto& tailEdge : neuron->m_tailConnections.value())
            {
                nextLayer.insert(tailEdge.m_tail);
            }

        }
        std::swap(currentLayer, nextLayer);
    }
}

void NGraph::ConnectLayers(const std::initializer_list<size_t> &layerSizes)
{
    size_t neuronID = 0;

    for (auto layerIt = layerSizes.begin(); layerIt != layerSizes.end() && std::next(layerIt) != layerSizes.end(); ++layerIt)
    {
        // Skip layers with zero neurons
        if (*layerIt == 0)
        {
            continue;
        }

        const size_t nextLayerSize = *std::next(layerIt);

        // Skip connections to next layer if it has zero neurons
        if (nextLayerSize == 0)
        {
            continue;
        }

        const size_t currentLayerEnd = neuronID + *layerIt;
        const size_t nextLayerEnd = currentLayerEnd + nextLayerSize;

        for (; neuronID < currentLayerEnd; ++neuronID)
        {
            for (size_t connectedNeuronID = currentLayerEnd; connectedNeuronID < nextLayerEnd; ++connectedNeuronID)
            {
                // Create mutual connection between neurons
                AddSourceToDestinationHead(neuronID, connectedNeuronID);
                AddSourceToDestinationTail(connectedNeuronID, neuronID);
            }
        }
    }
}
