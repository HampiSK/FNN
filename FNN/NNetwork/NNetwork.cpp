#include "NNetwork.hpp"

#include <stack>
#include <ranges>

using namespace fnn;

NNetwork::NNetwork() : m_network(std::make_shared<NGraph>(std::initializer_list<size_t>()))
{
}

NNetwork::NNetwork(const std::initializer_list<size_t> &layerSizes, const std::shared_ptr<IRandomStrategy> randomStrategy)
    : m_network(std::make_shared<NGraph>(layerSizes, randomStrategy))
{
}

bool NNetwork::Fit(const std::vector<std::vector<float>> &trainX, const std::vector<std::vector<float>> &trainY, const size_t epochs)
{
    if (m_network == nullptr || m_network->Size() == 0 || trainX.size() != trainY.size())
    {
        // Cannot fit empty network or invalid training data size
        return false;
    }

    // Detected cyclic routes using DFS
    if (HasCycleForward() || HasCycleBackward())
    {
        return false;
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        for (size_t i = 0; i < trainX.size(); ++i)
        {
            if (! ForwardPropagate(trainX[i]))
            {
                return false;
            }
            if (! BackwardPropagateError(trainY[i]))
            {
                return false;
            }
            if (! BackwardPropagateWeights())
            {
                return false;
            }
        }
    }
    return true;
}

bool NNetwork::Predict(const std::vector<std::vector<float>> &testX, std::vector<std::vector<float>> &output)
{
    if (m_network == nullptr)
    {
        // Cannot continue with non existing network
        return false;
    }

    // Detected cyclic routes using DFS
    if (HasCycleForward())
    {
        return false;
    }

    output.clear();
    output.reserve(testX.size());

    for (const auto &inputVector : testX) 
    {
        if (! ForwardPropagate(inputVector))
        {
            return false;
        }

        std::vector<float> currentOutput;
        currentOutput.reserve(m_network->m_outputs.size());

        for (const auto &neuronID : m_network->m_outputs)
        {
            const auto neuron = m_network->GetNeuron(neuronID);
            if (neuron == nullptr)
            {
                // neuron cannot be nullptr, something went terribly wrong
                return false;
            }

            currentOutput.push_back(neuron->m_value);
        }

        output.push_back(std::move(currentOutput));
    }
    return true;
}

bool NNetwork::HasCycleForward() const
{
    // Using DFS to detect cycles
    constexpr const int NOT_VISITED = 0;
    constexpr const int VISITING = 1;
    constexpr const int VISITED = 2;

    std::unordered_map<std::shared_ptr<Neuron>, int> state;
    state.reserve(m_network->Size());

    for (const auto &inputId : m_network->m_inputs)
    {
        // Skip if already visited or in progress
        const auto neruon = m_network->GetNeuron(inputId);
        if (state[neruon] != NOT_VISITED)
        { 
            continue;
        }

        std::stack<std::shared_ptr<Neuron>> stack;
        stack.push(neruon);

        while (! stack.empty())
        {
            const std::shared_ptr<Neuron> currentNeruon = stack.top();

            if (state[currentNeruon] == NOT_VISITED)
            {
                state[currentNeruon] = VISITING;
            }
            else if (state[currentNeruon] == VISITING)
            {
                state[currentNeruon] = VISITED;
                stack.pop();
                continue;
            }

            // There is no where to go, last neuron reached
            if(! currentNeruon->m_tailConnections.has_value())
            {
                state[currentNeruon] = VISITED;
                stack.pop();
                continue;
            }

            bool hasUnvisitedChildren = false;
            for (const auto &tailEdge : currentNeruon->m_tailConnections.value())
            {
                const std::shared_ptr<Neuron> childNeruon = tailEdge.m_tail;
                if (state[childNeruon] == VISITING)
                { 
                    return true; // Cycle detected
                }

                if (state[childNeruon] == NOT_VISITED)
                { 
                    stack.push(childNeruon);
                    hasUnvisitedChildren = true;
                    break; // Early break after pushing the first unvisited child
                }
            }

            if (! hasUnvisitedChildren)
            {
                // If all children are visited or the node has no children,
                // mark it as visited and pop it from the stack
                state[currentNeruon] = VISITED;
                stack.pop();
            }
        }
    }

    return false; // No cycles found
}

bool NNetwork::HasCycleBackward() const
{
    // Using DFS to detect cycles
    constexpr const int NOT_VISITED = 0;
    constexpr const int VISITING = 1;
    constexpr const int VISITED = 2;

    std::unordered_map<std::shared_ptr<Neuron>, int> state;
    state.reserve(m_network->Size());

    for (const auto& oututId : m_network->m_outputs)
    {
        // Skip if already visited or in progress
        const auto neruon = m_network->GetNeuron(oututId);
        if (state[neruon] != NOT_VISITED)
        {
            continue;
        }

        std::stack<std::shared_ptr<Neuron>> stack;
        stack.push(neruon);

        while (!stack.empty())
        {
            const std::shared_ptr<Neuron> currentNeruon = stack.top();

            if (state[currentNeruon] == NOT_VISITED)
            {
                state[currentNeruon] = VISITING;
            }
            else if (state[currentNeruon] == VISITING)
            {
                state[currentNeruon] = VISITED;
                stack.pop();
                continue;
            }

            // There is no where to go, last neuron reached
            if (! currentNeruon->m_headConnections.has_value())
            {
                state[currentNeruon] = VISITED; // Mark as black (visited)
                stack.pop();
                continue;
            }

            bool hasUnvisitedChildren = false;
            for (const auto &tailEdge : currentNeruon->m_headConnections.value())
            {
                const std::shared_ptr<Neuron> childNeruon = tailEdge.m_head;
                if (state[childNeruon] == VISITING)
                {
                    return true; // Cycle detected
                }

                if (state[childNeruon] == NOT_VISITED)
                {
                    stack.push(childNeruon);
                    hasUnvisitedChildren = true;
                    break; // Early break after pushing the first unvisited child
                }
            }

            if (! hasUnvisitedChildren)
            {
                // If all children are visited or the node has no children,
                // mark it as visited and pop it from the stack
                state[currentNeruon] = VISITED;
                stack.pop();
            }
        }
    }

    return false; // No cycles found
}

bool NNetwork::ForwardPropagate(const std::vector<float> &x)
{
    // Using unordered set to skip already included neurons
    std::unordered_set<std::shared_ptr<Neuron>> currentLayer;
    if (! SetInputsAndDiscoverConnections(x, currentLayer))
    {
        // Cannot update next layer container
        return false;
    }

    std::unordered_set<std::shared_ptr<Neuron>> nextLayer;

    // Note: If currentLayer becomes empty midway due to all neurons being output neurons or lacking
    // the necessary properties, the loop will terminate
    while (! currentLayer.empty())
    {
        nextLayer.clear();

        for (const auto &neuron : currentLayer)
        {
            if (neuron == nullptr)
            {
                // Neuron cannot be nullptr, something went terribly wrong
                return false;
            }

            // Neuron needs to have following properties to continue
            if (! neuron->m_valueCalculation.has_value() ||
                ! neuron->m_headConnections.has_value() ||
                ! neuron->m_activationFunction.has_value() ||
                neuron->m_activationFunction.value() == nullptr)
            {
                continue;
            }

            // Update neurons value
            neuron->m_value = neuron->m_valueCalculation.value()->CalculateValue(
                neuron->m_headConnections.value(),
                neuron->m_activationFunction.value()
            );

            // When neuron is output or does not have tail connections, do not include it
            if (neuron->m_neuronType == NeuronType::Output || ! neuron->m_tailConnections.has_value())
            {
                continue;
            }

            // Add tail neuron's connections to nextLayer
            for (const auto & tailEdge : neuron->m_tailConnections.value())
            {
                nextLayer.insert(tailEdge.m_tail);
            }
        }
        std::swap(currentLayer, nextLayer);
    }
    return true;
}

bool NNetwork::BackwardPropagateError(const std::vector<float> &y)
{
    // Using unordered set to skip already included neurons
    std::unordered_set<std::shared_ptr<Neuron>> currentLayer;
    if (! SetErrorsAndDiscoverConnections(y, currentLayer))
    {
        // Cannot update next layer container
        return false;
    }

    std::unordered_set<std::shared_ptr<Neuron>> nextLayer;

    // Note: If currentLayer becomes empty midway due to all neurons being input neurons or lacking
    // the necessary properties, the loop will terminate
    while (! currentLayer.empty())
    {
        nextLayer.clear();

        for (const auto &neuron : currentLayer)
        {
            if (neuron == nullptr)
            {
                // Neuron cannot be nullptr, something went terribly wrong
                return false;
            }

            // Neuron needs to have following properties to continue
            if (! neuron->m_tailConnections.has_value() ||
                ! neuron->m_headConnections.has_value() ||
                ! neuron->m_errorCalculation.has_value() ||
                neuron->m_errorCalculation.value() == nullptr)
            {
                continue;
            }

            // Update error
            neuron->m_error = neuron->m_errorCalculation.value()->CalculateError(
                neuron->m_tailConnections.value(),
                neuron->m_headConnections.value(),
                neuron->m_error);

            // When neuron is input, do not include it
            if (neuron->m_neuronType == NeuronType::Input)
            {
                continue;
            }

            // Add head neuron's connections to nextLayer
            for (const auto &nextEdge : neuron->m_headConnections.value())
            {
                nextLayer.insert(nextEdge.m_head);
            }
        }
        std::swap(currentLayer, nextLayer);
    }
    return true;
}

bool NNetwork::BackwardPropagateWeights()
{
    // Using unordered set to skip already included neurons
    std::unordered_set<std::shared_ptr<Neuron>> currentLayer;
    if (! SetWeightsAndDiscoverConnections(currentLayer))
    {
        // Cannot update next layer container
        return false;
    }

    std::unordered_set<std::shared_ptr<Neuron>> nextLayer;

    // Note: If currentLayer becomes empty midway due to all neurons being input neurons or lacking
    // the necessary properties, the loop will terminate
    while (! currentLayer.empty())
    {
        nextLayer.clear();
        for (const auto &neuron : currentLayer)
        {
            if (neuron == nullptr)
            {
                // Neuron cannot be nullptr, something went terribly wrong
                return false;
            }

            // Neuron needs to have following properties to continue
            if (! neuron->m_headConnections.has_value() ||
                ! neuron->m_learningRate.has_value() ||
                ! neuron->m_weightCalculation.has_value() ||
                neuron->m_weightCalculation.value() == nullptr)
            {
                continue;
            }

            // Update weights for output layer
            neuron->m_weightCalculation.value()->UpdateConectedWeights(
                neuron->m_headConnections.value(),
                neuron->m_learningRate.value(),
                neuron->m_error);

            // When neuron is input, do not include it
            if (neuron->m_neuronType == NeuronType::Input)
            {
                continue;
            }

            // Add head neuron's connections to nextLayer
            for (const auto &headEdge : neuron->m_headConnections.value())
            {
                nextLayer.insert(headEdge.m_head);
            }
        }
        std::swap(currentLayer, nextLayer);
    }
    return true;
}

bool NNetwork::SetInputsAndDiscoverConnections(const std::vector<float> &inputX, std::unordered_set<std::shared_ptr<Neuron>> &nextLayer)
{
    if (inputX.size() != m_network->m_inputs.size())
    {
        // Input layer has different size than inserted inputs
        return false;
    }

    auto it = inputX.begin();
    for (const auto &inputNeuronKey : m_network->m_inputs)
    {
        const auto inputNeuron = m_network->GetNeuron(inputNeuronKey);
        if (inputNeuron == nullptr)
        {
            // Neuron cannot be nullptr, something went terribly wrong
            return false;
        }

        inputNeuron->m_value = *it;
        ++it;

        // If neuron doesn't have tail connections, ignore it
        if(inputNeuron->m_tailConnections.has_value())
        {
            // Insert all connected tail neurons
            for (const auto &tailEdge : inputNeuron->m_tailConnections.value())
            {
                nextLayer.insert(tailEdge.m_tail);
            }
        }
    }
    return true;
}

bool NNetwork::SetErrorsAndDiscoverConnections(const std::vector<float> &target, std::unordered_set<std::shared_ptr<Neuron>> &nextLayer)
{
    if (m_network->m_outputs.size() != target.size())
    {
        // Output layer has different size than inserted target
        return false;
    }

    auto it = target.begin();
    for (const auto &outputNeuronKey : m_network->m_outputs)
    {
        auto outputNeuron = m_network->GetNeuron(outputNeuronKey);
        if (outputNeuron == nullptr)
        {
            // Neuron cannot be nullptr, something went terribly wrong
            return false;
        }

        outputNeuron->m_target = *it;
        ++it;

        // Output neuron should consist from these properties
        if (outputNeuron->m_neuronType == NeuronType::Output &&
            outputNeuron->m_headConnections.has_value() &&
            outputNeuron->m_target.has_value() &&
            outputNeuron->m_errorCalculation.has_value() &&
            outputNeuron->m_errorCalculation.value() != nullptr)
        {
            // Calculate error for output layer
            outputNeuron->m_error = outputNeuron->m_errorCalculation.value()->CalculateError(outputNeuron->m_target.value(), outputNeuron->m_error);

            for (const auto &headEdges : outputNeuron->m_headConnections.value())
            {
                nextLayer.insert(headEdges.m_head);
            }
        }
    }
    return true;
}

bool NNetwork::SetWeightsAndDiscoverConnections(std::unordered_set<std::shared_ptr<Neuron>> &nextLayer)
{
    for (const auto& inputNeuronKey : m_network->m_outputs)
    {
        auto outputNeuron = m_network->GetNeuron(inputNeuronKey);
        if (outputNeuron == nullptr)
        {
            // Neuron cannot be nullptr, something went terribly wrong
            return false;
        }

        // Output neuron should consist from these properties
        if (outputNeuron->m_neuronType == NeuronType::Output &&
            outputNeuron->m_headConnections.has_value() &&
            outputNeuron->m_learningRate.has_value() &&
            outputNeuron->m_weightCalculation.has_value() &&
            outputNeuron->m_weightCalculation.value() != nullptr)
        {
            // Update weights for output layer
            outputNeuron->m_weightCalculation.value()->UpdateConectedWeights(
                outputNeuron->m_headConnections.value(),
                outputNeuron->m_learningRate.value(),
                outputNeuron->m_error);

            for (const auto &headEdge : outputNeuron->m_headConnections.value())
            {
                nextLayer.insert(headEdge.m_head);
            }
        }
    }
    return true;
}
