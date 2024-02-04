#include "NeuronBuilder.hpp"

#include "ActivationStrategy.hpp"
#include "NeuronStrategy.hpp"
#include "../Edge/Edge.hpp"

using namespace fnn;

NeuronBuilder NeuronBuilder::Create()
{
    return NeuronBuilder();
}

NeuronBuilder NeuronBuilder::CreateAsType(const NeuronType type)
{
    switch (type)
    {
    case NeuronType::Input:
        return NeuronBuilder()
            .AsType(NeuronType::Input)
            .HasTailConnection()
            .WithErrorCalculation(std::make_shared<NeuronErrorStrategy>());
    case NeuronType::Output:
        return NeuronBuilder()
            .AsType(NeuronType::Output)
            .HasTarget()
            .HasLearningRate()
            .HasHeadConnection()
            .WithActivationFunction(std::make_shared<EmptyActivationStrategy>())
            .WithErrorCalculation(std::make_shared<NeuronErrorStrategy>())
            .WithValueCalculation(std::make_shared<NeuronValueStrategy>())
            .WithWeightCalculation(std::make_shared<NeuronWeightStrategy>());
    case NeuronType::Hidden:
        return NeuronBuilder()
            .AsType(NeuronType::Hidden)
            .HasLearningRate()
            .HasHeadConnection()
            .HasTailConnection()
            .WithActivationFunction(std::make_shared<EmptyActivationStrategy>())
            .WithErrorCalculation(std::make_shared<NeuronErrorStrategy>())
            .WithValueCalculation(std::make_shared<NeuronValueStrategy>())
            .WithWeightCalculation(std::make_shared<NeuronWeightStrategy>());
    default:
        return NeuronBuilder();
    }
    
}

NeuronBuilder NeuronBuilder::AsType(const NeuronType type)
{
    m_neuron->m_neuronType = type;
    return *this;
}

NeuronBuilder NeuronBuilder::HasTarget(const float target)
{
    m_neuron->m_target = target;
    return *this;
}

NeuronBuilder NeuronBuilder::HasLearningRate(const float learningRate)
{
    m_neuron->m_learningRate = learningRate;
    return *this;
}

NeuronBuilder NeuronBuilder::HasHeadConnection(const size_t size)
{
    m_neuron->m_headConnections = std::vector<Edge>();
    m_neuron->m_headConnections->reserve(size);
    return *this;
}

NeuronBuilder NeuronBuilder::HasTailConnection(const size_t size)
{
    m_neuron->m_tailConnections = std::vector<Edge>();
    m_neuron->m_tailConnections->reserve(size);
    return *this;
}

NeuronBuilder NeuronBuilder::WithActivationFunction(const std::shared_ptr<INeuronFunctionStrategy> activationFunction)
{
    m_neuron->m_activationFunction = activationFunction;
    return *this;
}

NeuronBuilder NeuronBuilder::WithErrorCalculation(const std::shared_ptr<INeuronErrorStrategy> errorCalculation)
{
    m_neuron->m_errorCalculation = errorCalculation;
    return *this;
}

NeuronBuilder NeuronBuilder::WithValueCalculation(const std::shared_ptr<INeuronValueStrategy> valueCalculation)
{
    m_neuron->m_valueCalculation = valueCalculation;
    return *this;
}

NeuronBuilder NeuronBuilder::WithWeightCalculation(const std::shared_ptr<INeuronWeightStrategy> weightCalculation)
{
    m_neuron->m_weightCalculation = weightCalculation;
    return *this;
}

std::shared_ptr<Neuron> NeuronBuilder::Build()
{
    return m_neuron;
}
