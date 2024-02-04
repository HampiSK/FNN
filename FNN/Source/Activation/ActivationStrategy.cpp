#include "ActivationStrategy.hpp"

#include <cmath>
#include <math.h>
#include <numeric>

using namespace fnn;

float EmptyActivationStrategy::Activation(const float input)
{
    return input;
}

float EmptyActivationStrategy::Derivation(const float activationOutput)
{
    return activationOutput;
}


ReLUStrategy::ReLUStrategy(const float threshold)
    : m_threshold(threshold)
{
}

float ReLUStrategy::Activation(const float input)
{
    return input >= m_threshold ? 1.0f : 0.0f;
}

float ReLUStrategy::Derivation(const float activationOutput)
{
    return activationOutput < 0.0f ? 0.0f : 1.0f;
}


float SigmoidStrategy::Activation(const float input)
{
    return 1.0f / (1.0f + (exp(-1.0f * input)));
}

float SigmoidStrategy::Derivation(const float activationOutput)
{
    return activationOutput * (1.0f - activationOutput);
}


float TanhStrategy::Activation(const float input)
{
    return std::tanh(input);
}

float TanhStrategy::Derivation(const float activationOutput)
{
    return 1.0f - activationOutput * activationOutput;
}


float LinearStrategy::Activation(const float input)
{
    return input;
}

float LinearStrategy::Derivation(const float activationOutput)
{
    return 1.0f;
}
