#include "Neuron.hpp"

using namespace fnn;

std::string fnn::NeuronTypeToString(const NeuronType type)
{
    using enum NeuronType;

    switch (type)
    {
        case Unknown:
            return "Unknown";
        case Input:
            return "Input";
        case Hidden:
            return "Hidden";
        case Output:
            return "Output";
        default:
            return "Unknown";
    }
}
