#pragma once

#include <memory>

namespace fnn
{
    struct Neuron;

    struct Edge final
    {
    public:
        float m_weight; // Associated weight with an edge
        std::shared_ptr<Neuron> m_head; // Facing to input layer
        std::shared_ptr<Neuron> m_tail; // Facing to output layer

        Edge(const float weight, const std::shared_ptr<Neuron> head, const std::shared_ptr<Neuron> tail);

        bool operator==(Edge const&) const = default;
    };
}
