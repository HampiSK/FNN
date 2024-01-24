#include "Edge.hpp"

using namespace fnn;

Edge::Edge(const float weight, const std::shared_ptr<Neuron> head, const std::shared_ptr<Neuron> tail)
    : m_weight(weight), m_head(head), m_tail(tail)
{
}
