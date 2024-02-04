#pragma once

#include <memory>

namespace fnn
{
    struct Neuron;

    /**
     * @struct Edge
     * @brief Represents a connection between two neurons in a neural network
     *
     * This structure models the edge in a neural network graph, encapsulating the weight of the connection and pointers to the connected neurons
     */
    struct Edge final
    {
    public:
        float m_weight; ///< Weight associated with the edge
        std::shared_ptr<Neuron> m_head; ///< Shared pointer to the head neuron (input side)
        std::shared_ptr<Neuron> m_tail; ///< Shared pointer to the head neuron (input side)


        /**
         * @brief Constructor for Edge
         * @param weight [in] Weight of the edge
         * @param head [in] Shared pointer to the head neuron
         * @param tail [in] Shared pointer to the tail neuron
         */
        Edge(const float weight, const std::shared_ptr<Neuron> head, const std::shared_ptr<Neuron> tail);

        /**
         * @brief Equality comparison operator
         * @param other [in] Edge to compare with
         * @return true if edges are equal, false otherwise
         */
        bool operator==(Edge const&) const = default;
    };
}
