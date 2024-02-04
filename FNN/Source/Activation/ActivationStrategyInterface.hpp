#pragma once

namespace fnn
{
    /**
     * @interface INeuronFunctionStrategy
     * @brief Interface for neuron activation functions and their derivatives
     *
     * Defines the structure for activation and derivative functions used in neural networks
     */
    class INeuronFunctionStrategy
    {
    public:
         /**
         * @brief Virtual destructor for interface
         */
        virtual ~INeuronFunctionStrategy() = default;


        /**
         * @brief Computes neuron activation
         * @param input [in] Input value to the neuron
         * @return Activation function output
         */
        virtual float Activation(const float input) = 0;

        /**
         * @brief Computes derivative of the activation function
         * @param activationOutput [in] Output of the activation function
         * @return Derivative at the given output value
         */
        virtual float Derivation(const float activationOutput) = 0;
    };
}
