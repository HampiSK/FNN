#pragma once

#include "ActivationStrategyInterface.hpp"

namespace fnn
{
    /// Concrete activation functions

    /**
     * @class EmptyActivationStrategy
     * @brief An empty activation function implementation
     *
     * Provides no specific activation function
     */
    class EmptyActivationStrategy final : public INeuronFunctionStrategy
    {
    public:
        ~EmptyActivationStrategy() override = default;

        /**
         * @brief Activation function
         * @param input [in] Input value
         * @return Output value after activation
         */
        float Activation(const float input) override;

        /**
         * @brief Derivative of activation function
         * @param activationOutput [in] Activation function output
         * @return Derivative value
         */
        float Derivation(const float activationOutput) override;
    };

    /**
     * @class ReLUStrategy
     * @brief Rectified Linear Unit (ReLU) activation function implementation
     *
     * Implements the ReLU activation function with an adjustable threshold
     */
    class ReLUStrategy final : public INeuronFunctionStrategy
    {
    public:
        const float m_threshold; ///< Activation threshold for the ReLU function, default is 0.5

        /**
         * @brief Constructor with optional threshold parameter
         * @param threshold [in] Activation threshold, default is 0.5
         */
        explicit ReLUStrategy(const float threshold = 0.5f);
        ~ReLUStrategy() override = default;

        /**
         * @brief Activation function
         * @param input [in] Input value
         * @return Output value after activation
         */
        float Activation(const float input) override;

        /**
         * @brief Derivative of activation function
         * @param activationOutput [in] Activation function output
         * @return Derivative value
         */
        float Derivation(const float activationOutput) override;
    };

    /**
     * @class SigmoidStrategy
     * @brief Sigmoid activation function implementation
     *
     * Implements the Sigmoid activation function
     */
    class SigmoidStrategy final : public INeuronFunctionStrategy
    {
    public:
        ~SigmoidStrategy() override = default;

        /**
         * @brief Activation function
         * @param input [in] Input value
         * @return Output value after activation
         */
        float Activation(const float input) override;

        /**
         * @brief Derivative of activation function
         * @param activationOutput [in] Activation function output
         * @return Derivative value
         */
        float Derivation(const float activationOutput) override;
    };

    /**
     * @class TanhStrategy
     * @brief Hyperbolic tangent (tanh) activation function implementation
     *
     * Implements the tanh activation function
     */
    class TanhStrategy final : public INeuronFunctionStrategy
    {
    public:
        ~TanhStrategy() override = default;

        /**
         * @brief Activation function
         * @param input [in] Input value
         * @return Output value after activation
         */
        float Activation(const float input) override;

        /**
         * @brief Derivative of activation function
         * @param activationOutput [in] Activation function output
         * @return Derivative value
         */
        float Derivation(const float activationOutput) override;
    };

    /**
     * @class LinearStrategy
     * @brief Linear activation function implementation
     *
     * Implements a linear activation function
     */
    class LinearStrategy final : public INeuronFunctionStrategy
    {
    public:
        ~LinearStrategy() override = default;


        /**
         * @brief Activation function
         * @param input [in] Input value
         * @return Output value after activation
         */
        float Activation(const float input) override;

        /**
         * @brief Derivative of activation function
         * @param activationOutput [in] Activation function output
         * @return Derivative value
         */
        float Derivation(const float activationOutput) override;
    };

}
