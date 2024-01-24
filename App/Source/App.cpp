#include <iostream>
#include <ranges>
#include <iostream>
#include <vector>
#include <algorithm>

#include "ActivationStrategy.hpp"
#include "NeuronStrategy.hpp"
#include "NeuronBuilder.hpp"
#include "Neuron.hpp"
#include "NNetwork.hpp"

// More examples yet to come (lazy dev problem)

int main()
{
    // Create network with 3 layers -> 2 input, 4 hidden, 1 output
    auto fnn = fnn::NNetwork({ 2, 4, 1 });

    // Learning how to compute XOR (exclusive OR) function outputs
    const std::vector<std::vector<float>> trainX =
    {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},

    };
    const std::vector<std::vector<float>> trainY =
    {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f},
    };

    // Set number of complete cycles through the entire training dataset during the model's learning process
    const size_t epochs = 1;

    // Map ReLU activation function and learning rate for each neuron
    fnn.m_network->MapFunction(std::make_shared<fnn::ReLUStrategy>());
    fnn.m_network->MapLearningRate(0.2f);

    // Train network
    fnn.Fit(trainX, trainY, epochs);

    // Predict answer using same data used for training
    std::vector<std::vector<float>> answer;
    fnn.Predict(trainX, answer);

    // Display predictions
    for (size_t i = 0; i < answer.size(); ++i)
    {
        printf("Predicted: ");
        for (const auto& val : answer[i])
        {
            printf("%.2f ", val);
        }
        printf("Expected: %.2f\n", trainY[i].front());
    }

    return 0;
}
