#include <iostream>
#include <vector>

#include "Example/ExampleConnections.hpp"

int main()
{
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
    const size_t epochs = 2;

    // Show examples

    exampleConnections1(trainX, trainY, epochs);
    exampleConnections2(trainX, trainY, epochs);
    exampleConnections3(trainX, trainY, epochs);
    exampleConnections4(trainX, trainY, epochs);
    exampleConnections5(trainX, trainY, epochs);

    return 0;
}
