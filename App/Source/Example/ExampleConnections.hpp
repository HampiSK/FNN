#pragma once

#include <iostream>
#include <vector>

#include "ActivationStrategy.hpp"
#include "NeuronStrategy.hpp"
#include "NeuronBuilder.hpp"
#include "Neuron.hpp"
#include "NNetwork.hpp"

/**
 * @function exampleConnections1
 * @brief Demonstrates the setup, training, and prediction phases of a neural network using the FNN library.
 *
 * This function illustrates a basic example of constructing a neural network with a specific layer configuration,
 * establishing direct connections between input and output layers, and applying an activation function and learning rate.
 * It then proceeds to train the network with provided data and epochs, followed by making predictions on the same data set.
 * The predicted values are displayed alongside the expected values for comparison.
 *
 * @param trainX A 2D vector containing the input features for training. Each inner vector represents one training example.
 * @param trainY A 2D vector containing the target outputs for each training example in trainX.
 * @param epochs The number of training iterations to perform over the training data set.
 *
 * Key Steps:
 * 1. Network Configuration: Defines a neural network with 3 layers (2 input neurons, 4 hidden neurons, 1 output neuron).
 * 2. Connection Setup: Directly connects input neurons to the output neuron, bypassing the hidden layer.
 * 3. Activation and Learning: Applies a ReLU activation function to all neurons and sets a uniform learning rate.
 * 4. Training: Trains the network on the provided data for a specified number of epochs.
 * 5. Prediction: Uses the trained network to predict outputs on the training data and displays the results.
 *
 * Note: This example uses direct input-output connections, which is unconventional and used here for demonstration purposes.
 */
void exampleConnections1(const std::vector<std::vector<float>> &trainX, const std::vector<std::vector<float>> &trainY, const size_t epochs)
{
    printf("%s\n", __FUNCTION__);

    // Create network with 3 layers -> 2 input, 4 hidden, 1 output
    auto fnn = fnn::NNetwork({ 2, 4, 1 });

    // Create mutual connection between input and output layer
    fnn.m_network->AddSourceToDestinationHead(0, 6);
    fnn.m_network->AddSourceToDestinationHead(1, 6);
    fnn.m_network->AddSourceToDestinationTail(0, 6);
    fnn.m_network->AddSourceToDestinationTail(1, 6);

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
}

/**
 * @function exampleConnections2
 * @brief Demonstrates advanced connection setup and training in a neural network using the FNN library.
 *
 * This function builds upon the basic neural network setup by creating multiple mutual connections between the input and output layers,
 * bypassing the hidden layer entirely. It then applies an activation function, sets a learning rate, trains the network on provided data,
 * and finally, uses the trained network to make predictions. The function showcases how to handle more complex connection patterns and
 * illustrates the training and prediction phases with such a setup.
 *
 * @param trainX A 2D vector of input features for training, where each inner vector represents one training example.
 * @param trainY A 2D vector of target outputs corresponding to each training example in trainX.
 * @param epochs The number of iterations to train the network over the provided data set.
 *
 * Key Steps:
 * 1. Network Configuration: Defines a neural network with 3 layers (2 input neurons, 4 hidden neurons, 1 output neuron).
 * 2. Advanced Connection Setup: Establishes multiple connections directly between input neurons and the output neuron, multiple times, creating a denser connection pattern.
 * 3. Activation and Learning: Applies the ReLU activation function to all neurons and sets a global learning rate of 0.2.
 * 4. Training: Trains the network with the specified training data and epochs.
 * 5. Prediction: Uses the trained network to predict outputs for the training data and displays the results.
 *
 * Note: The multiple direct connections from input to output neurons are unconventional and used here for demonstration purposes to show the flexibility of the network setup.
 */
void exampleConnections2(const std::vector<std::vector<float>>& trainX, const std::vector<std::vector<float>>& trainY, const size_t epochs)
{
    printf("%s\n", __FUNCTION__);

    // Create network with 3 layers -> 2 input, 4 hidden, 1 output
    auto fnn = fnn::NNetwork({ 2, 4, 1 });

    // Create multiple mutual connection between input and output layer
    for (size_t i = 0; i < 3; ++i)
    {
        fnn.m_network->AddSourceToDestinationHead(0, 6);
        fnn.m_network->AddSourceToDestinationHead(1, 6);
        fnn.m_network->AddSourceToDestinationTail(0, 6);
        fnn.m_network->AddSourceToDestinationTail(1, 6);
    }

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
}

/**
 * @function exampleConnections3
 * @brief Demonstrates the construction and usage of a custom neural network with specific inter-layer connections.
 *
 * This function exemplifies the process of manually constructing a neural network layer by layer, and neuron by neuron,
 * setting up specific connections between them. It uses the FNN library's capabilities to build a network with 2 input neurons,
 * 2 hidden neurons, and 1 output neuron, with a particular pattern of connectivity. After defining the network structure and
 * characteristics, the function proceeds to train the network on provided data and epochs and then uses the trained model to make predictions.
 * The function concludes by comparing the predicted values against the actual target values.
 *
 * @param trainX A 2D vector containing the input features for training, where each inner vector is one training sample.
 * @param trainY A 2D vector containing the target outputs for each training sample in trainX.
 * @param epochs The number of training iterations to perform.
 *
 * Key Steps:
 * 1. Network Configuration: Manually adds neurons to the network, specifying their types (Input, Hidden, Output).
 * 2. Connection Setup: Establishes specific connections between neurons to form a custom network topology.
 * 3. Activation and Learning: Assigns the ReLU activation function and a learning rate of 0.2 to all neurons.
 * 4. Training: Trains the network with the specified data and number of epochs.
 * 5. Prediction and Evaluation: Predicts outputs using the trained network on the training data and displays the predicted versus expected values.
 *
 * Note: The function demonstrates a hands-on approach to constructing and configuring a neural network, allowing for detailed control over its architecture and properties.
 */
void exampleConnections3(const std::vector<std::vector<float>>& trainX, const std::vector<std::vector<float>>& trainY, const size_t epochs)
{
    printf("%s\n", __FUNCTION__);

    // Create custom network with 3 layers -> 2 input, 2 hidden, 1 output
    auto fnn = fnn::NNetwork();
    fnn.m_network->AddNeuron(0, fnn::NeuronBuilder::CreateAsType(fnn::NeuronType::Input).Build());
    fnn.m_network->AddNeuron(1, fnn::NeuronBuilder::CreateAsType(fnn::NeuronType::Input).Build());
    fnn.m_network->AddNeuron(2, fnn::NeuronBuilder::CreateAsType(fnn::NeuronType::Hidden).Build());
    fnn.m_network->AddNeuron(3, fnn::NeuronBuilder::CreateAsType(fnn::NeuronType::Hidden).Build());
    fnn.m_network->AddNeuron(4, fnn::NeuronBuilder::CreateAsType(fnn::NeuronType::Output).Build());

    // Create mutual connection between layers as following:
    // Input0 <-> Hidden2 <-> Otuput4
    // Input1 <-> Hidden3 <-> Otuput4
    fnn.m_network->AddSourceToDestinationHead(0, 2);
    fnn.m_network->AddSourceToDestinationTail(2, 0);
    fnn.m_network->AddSourceToDestinationHead(1, 3);
    fnn.m_network->AddSourceToDestinationTail(3, 1);
    fnn.m_network->AddSourceToDestinationHead(2, 4);
    fnn.m_network->AddSourceToDestinationTail(4, 2);
    fnn.m_network->AddSourceToDestinationHead(3, 4);
    fnn.m_network->AddSourceToDestinationTail(4, 3);

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
}

/**
 * @function exampleConnections4
 * @brief Demonstrates expanding a neural network by adding a new neuron and creating custom connections.
 *
 * This function illustrates a more advanced usage of the FNN library by starting with a predefined neural network structure
 * and then expanding it by adding an additional hidden neuron. It showcases how to establish custom connections between the new neuron
 * and existing layers, including both input and output neurons. The network is then trained with the specified dataset and epochs, and
 * its performance is evaluated by making predictions on the training data. The function highlights the flexibility of the FNN library in
 * modifying network architectures and connections after initial construction.
 *
 * @param trainX A 2D vector containing the input features for training, with each inner vector representing a single training sample.
 * @param trainY A 2D vector containing the target outputs corresponding to each input vector in trainX.
 * @param epochs The number of training iterations to run.
 *
 * Key Steps:
 * 1. Initial Network Configuration: Constructs a neural network with a specified layer structure (2 input neurons, two layers of 2 hidden neurons each, and 1 output neuron).
 * 2. Network Expansion: Adds an additional hidden neuron to the network, showcasing the dynamic modification of the network's structure.
 * 3. Custom Connection Setup: Creates unique bidirectional connections between the new hidden neuron and both the input and output neurons, illustrating the customization of neuron connectivity.
 * 4. Activation and Learning Configuration: Applies the ReLU activation function to all neurons and sets a uniform learning rate across the network.
 * 5. Training: Trains the network on the provided dataset for the specified number of epochs.
 * 6. Prediction and Evaluation: Predicts the outputs using the trained network on the same training data and compares the predicted results with the expected values.
 *
 * Note: This example highlights the FNN library's capability to modify and customize neural network architectures post-initialization, allowing for intricate network designs and connectivity patterns.
 */
void exampleConnections4(const std::vector<std::vector<float>>& trainX, const std::vector<std::vector<float>>& trainY, const size_t epochs)
{
    printf("%s\n", __FUNCTION__);

    // Create network with 3 layers -> 2 input, 2 hidden, 2 hidden, 1 output
    auto fnn = fnn::NNetwork({ 2, 2, 2, 1 });

    // Expand network with new hidden neuron
    fnn.m_network->AddNeuron(7, fnn::NeuronBuilder::CreateAsType(fnn::NeuronType::Hidden).Build());

    // Create mutual connection between input and output for new neuron
    // Hidden7 <-> Input0
    // Hidden7 <-> Input1
    // Hidden7 <-> Otuput6
    fnn.m_network->AddSourceToDestinationHead(0, 7);
    fnn.m_network->AddSourceToDestinationTail(7, 0);
    fnn.m_network->AddSourceToDestinationHead(1, 7);
    fnn.m_network->AddSourceToDestinationTail(7, 1);
    fnn.m_network->AddSourceToDestinationHead(6, 7);
    fnn.m_network->AddSourceToDestinationTail(7, 6);

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
}

/**
 * @function exampleConnections5
 * @brief Demonstrates the addition of a specialized hidden neuron to an existing neural network and the customization of its properties and connections.
 *
 * This function takes a step further in neural network customization by adding a new hidden neuron with unique characteristics to a predefined network structure. It showcases the use of a neuron builder to configure the neuron's properties, such as its activation function, learning rate, and connections. The newly added neuron is integrated into the network through specific connections to both input and output neurons, demonstrating the flexibility in designing network topologies. The network is then trained on a given dataset for a specified number of epochs, followed by the evaluation of its predictive performance on the same dataset.
 *
 * @param trainX A 2D vector containing the input features for training, where each inner vector represents one training sample.
 * @param trainY A 2D vector containing the target outputs for each training sample in trainX.
 * @param epochs The number of iterations for which the network should be trained.
 *
 * Key Steps:
 * 1. Initial Network Configuration: Constructs a neural network with a predefined layer structure (2 input neurons, two layers of 2 hidden neurons each, and 1 output neuron).
 * 2. Neuron Customization and Expansion: Adds a new hidden neuron configured with specific properties, including a non-standard activation function (EmptyActivationStrategy), indicating it doesn't actively participate in error calculation or propagation.
 * 3. Custom Connection Setup: Establishes unique connections for the new neuron with both input and output layers, highlighting the capability to create complex and tailored network architectures.
 * 4. Activation and Learning Configuration: Applies the ReLU activation function to all original neurons in the network and sets a uniform learning rate, maintaining standard network behavior outside the newly added neuron.
 * 5. Training: Trains the network with the provided data and number of epochs, adjusting neuron weights based on the learning algorithm.
 * 6. Prediction and Evaluation: Uses the trained network to predict outcomes based on the training data, showcasing the network's ability to generalize from its training.
 *
 * Note: This example emphasizes the FNN library's support for advanced network customization, allowing for the integration of neurons with specialized roles within the network's overall architecture.
 */
void exampleConnections5(const std::vector<std::vector<float>>& trainX, const std::vector<std::vector<float>>& trainY, const size_t epochs)
{
    printf("%s\n", __FUNCTION__);

    // Create network with 3 layers -> 2 input, 2 hidden, 2 hidden, 1 output
    auto fnn = fnn::NNetwork({ 2, 2, 2, 1 });

    // Expand network with new hidden neuron which doesn't calculate error
    fnn.m_network->AddNeuron(7,
        fnn::NeuronBuilder::Create()
        .AsType(fnn::NeuronType::Hidden)
        .HasLearningRate()
        .HasHeadConnection()
        .HasTailConnection()
        .WithActivationFunction(std::make_shared<fnn::EmptyActivationStrategy>())
        .WithValueCalculation(std::make_shared<fnn::NeuronValueStrategy>())
        .WithWeightCalculation(std::make_shared<fnn::NeuronWeightStrategy>())
        .Build()
    );

    // Create mutual connection between input and output for new neuron
    // Hidden7 <-> Input0
    // Hidden7 <-> Input1
    // Hidden7 <-> Otuput6
    fnn.m_network->AddSourceToDestinationHead(0, 7);
    fnn.m_network->AddSourceToDestinationTail(7, 0);
    fnn.m_network->AddSourceToDestinationHead(1, 7);
    fnn.m_network->AddSourceToDestinationTail(7, 1);
    fnn.m_network->AddSourceToDestinationHead(6, 7);
    fnn.m_network->AddSourceToDestinationTail(7, 6);

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
}
