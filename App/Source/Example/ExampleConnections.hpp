#pragma once

#include <iostream>
#include <vector>

#include "ActivationStrategy.hpp"
#include "NeuronStrategy.hpp"
#include "NeuronBuilder.hpp"
#include "Neuron.hpp"
#include "NNetwork.hpp"

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