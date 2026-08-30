using NeutralNET.Matrices;

namespace NeutralNET.Framework.Convolutional;

public interface ICnnOptimizer : IDisposable
{
    void UpdateConvWeights(CnnMatrix weights, CnnMatrix biases, NeuralMatrix dW, NeuralMatrix dB);
    void UpdateDenseWeights(NeuralMatrix weights, NeuralMatrix biases, NeuralMatrix dW, NeuralMatrix dB);
}
