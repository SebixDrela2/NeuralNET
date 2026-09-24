using NeutralNET.Matrices;

namespace NeutralNET.Framework.Convolutional;

public interface ICnnOptimizer : IDisposable
{
    void Update(CnnMatrix weights, CnnMatrix biases, NeuralMatrix dW, NeuralMatrix dB);
    void Update(NeuralMatrix weights, NeuralMatrix biases, NeuralMatrix dW, NeuralMatrix dB);
}
