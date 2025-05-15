using NeutralNET.Matrices;

namespace NeutralNET.Framework;

public interface IArchitecture<TSelf> where TSelf : IArchitecture<TSelf>
{
    int Count { get; }

    NeuralMatrix[] MatrixNeurons { get; }
    NeuralMatrix[] MatrixWeights { get; }
    NeuralMatrix[] MatrixBiases { get; }

    static abstract TSelf Create(params ReadOnlySpan<int> architecture);
}
