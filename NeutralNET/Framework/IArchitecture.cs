using NeutralNET.Framework.Neural;
using NeutralNET.Matrices;
using NeutralNET.Models;

namespace NeutralNET.Framework;

public interface IArchitecture<TSelf> where TSelf : IArchitecture<TSelf>
{
    int Count { get; }

    NeuralMatrix[] MatrixNeurons { get; }
    NeuralMatrix[] MatrixWeights { get; }
    NeuralMatrix[] MatrixBiases { get; }

    public NeuralMatrix[] MatrixMWeights { get; }
    public NeuralMatrix[] MatrixVWeights { get; }
    public NeuralMatrix[] MatrixMBiases { get; }
    public NeuralMatrix[] MatrixVBiases { get; }

    static abstract TSelf Create(params ReadOnlySpan<int> architecture);

    void ZeroOut();
}
