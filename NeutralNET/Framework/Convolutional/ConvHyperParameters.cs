using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public sealed record class ConvHyperParameters(
    CnnMatrix Weights,
    NeuralMatrix FlattenedWeights,
    CnnMatrix Biases,
    CnnMatrix PreAct,
    CnnMatrix PostAct,
    NeuralMatrix PoolIndices) : IDisposable
{
    public void SetBatchLimit(int batchSize)
    {
        PreAct.Batch = batchSize;
        PostAct.Batch = batchSize;
    }

    public void Dispose()
    {
        Weights.Dispose();
        FlattenedWeights.Dispose();
        Biases.Dispose();
        PreAct.Dispose();
        PostAct.Dispose();
        PoolIndices.Dispose();
    }
}
