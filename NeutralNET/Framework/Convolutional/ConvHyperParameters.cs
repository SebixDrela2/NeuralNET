using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public sealed record class ConvHyperParameters(
    CnnMatrix Input,
    NeuralMatrix ColInput,
    CnnMatrix Weights,
    NeuralMatrix FlattenedWeights,
    CnnMatrix Biases,
    CnnMatrix PreAct,
    CnnMatrix PostAct,
    NeuralMatrix PoolIndices,
    CnnMatrix GradInput,
    CnnMatrix PreGrad,
    NeuralMatrix PreGradMatrix,
    NeuralMatrix DWeights,
    NeuralMatrix DBiases,
    NeuralMatrix Convolution,
    CnnMatrix InputGrad) : IDisposable
{
    public void SetBatchLimit(int batchSize)
    {
        PreAct.Batch = batchSize;
        PostAct.Batch = batchSize;
    }

    public void Dispose()
    {
        Input.Dispose();
        ColInput.Dispose();
        Weights.Dispose();
        FlattenedWeights.Dispose();
        Biases.Dispose();
        PreAct.Dispose();
        PostAct.Dispose();
        PoolIndices.Dispose();
        GradInput.Dispose();
        PreGrad.Dispose();
        PreGradMatrix.Dispose();
        DWeights.Dispose();
        DBiases.Dispose();
        Convolution.Dispose();
        InputGrad.Dispose();
    }
}
