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
    CnnMatrix InputGrad,
    NeuralMatrix GradPatchMat,
    CnnMatrix Pooled) : IDisposable
{
    public void Init(int i)
    {
        Input.DisplayName = $"Conv_Input[{i}]";
        ColInput.DisplayName = $"Conv_ColInput[{i}]";
        Weights.DisplayName = $"Conv_Weights[{i}]";
        FlattenedWeights.DisplayName = $"Conv_FlattenedWeights[{i}]";
        Biases.DisplayName = $"Conv_Biases[{i}]";
        PreAct.DisplayName = $"Conv_PreAct[{i}]";
        PostAct.DisplayName = $"Conv_PostAct[{i}]";
        PoolIndices.DisplayName = $"Conv_PoolIndices[{i}]";
        GradInput.DisplayName = $"Conv_GradInput[{i}]";
        PreGrad.DisplayName = $"Conv_PreGrad[{i}]";
        PreGradMatrix.DisplayName = $"Conv_PreGradMatrix[{i}]";
        DWeights.DisplayName = $"Conv_DWeights[{i}]";
        DBiases.DisplayName = $"Conv_DBiases[{i}]";
        Convolution.DisplayName = $"Conv_Convolution[{i}]";
        InputGrad.DisplayName = $"Conv_InputGrad[{i}]";
        GradPatchMat.DisplayName = $"Conv_GradPatchMat[{i}]";
    }
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
        GradPatchMat.Dispose();
        Pooled.Dispose();
    }
}
