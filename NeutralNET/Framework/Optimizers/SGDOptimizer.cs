using NeutralNET.Framework.Neural;
using NeutralNET.Matrices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NeutralNET.Framework.Optimizers;

internal unsafe class SGDOptimizer<TArch>(
    NeuralNetworkConfig config,
    TArch architecture,
    TArch gradientArchitecture) : IOptimizer where TArch : IArchitecture<TArch>
{
    public void Learn()
    {
        for (var i = 0; i < architecture.Count; i++)
        {
            LearnInternalVectorized(architecture.MatrixWeights, gradientArchitecture.MatrixWeights, i);
            LearnInternalVectorized(architecture.MatrixBiases, gradientArchitecture.MatrixBiases, i);
        }
    }

    private void LearnInternalVectorized(
    NeuralMatrix[] matrixes,
    NeuralMatrix[] gradientMatrixes,
    int index)
    {
        var weightDecay = config.WeightDecay;
        float factor = 1.0f - config.LearningRate * weightDecay;

        float* aPtr = matrixes[index].Pointer;
        float* bPtr = gradientMatrixes[index].Pointer;
        float* aEnd = aPtr + matrixes[index].AllocatedLength;

        if (Avx2.IsSupported)
        {
            var factorVec = Vector512.Create(factor);
            var rateVec = Vector512.Create(-config.LearningRate);

            for (; aPtr != aEnd; aPtr += NeuralMatrix.Alignment, bPtr += NeuralMatrix.Alignment)
            {
                var aVec = Vector512.LoadAligned(aPtr);
                var bVec = Vector512.LoadAligned(bPtr);

                var result = Avx512F.FusedMultiplyAdd(bVec, rateVec, Avx512F.Multiply(aVec, factorVec));

                result.StoreAligned(aPtr);
            }
        }
        else
        {
            for (var i = 0; i < matrixes[index].AllocatedLength; i++)
            {
                aPtr[i] = aPtr[i] * factor - config.LearningRate * bPtr[i];
            }
        }
    }
}
