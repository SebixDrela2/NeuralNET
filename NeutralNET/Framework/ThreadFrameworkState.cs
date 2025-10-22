using NeutralNET.Framework.Neural;
using NeutralNET.Matrices;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using static NeutralNET.Activation.ActivationSelector;

namespace NeutralNET.Framework;

internal record ThreadFrameworkState<TArch>(
    TArch Architecture, 
    TArch GradientArchitecture,
    int Size,
    int Offset,
    FiniteBatchesView BatchesView,
    int[] indices,
    List<BatchNormLayer> BatchNormLayer,
    NeuralTuner NeuralWinder,
    NeuralNetworkConfig Config,
    ActivationFunction OutputActivation,
    ActivationFunction HiddenActivation,
    DerivativeFunction OutputDerivative,
    DerivativeFunction HiddenDerivative) 
    where TArch : IArchitecture<TArch>
{
    public NeuralMatrix Forward()
    {
        var index = 0;

        while (true)
        {
            Architecture.MatrixNeurons[index].DotVectorized(Architecture.MatrixWeights[index], Architecture.MatrixNeurons[index + 1]);
            Architecture.MatrixNeurons[index + 1].SumVectorized(Architecture.MatrixBiases[index]);

            index++;

            if (index < BatchNormLayer.Count)
            {
                NeuralWinder.ApplyBatchNormForward(Architecture.MatrixNeurons[index + 1], BatchNormLayer[index]);
            }

            if (index < Architecture.Count - 1)
            {
                NeuralWinder.ApplyDropout(Architecture.MatrixNeurons[index], Config.DropoutRate);
            }

            if (index >= Architecture.Count)
            {
                OutputActivation(Architecture.MatrixNeurons[^1]);
                break;
            }

            HiddenActivation(Architecture.MatrixNeurons[index]);
        }

        return Architecture.MatrixNeurons[^1];
    }

    public unsafe void ComputeOutputLayer(float* trainingOutputPointer)
    {
        var archOutputPtr = Architecture.MatrixNeurons[^1].Pointer;
        var gradOutputErrorPtr = GradientArchitecture.MatrixNeurons[^1].Pointer;
        float* aEnd = archOutputPtr + Architecture.MatrixNeurons[^1].AllocatedLength;

        if (Avx512F.IsSupported)
        {
            for (; archOutputPtr != aEnd; archOutputPtr += NeuralMatrix.Alignment, gradOutputErrorPtr += NeuralMatrix.Alignment, trainingOutputPointer += NeuralMatrix.Alignment)
            {
                var predVec = Vector512.LoadAligned(archOutputPtr);
                var targetVec = Vector512.LoadAligned(trainingOutputPointer);
                var diff = Avx512F.Subtract(predVec, targetVec);
                diff.StoreAligned(gradOutputErrorPtr);
            }
        }
        else
        {
            for (; archOutputPtr < aEnd; ++archOutputPtr, ++gradOutputErrorPtr, ++trainingOutputPointer)
            {
                *gradOutputErrorPtr = *archOutputPtr - *trainingOutputPointer;
            }
        }
    }

    public void PropagateToPreviousLayer()
    {
        for (int layerIdx = Architecture.Count; layerIdx > 0; layerIdx--)
        {
            var currentArchNeurons = Architecture.MatrixNeurons[layerIdx];
            var currentGradErrors = GradientArchitecture.MatrixNeurons[layerIdx];

            BackPropagateLayerVectorized(layerIdx, currentArchNeurons, currentGradErrors);
        }
    }

   [MethodImpl(MethodImplOptions.AggressiveInlining)]
   private unsafe void BackPropagateLayerVectorized(
   int layerIndex,
   NeuralMatrix currentActivations,
   NeuralMatrix currentErrors)
   {
       var previousLayerIndex = layerIndex - 1;
       var isOutputLayer = layerIndex == (Architecture.Count - 1);

       var prevArchNeurons = Architecture.MatrixNeurons[previousLayerIndex];
       var prevArchWeightsPtr = Architecture.MatrixWeights[previousLayerIndex].Pointer;

       var prevGradNeurons = GradientArchitecture.MatrixNeurons[previousLayerIndex].Pointer;
       var prevGradWeights = GradientArchitecture.MatrixWeights[previousLayerIndex].Pointer;
       var prevGradBiases = GradientArchitecture.MatrixBiases[previousLayerIndex].Pointer;

       var prevArchNeuronsPtr = prevArchNeurons.Pointer;
       var prevArchNeuronsPtrEnd = prevArchNeuronsPtr + prevArchNeurons.ColumnsStride;

       for (var i = 0; i < currentActivations.UsedColumns; i++)
       {
           var activation = currentActivations.Pointer[i];
           var error = currentErrors.Pointer[i];

           var neuronGradient = CalculateNeuronGradient(activation, error, isOutputLayer);

           prevGradBiases[i] += neuronGradient;

           NeuralFramework<TArch>.AccumulateVectorizedGradients(prevArchNeuronsPtr, prevArchNeuronsPtrEnd, ref prevArchWeightsPtr, ref prevGradWeights, prevGradNeurons, neuronGradient);
       }
   }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float CalculateNeuronGradient(float activation, float error, bool isOutput)
    {
        var derivativeFn = isOutput ? OutputDerivative : HiddenDerivative;
        var gradient = derivativeFn(activation);

        return 2 * Math.Clamp(error, -100f, 100f) * gradient;
    }
}
