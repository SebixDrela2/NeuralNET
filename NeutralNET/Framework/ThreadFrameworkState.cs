using NeutralNET.Activation;
using NeutralNET.Framework.Neural;
using NeutralNET.Matrices;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using static NeutralNET.Activation.ActivationSelector;

namespace NeutralNET.Framework;

unsafe struct ThreadFrameworkStateUnsafeStuff()
{
    public float* TrainingInput;
    public float* TrainingOutput;
}

internal unsafe class ThreadFrameworkState<TArch> : IDisposable
    where TArch : IArchitecture<TArch>
{
    public readonly TArch Architecture;
    public readonly TArch GradientArchitecture;
    public readonly int[] Indices;
    public readonly NeuralNetworkConfig Config;
    private ThreadFrameworkStateUnsafeStuff Stuff;
    private ActivationFunctionCollection Activations;
    private (int Input, int Output) Stride;

    public NeuralMatrix Forward()
    {
        var index = 0;

        while (true)
        {
            Architecture.MatrixNeurons[index].DotVectorized(Architecture.MatrixWeights[index], Architecture.MatrixNeurons[index + 1]);
            Architecture.MatrixNeurons[index + 1].SumVectorized(Architecture.MatrixBiases[index]);

            index++;

            // if (index < BatchNormLayer.Count)
            // {
            //     NeuralWinder.ApplyBatchNormForward(Architecture.MatrixNeurons[index + 1], BatchNormLayer[index]);
            // }

            // if (index < Architecture.Count - 1)
            // {
            //     NeuralWinder.ApplyDropout(Architecture.MatrixNeurons[index], Config.DropoutRate);
            // }

            if (index >= Architecture.Count)
            {
                Activations.Output.Activation(Architecture.MatrixNeurons[^1]);
                break;
            }

            Activations.Hidden.Activation(Architecture.MatrixNeurons[index]);
        }

        return Architecture.MatrixNeurons[^1];
    }

    public unsafe void ComputeOutputLayer(float* trainingOutputPointer)
    {
        var archOutputPtr = Architecture.MatrixNeurons[^1].Pointer;
        var gradOutputErrorPtr = GradientArchitecture.MatrixNeurons[^1].Pointer;
        float* aEnd = archOutputPtr + Architecture.MatrixNeurons[^1].AllocatedLength;

        if (Avx2.IsSupported)
        {
            for (; archOutputPtr != aEnd; archOutputPtr += NeuralMatrix.Alignment, gradOutputErrorPtr += NeuralMatrix.Alignment, trainingOutputPointer += NeuralMatrix.Alignment)
            {
                var predVec = Vector256.LoadAligned(archOutputPtr);
                var targetVec = Vector256.LoadAligned(trainingOutputPointer);
                var diff = Avx.Subtract(predVec, targetVec);
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
        var derivativeFn = isOutput ? Activations.Output.Derivative : Activations.Hidden.Derivative;
        var gradient = derivativeFn(activation);

        return 2 * Math.Clamp(error, -100f, 100f) * gradient;
    }

    public static ThreadFrameworkState<TArch> ParallelLoopBody(int i, ParallelLoopState state, ThreadFrameworkState<TArch> self)
        => self.ParallelLoopBody(i, state);
    public ThreadFrameworkState<TArch> ParallelLoopBody(int i, ParallelLoopState _)
    {
        var index = Indices[i];
        // var offset = Offset + i;
        // var size = Size;
        var architecture = Architecture;
        var gradientArchitecture = GradientArchitecture;

        var ptrInput = (index * Stride.Input) + Stuff.TrainingInput;
        var ptrOutput = (index * Stride.Output) + Stuff.TrainingOutput;

        NativeMemory.Copy(ptrInput, Architecture.MatrixNeurons[0].Pointer, sizeof(float) * (nuint)Stride.Input);

        Forward();

        for (var j = 0; j < architecture.Count; j++)
        {
            gradientArchitecture.MatrixNeurons[j].Clear();
        }

        ComputeOutputLayer(ptrOutput);
        PropagateToPreviousLayer();

        return this;
    }

    public ThreadFrameworkState<TArch> Reuse(NeuralFramework<TArch> fx)
    {
        fx.Architecture.CopyTo(Architecture);
        fx.GradientArchitecture.CopyTo(GradientArchitecture);
        return this;
    }
    public static ThreadFrameworkState<TArch> Create(NeuralFramework<TArch> fx, FiniteBatchesView batch) => new(fx, batch);
    public ThreadFrameworkState(NeuralFramework<TArch> fx, FiniteBatchesView bv)
    {
        Architecture = fx.Architecture.Copy();
        GradientArchitecture = fx.GradientArchitecture.Copy();
        Indices = fx.Indices;
        Config = fx.Config;
        Activations = fx._activations;
        Stride = bv.Stride;
        Stuff = new()
        {
            TrainingInput = bv.TrainingInput,
            TrainingOutput = bv.TrainingOutput,
        };
    }

    public void Dispose()
    {
        Architecture.Dispose();
        GradientArchitecture.Dispose();
    }
}
