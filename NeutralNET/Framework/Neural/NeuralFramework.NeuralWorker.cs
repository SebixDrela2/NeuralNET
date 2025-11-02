// using System.Runtime.CompilerServices;
// using System.Runtime.InteropServices;
// using System.Runtime.Intrinsics;
// using System.Runtime.Intrinsics.X86;
// using NeutralNET.Matrices;
// using NeutralNET.Unmanaged;

// namespace NeutralNET.Framework.Neural;

// public unsafe partial class NeuralFramework<TArch>
//     where TArch : IArchitecture<TArch>
// {
//     private unsafe class NeuralWorker(NeuralFramework<TArch> fx, TArch arch, TArch grad, TrainingPair trRow) : IDisposable
//     {
//         public TArch Arch => arch;
//         public TArch Grad => grad;
//         public NeuralWorker(NeuralFramework<TArch> fx, TrainingPair trainingPair)
//             : this(fx, fx.Architecture.Copy(), fx.GradientArchitecture.Copy(), trainingPair)
//         { }
//         public NeuralWorker Reuse(NeuralFramework<TArch> fx)
//         {
//             fx.Architecture.CopyTo(arch);
//             fx.GradientArchitecture.CopyTo(grad);
//             return this;
//         }

//         public void StepOnce(int i, ParallelLoopState _)
//         {
//             var index = fx.Indices[i];

//             var ptrInput = (index * fx.Stride.Input) + trRow.Input;
//             var ptrOutput = (index * fx.Stride.Output) + trRow.Output;

//             NativeMemory.Copy(ptrInput, fx.Architecture.MatrixNeurons[0].Pointer, sizeof(float) * (nuint)fx.Stride.Input);

//             Forward();

//             for (var j = 0; j < arch.Count; j++) grad.MatrixNeurons[j].Clear();

//             ComputeOutputLayer(ptrOutput);
//             PropagateToPreviousLayer();
//         }

//         public NeuralMatrix Forward()
//         {
//             int n = arch.Count;
//             for (var i = 0; ;)
//             {
//                 arch.MatrixNeurons[i].DotVectorized(arch.MatrixWeights[i], arch.MatrixNeurons[i + 1]);
//                 arch.MatrixNeurons[i + 1].SumVectorized(arch.MatrixBiases[i]);

//                 if (++i >= n) break;
//                 fx._activations.Hidden.Activation(arch.MatrixNeurons[i]);
//             }

//             fx._activations.Output.Activation(arch.MatrixNeurons[n]);
//             return arch.MatrixNeurons[n];
//         }
//         public unsafe void ComputeOutputLayer(float* trainingOutputPointer)
//         {
//             var archOutputPtr = arch.MatrixNeurons[^1].Pointer;
//             var gradOutputErrorPtr = grad.MatrixNeurons[^1].Pointer;
//             float* aEnd = archOutputPtr + arch.MatrixNeurons[^1].AllocatedLength;

//             for (;
//                 archOutputPtr != aEnd;
//                 archOutputPtr += NeuralMatrix.Alignment,
//                 gradOutputErrorPtr += NeuralMatrix.Alignment,
//                 trainingOutputPointer += NeuralMatrix.Alignment)
//             {
//                 var predVec = Vector256.LoadAligned(archOutputPtr);
//                 var targetVec = Vector256.LoadAligned(trainingOutputPointer);
//                 var diff = Avx.Subtract(predVec, targetVec);
//                 diff.StoreAligned(gradOutputErrorPtr);
//             }
//         }

//         public unsafe void PropagateToPreviousLayer()
//         {
//             for (int layerIndex = arch.Count; layerIndex > 0; --layerIndex)
//             {
//                 var currentActivations = arch.MatrixNeurons[layerIndex];
//                 var currentErrors = grad.MatrixNeurons[layerIndex];

//                 var previousLayerIndex = layerIndex - 1;
//                 var derivativeFn = (layerIndex == (arch.Count - 1))
//                     ? fx._activations.Output.Derivative
//                     : fx._activations.Hidden.Derivative;

//                 var prevArchNeurons = arch.MatrixNeurons[previousLayerIndex];
//                 var prevArchWeightsPtr = arch.MatrixWeights[previousLayerIndex].Pointer;

//                 var prevGradNeurons = grad.MatrixNeurons[previousLayerIndex].Pointer;
//                 var prevGradWeights = grad.MatrixWeights[previousLayerIndex].Pointer;
//                 var prevGradBiases = grad.MatrixBiases[previousLayerIndex].Pointer;

//                 var prevArchNeuronsPtr = prevArchNeurons.Pointer;
//                 var prevArchNeuronsPtrEnd = prevArchNeuronsPtr + prevArchNeurons.ColumnsStride;

//                 for (var i = 0; i < currentActivations.UsedColumns; ++i)
//                 {
//                     var activation = currentActivations.Pointer[i];
//                     var error = currentErrors.Pointer[i];

//                     var gradient = derivativeFn(activation);
//                     var neuronGradient = 2 * Math.Clamp(error, -100f, 100f) * gradient;

//                     prevGradBiases[i] += neuronGradient;

//                     NeuralFramework<TArch>.AccumulateVectorizedGradients(
//                         prevArchNeuronsPtr,
//                         prevArchNeuronsPtrEnd,
//                         ref prevArchWeightsPtr,
//                         ref prevGradWeights,
//                         prevGradNeurons,
//                         neuronGradient
//                     );
//                 }
//             }
//         }

//         public void Dispose()
//         {
//             arch.Dispose();
//             grad.Dispose();
//         }
//     }
// }
