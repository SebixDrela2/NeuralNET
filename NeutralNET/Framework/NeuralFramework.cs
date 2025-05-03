using NeutralNET.Matrices;
using NeutralNET.Models;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NeutralNET.Framework;

public unsafe class NeuralFramework
{
    private MatrixBatchProcessor _batchProcessor = null!;

    private Matrix[] _matrixNeurons = null!;
    private Matrix[] _matrixWeights = null!;
    private Matrix[] _matrixBiases = null!;
    private uint[] _trainingOutputStrideMask = null!;

    private readonly NeuralNetworkConfig _config;

    public readonly int Count;

    public NeuralFramework(NeuralNetworkConfig config)
    {
        if (config.Architecture.Length <= 0)
        {
            throw new ArgumentException("Negative or empty architecture.");
        }

        _config = config;    
        Count = config.Architecture.Length - 1;

        Initialize();
    }

    private void Initialize()
    {
        _matrixNeurons = new Matrix[_config.Architecture.Length];
        _matrixWeights = new Matrix[Count];
        _matrixBiases = new Matrix[Count];

        _matrixNeurons[0] = new Matrix(1, _config.Architecture[0]);

        for (var i = 1; i < _config.Architecture.Length; i++)
        {
            _matrixWeights[i - 1] = new Matrix(_config.Architecture[i], _matrixNeurons[i - 1].UsedColumns);
            _matrixBiases[i - 1] = new Matrix(1, _config.Architecture[i]);
            _matrixNeurons[i] = new Matrix(1, _config.Architecture[i]);
        }
        
        _batchProcessor = new MatrixBatchProcessor();
    }
    
    public void Print(string name)
    {
        Console.WriteLine($"{name} = [");

        for (var i = 0; i < Count; i++)
        {
            _matrixWeights[i].Print($"{nameof(_matrixWeights)}[{i}]");
            _matrixBiases[i].Print($"{nameof(_matrixWeights)}[{i}]");
        }

        Console.WriteLine("]");
    }
    
    public IModelRunner Run(NeuralFramework gradientFramework, IModel model)
    {
        var trainingInput = model.TrainingInput;
        var trainingOutput = model.TrainingOutput;
        _trainingOutputStrideMask = model.TrainingOutputStrideMask;

        _matrixNeurons[0].CopyRowFrom(trainingInput, 0);

        RandomizeWeights();

        HandleTraining(gradientFramework, trainingInput, trainingOutput);

        return new ModelRunner
        {
            Input = _matrixNeurons[0],
            Forward = Forward
        };
    }

    private void HandleTraining(NeuralFramework gradientFramework, Matrix trainingInput, Matrix trainingOutput)
    {
        var rng = new Random();
        int[] indices = [.. Enumerable.Range(0, trainingInput.Rows)];

        var batchProcessCount = 0;
        var stopWatch = Stopwatch.StartNew();

        for (var epoch = 0; epoch < _config.Epochs; epoch++)
        {
             float loss = 0;
             
             if (_config.WithShuffle)
             {
                rng.Shuffle(indices);               
             }
             
             var batches = _batchProcessor.GetBatches(
                 trainingInput,
                 trainingOutput,
                 indices,
                 trainingInput.Rows,
                 _config.BatchSize
             );
             
             foreach (var batch in batches)
             {            
                 loss += ProcessBatch(gradientFramework, batch);
             
                 batchProcessCount++;
             }
             
             loss /= _config.BatchSize;
             
             if (epoch % _config.BatchSize is 0)
             {
                var elasped = stopWatch.Elapsed;
                var timePerSecond = batchProcessCount / elasped.TotalSeconds;

                Console.WriteLine($"Epoch ({epoch}/{_config.Epochs}) Loss: {loss} BPC:{timePerSecond:F2}/s TB:{batchProcessCount} TP:{elasped}");
             }
        }
    }

    private float ProcessBatch(
        NeuralFramework gradientFramework, 
        IEnumerable<(MatrixRow Input, MatrixRow Output)> batch)
    {
        BackPropagate(gradientFramework, batch);
        Learn(gradientFramework);

        return Loss(batch);
    }

    private void Learn(NeuralFramework gradient)
    {
        for (var i = 0; i < Count; i++)
        {
            gradient._matrixWeights[i].Clamp(-1f, 1f);
            gradient._matrixBiases[i].Clamp(-1f, 1f);

            LearnInternalVectorized(_matrixWeights.AsSpan(), gradient._matrixWeights.AsSpan(), _config.LearningRate, i);
            LearnInternalVectorized(_matrixBiases.AsSpan(), gradient._matrixBiases.AsSpan(), _config.LearningRate, i);
        }
    }

    [Obsolete]
    private void LearnInternal(
    ReadOnlySpan<Matrix> matrixes,
    ReadOnlySpan<Matrix> gradientMatrixes,
    float rate,
    int index)
    {
        throw new NotSupportedException("Old Learn Internal Activated.");

        //var totalIndex = 0;

        //var a = rate;
        //var weightDecay = _config.WeightDecay;
        //var A = matrixes[index].Span;
        //var B = gradientMatrixes[index].Span;

        //var totalLength = matrixes[index].Rows * matrixes[index].Columns;

        //float factor = 1.0f - rate * weightDecay;

        //for (var i = 0; i < totalLength; i++)
        //{
        //    A[i] = A[i] - rate * (weightDecay * A[i] + B[i]);
        //    A[i] = A[i] * factor - rate * B[i];
        //}       
    }

    private void LearnInternalVectorized(
    ReadOnlySpan<Matrix> matrixes,
    ReadOnlySpan<Matrix> gradientMatrixes,
    float rate,
    int index)
    {
        var weightDecay = _config.WeightDecay;
        float factor = 1.0f - rate * weightDecay;

        float* aPtr = matrixes[index].Pointer;
        float* bPtr = gradientMatrixes[index].Pointer;
        float* aEnd = aPtr + matrixes[index].AllocatedLength;

        if (Avx2.IsSupported)
        {
            var factorVec = Vector256.Create(factor);
            var rateVec = Vector256.Create(-rate);

            for (; aPtr != aEnd; aPtr += Vector256<float>.Count, bPtr += Vector256<float>.Count)
            {
                var aVec = Vector256.LoadAligned(aPtr);
                var bVec = Vector256.LoadAligned(bPtr);

                var result = Fma.MultiplyAdd(bVec, rateVec, Avx.Multiply(aVec, factorVec));

                result.StoreAligned(aPtr);
            }
        }
        else
        {
            for (int i = 0; i < matrixes[index].AllocatedLength; i++)
            {
                aPtr[i] =
                    aPtr[i] * factor - rate * bPtr[i];
            }
        }
    }

    private void BackPropagate(
        NeuralFramework gradient,
        IEnumerable<(MatrixRow Input, MatrixRow Output)> batch)
    {       
        gradient.ZeroOut();

        int rowCount = 0;
        foreach (var (input, output) in batch)
        {           
            input.CopyTo(_matrixNeurons[0].Pointer);
            Forward();

            for (var j = 0; j < Count ; j++)
            {
                gradient._matrixNeurons[j].Clear();
            }

            ComputeOutputLayer(gradient, output);
            PropagateToPreviousLayer(gradient);

            ++rowCount;
        }

        NormalizeGradientsVectorized(gradient, rowCount);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ComputeOutputLayer(NeuralFramework gradient, MatrixRow output)
    {
        var aPtr = _matrixNeurons[Count].Pointer;
        var bPtr = gradient._matrixNeurons[Count].Pointer;
        var cPtr = output.Pointer;
        float* aEnd = aPtr + _matrixNeurons[Count].AllocatedLength;

        if (Avx2.IsSupported)
        {
            for (; aPtr != aEnd; aPtr += Vector256<float>.Count, bPtr += Vector256<float>.Count, cPtr += Vector256<float>.Count)
            {
                var predVec = Vector256.LoadAligned(aPtr);
                var targetVec = Vector256.LoadAligned(cPtr);
                var diff = Avx.Subtract(predVec, targetVec);
                diff.StoreAligned(bPtr);
            }
        }
        else
        {
            for (; aPtr < aEnd; ++aPtr, ++bPtr, ++cPtr)
            {
                *bPtr = *aPtr - *cPtr;
            }
        }
    }

    private void PropagateToPreviousLayer(NeuralFramework gradient)
    {       
        for (int layerIdx = Count; layerIdx > 0; layerIdx--)
        {
            var currentActivations = _matrixNeurons[layerIdx];
            var currentErrors = gradient._matrixNeurons[layerIdx];

            BackPropagateLayerVectorized(layerIdx, gradient, currentActivations, currentErrors);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void BackPropagateLayerVectorized(
    int layerIndex,
    NeuralFramework gradient,
    Matrix currentActivations,
    Matrix currentErrors)
    {
        var prevNeuronGradients = gradient._matrixNeurons[layerIndex - 1].Pointer;
        var gradientLayerIndexBias = gradient._matrixBiases[layerIndex - 1].Pointer;
        var gradientLayerIndexWeight = gradient._matrixWeights[layerIndex - 1];

        var lastRealNeuronMatrix = _matrixNeurons[layerIndex - 1];
        var prevActivations = lastRealNeuronMatrix.Pointer;
        var prevActivationsEnd = prevActivations + lastRealNeuronMatrix.ColumnsStride;
        var realLayerIndexWeight = _matrixWeights[layerIndex - 1];
      
        var neuronCount = currentActivations.UsedColumns;

        var weights = realLayerIndexWeight.Pointer;
        var weightsGradients = gradientLayerIndexWeight.Pointer;

        for (var neuronIdx = 0; neuronIdx < neuronCount; neuronIdx++)
        {
            var activation = currentActivations.Pointer[neuronIdx];
            var error = currentErrors.Pointer[neuronIdx];

            var neuronGradient = CalculateNeuronGradient(activation, error);

            gradientLayerIndexBias[neuronIdx] += neuronGradient;

            AccumulateVectorizedGradients(prevActivations, prevActivationsEnd, ref weights, ref weightsGradients, prevNeuronGradients, neuronGradient);           
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateVectorizedGradients(
        float* aPtr,
        float* aEnd,
        ref float* bPtr, 
        ref float* cPtr, 
        float* dPtr, 
        float neuronGradient)
    {
        var ngVec = Vector256.Create(neuronGradient);
        
        if (Avx2.IsSupported)
        {
            for (; aPtr != aEnd;
                aPtr += Vector256<float>.Count,
                bPtr += Vector256<float>.Count,
                cPtr += Vector256<float>.Count,
                dPtr += Vector256<float>.Count)
            {
                var paVec = Vector256.LoadAligned(aPtr);
                var wVec = Vector256.LoadAligned(bPtr);

                var wGrad = Avx.Multiply(ngVec, paVec);
                var pGrad = Avx.Multiply(ngVec, wVec);

                var existingWGrad = Vector256.LoadAligned(cPtr);
                var existingPGrad = Vector256.LoadAligned(dPtr);

                var grad = Avx.Add(existingWGrad, wGrad);
                grad.StoreAligned(cPtr);

                grad = Avx.Add(existingPGrad, pGrad);
                grad.StoreAligned(dPtr);
            }
        }
        else
        {
            for (; aPtr != aEnd; ++aPtr, ++bPtr, ++cPtr, ++dPtr)
            {
                *cPtr += neuronGradient * (*aPtr);
                *dPtr += neuronGradient * (*bPtr);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void NormalizeGradientsVectorized(NeuralFramework gradient, int rowNumber)
    {
        var divisorVec = Vector256.Create((float)rowNumber);
        var divisorScalar = (float)rowNumber;

        for (var i = 0; i < gradient.Count; i++)
        {
            NormalizeArray(gradient._matrixWeights[i], divisorVec, divisorScalar);
            NormalizeArray(gradient._matrixBiases[i], divisorVec, divisorScalar);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeArray(Matrix matrix, Vector256<float> divisorVec, float divisorScalar)
    {
        var ptr = matrix.Pointer;
        float* end = ptr + matrix.AllocatedLength;

        for (; ptr != end; ptr += Vector256<float>.Count)
        {
            var vec = Vector256.LoadAligned(ptr);
            vec = Avx.Divide(vec, divisorVec);
            vec.StoreAligned(ptr);          
        }

        //for (; i < data.Length; i++)
        //{
        //    data[i] /= divisorScalar;
        //}
    }

    private float CalculateNeuronGradient(float activation, float error)
    {
        return 2 * error * (activation > 0 ? 1 : 0);
    }

    [Obsolete]
    private void FiniteDifference(
        NeuralFramework gradient, 
        float epsillon, 
        Matrix trainingInput, 
        Matrix trainingOutput)
    {
        throw new NotSupportedException();
        // var loss = Loss(trainingInput, trainingOutput);

        // for (var i = 0; i < Count; i++)
        // {
        //     ComputeGradient(_matrixWeights, gradient._matrixWeights, trainingInput, trainingOutput, epsillon, loss, i);
        //     ComputeGradient(_matrixBiases, gradient._matrixBiases, trainingInput, trainingOutput, epsillon, loss, i);           
        // }
    }

    private void ZeroOut()
    {
        for (var i = 0; i < Count; i++)
        {
            _matrixNeurons[i].Clear();
            _matrixWeights[i].Clear();
            _matrixBiases[i].Clear();
        }
    }

    [Obsolete]
    private void ComputeGradient(
        in ArraySegment<Matrix> matrixes, 
        in ArraySegment<Matrix> gradientMatrixes, 
        in Matrix trainingInput,
        in Matrix trainingOutput,
        float epsillon,
        float loss,
        int index)
    {
        throw new NotSupportedException();
        // for (var j = 0; j < matrixes[index].Rows; j++)
        // {
        //     for (var k = 0; k < matrixes[index].Columns; k++)
        //     {
        //         var temp = matrixes[index].At(j, k);

        //         matrixes[index].Add(j, k, epsillon);

        //         var computedLoss = (Loss(trainingInput, trainingOutput) - loss) / epsillon;
        //         gradientMatrixes[index].Set(j, k, computedLoss);

        //         matrixes[index].Set(j, k, temp);
        //     }
        // }
    }

    private float Loss(IEnumerable<(MatrixRow Input, MatrixRow Output)> batch)
    {
        var totalLoss = 0f;
        var count = 0;

        var realFirstNeuronMatrix = _matrixNeurons[0];
        var realLastNeuronMatrix = _matrixNeurons[Count];

        var aPtr = realFirstNeuronMatrix.Pointer;
        var realFirstNeuronPtr = realLastNeuronMatrix.Pointer;

        var sumMask = Vector256.Create(_trainingOutputStrideMask).AsSingle();

        foreach (var pair in batch)
        {
            var bPtr = realFirstNeuronPtr;

            pair.Input.CopyTo(aPtr);
            Forward();

            var cPtr = pair.Output.Pointer;
            var cEnd = cPtr + pair.Output.Stride;

            var outputRow = pair.Output;
            var predicted = realLastNeuronMatrix;
            var batchLoss = 0f;
            var j = 0;

            var lossVec = Vector256<float>.Zero;
            //while (j <= outputRow.Columns - Vector256<float>.Count)
            //{
            //    var predVec = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(predicted), (nuint)j);
            //    var targetVec = outputRow.LoadVectorUnaligned(j);
            //    var diff = Avx.Subtract(predVec, targetVec);
            //    lossVec = Avx.Add(lossVec, Avx.Multiply(diff, diff));
            //    j += Vector256<float>.Count;
            //}

            //batchLoss += Vector256.Sum(lossVec);

            //for (; j < outputRow.Columns; j++)
            //{
            //    float diff = predicted[j] - outputRow.Span[j];
            //    batchLoss += diff * diff;
            //}
            
            for (; cPtr != cEnd; bPtr += Vector256<float>.Count, cPtr += Vector256<float>.Count)
            {
                var predVec = Vector256.LoadAligned(bPtr);
                var targetVec = Vector256.LoadAligned(cPtr);
                var diff = Avx.Subtract(predVec, targetVec);
                lossVec = Avx.Add(lossVec, Avx.Multiply(diff, diff));
            }

            lossVec = Avx.And(lossVec, sumMask);
            batchLoss += Vector256.Sum(lossVec);

            totalLoss += batchLoss;
            count++;
        }

        return totalLoss / count;
    }

    private Matrix Forward()
    {
        if (Count > 0)
        {
            var index = 0;

            while(true)
            {
                _matrixNeurons[index].DotVectorized(_matrixWeights[index], _matrixNeurons[index + 1]);
                _matrixNeurons[index + 1].Sum(_matrixBiases[index]);

                index++;

                if (index >= Count)
                {
                    _matrixNeurons[Count].ApplySigmoidVectorized();
                    break;
                }

                _matrixNeurons[index].ApplyReLUVectorized();
            }
        }
        
        return _matrixNeurons[Count];
    }

    private void RandomizeWeights()
    {
        for (var i = 0; i < Count; i++)
        {
            float scale = float.Sqrt(2f / _matrixWeights[i].Rows);
            _matrixWeights[i].Randomize(-scale, scale);
        }
    }
}
