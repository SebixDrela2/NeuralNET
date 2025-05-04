using NeutralNET.Matrices;
using NeutralNET.Models;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NeutralNET.Framework;

public unsafe class NeuralFramework
{
    private readonly NeuralNetworkConfig _config;

    private MatrixBatchProcessor _batchProcessor = null!;
    private Architecture _architecture = null!;
    private uint[] _trainingOutputStrideMask = null!;

    public NeuralFramework(NeuralNetworkConfig config)
    {
        if (config.Architecture.Length <= 0)
        {
            throw new ArgumentException("Negative or empty architecture.");
        }

        _config = config;

        Initialize();
    }

    private void Initialize()
    {
        _architecture = new Architecture(_config.Architecture);       
        _batchProcessor = new MatrixBatchProcessor();
    }
    
    public void Print(string name)
    {
        Console.WriteLine($"{name} = [");

        for (var i = 0; i < _architecture.Count; i++)
        {
            _architecture.MatrixWeights[i].Print($"{nameof(_architecture.MatrixWeights)}[{i}]");
            _architecture.MatrixBiases[i].Print($"{nameof(_architecture.MatrixWeights)}[{i}]");
        }

        Console.WriteLine("]");
    }
    
    public void Run(IModel model)
    {
        var trainingInput = model.TrainingInput;
        var trainingOutput = model.TrainingOutput;
        _trainingOutputStrideMask = model.TrainingOutputStrideMask;

        _architecture.MatrixNeurons[0].CopyRowFrom(trainingInput, 0);

        RandomizeWeights();
        HandleTraining(trainingInput, trainingOutput);

        model.TrainingInput = _architecture.MatrixNeurons[0];
        model.Forward = Forward;
    }

    private void HandleTraining(Matrix trainingInput, Matrix trainingOutput)
    {
        var gradientArchitecture = new Architecture(_config.Architecture);

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
                 loss += ProcessBatch(gradientArchitecture, batch);
             
                 batchProcessCount++;
             }
             
             loss /= _config.BatchSize;
             
             if (epoch % _config.BatchSize is 0)
             {
                var elasped = stopWatch.Elapsed;
                var timePerSecond = batchProcessCount / elasped.TotalSeconds;

                Console.WriteLine($"Epoch ({epoch}/{_config.Epochs}) Loss: {loss} BPS:{timePerSecond:F2}/s TB:{batchProcessCount} TP:{elasped}");
             }
        }
    }

    private float ProcessBatch(
        Architecture gradientArchitecture, 
        IEnumerable<(MatrixRow Input, MatrixRow Output)> batch)
    {
        BackPropagate(gradientArchitecture, batch);
        Learn(gradientArchitecture);

        return Loss(batch);
    }

    private void Learn(Architecture gradient)
    {
        for (var i = 0; i < _architecture.Count; i++)
        {           
            LearnInternalVectorized(_architecture.MatrixWeights, gradient.MatrixWeights, i);
            LearnInternalVectorized(_architecture.MatrixBiases, gradient.MatrixBiases, i);
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
    Matrix[] matrixes,
    Matrix[] gradientMatrixes,
    int index)
    {
        var weightDecay = _config.WeightDecay;
        float factor = 1.0f - _config.LearningRate * weightDecay;

        float* aPtr = matrixes[index].Pointer;
        float* bPtr = gradientMatrixes[index].Pointer;
        float* aEnd = aPtr + matrixes[index].AllocatedLength;

        if (Avx2.IsSupported)
        {
            var factorVec = Vector256.Create(factor);
            var rateVec = Vector256.Create(-_config.LearningRate);

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
            for (var i = 0; i < matrixes[index].AllocatedLength; i++)
            {
                aPtr[i] = aPtr[i] * factor - _config.LearningRate * bPtr[i];
            }
        }
    }

    private void BackPropagate(
        Architecture gradient,
        IEnumerable<(MatrixRow Input, MatrixRow Output)> batch)
    {       
        gradient.ZeroOut();

        int rowCount = 0;
        foreach (var (input, output) in batch)
        {           
            input.CopyTo(_architecture.MatrixNeurons[0].Pointer);
            Forward();

            for (var j = 0; j < _architecture.Count ; j++)
            {
                gradient.MatrixNeurons[j].Clear();
            }

            ComputeOutputLayer(gradient, output);
            PropagateToPreviousLayer(gradient);

            ++rowCount;
        }

        NormalizeGradientsVectorized(gradient, rowCount);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ComputeOutputLayer(Architecture gradient, MatrixRow output)
    {
        var aPtr = _architecture.MatrixNeurons[^1].Pointer;
        var bPtr = gradient.MatrixNeurons[^1].Pointer;
        var cPtr = output.Pointer;
        float* aEnd = aPtr + _architecture.MatrixNeurons[^1].AllocatedLength;

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

    private void PropagateToPreviousLayer(Architecture gradient)
    {       
        for (int layerIdx = _architecture.Count; layerIdx > 0; layerIdx--)
        {
            var currentActivations = _architecture.MatrixNeurons[layerIdx];
            var currentErrors = gradient.MatrixNeurons[layerIdx];

            BackPropagateLayerVectorized(layerIdx, gradient, currentActivations, currentErrors);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void BackPropagateLayerVectorized(
    int layerIndex,
    Architecture gradient,
    Matrix currentActivations,
    Matrix currentErrors)
    {        
        var prevNeuronGradients = gradient.MatrixNeurons[layerIndex - 1].Pointer;
        var gradientLayerIndexBias = gradient.MatrixBiases[layerIndex - 1].Pointer;
        var weightsGradient = gradient.MatrixWeights[layerIndex - 1].Pointer;

        var lastRealNeuronMatrix = _architecture.MatrixNeurons[layerIndex - 1];
        var prevActivations = lastRealNeuronMatrix.Pointer;
        var prevActivationsEnd = prevActivations + lastRealNeuronMatrix.ColumnsStride;
        var realLayerIndexWeight = _architecture.MatrixWeights[layerIndex - 1];
      
        var neuronCount = currentActivations.UsedColumns;

        var weights = realLayerIndexWeight.Pointer;

        for (var neuronIdx = 0; neuronIdx < neuronCount; neuronIdx++)
        {
            var activation = currentActivations.Pointer[neuronIdx];
            var error = currentErrors.Pointer[neuronIdx];

            var neuronGradient = CalculateNeuronGradient(activation, error);

            gradientLayerIndexBias[neuronIdx] += neuronGradient;

            AccumulateVectorizedGradients(prevActivations, prevActivationsEnd, ref weights, ref weightsGradient, prevNeuronGradients, neuronGradient);           
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
    private void NormalizeGradientsVectorized(Architecture gradient, int rowNumber)
    {
        var divisorVec = Vector256.Create((float)rowNumber);
        var divisorScalar = (float)rowNumber;

        for (var i = 0; i < gradient.Count; i++)
        {
            NormalizeArray(gradient.MatrixWeights[i], divisorVec, divisorScalar);
            NormalizeArray(gradient.MatrixBiases[i], divisorVec, divisorScalar);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeArray(Matrix matrix, Vector256<float> divisorVec, float divisorScalar)
    {
        var ptr = matrix.Pointer;
        float* end = ptr + matrix.AllocatedLength;

        if (Avx2.IsSupported)
        {
            for (; ptr != end; ptr += Vector256<float>.Count)
            {
                var vec = Vector256.LoadAligned(ptr);
                vec = Avx.Divide(vec, divisorVec);
                vec.StoreAligned(ptr);
            }
        }
        else
        {
            for (; ptr < end; ptr++)
            {
                *ptr /= divisorScalar;
            }
        }
    }

    private float CalculateNeuronGradient(float activation, float error) => 2 * error * (activation > 0 ? 1 : 0);

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
        //     ComputeGradient(_architecture.MatrixWeights, gradient.MatrixWeights, trainingInput, trainingOutput, epsillon, loss, i);
        //     ComputeGradient(_architecture.MatrixBiases, gradient.MatrixBiases, trainingInput, trainingOutput, epsillon, loss, i);           
        // }
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

        var realFirstNeuronMatrix = _architecture.MatrixNeurons[0];
        var realLastNeuronMatrix = _architecture.MatrixNeurons[^1];

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
        if (_architecture.Count > 0)
        {
            var index = 0;

            while(true)
            {
                _architecture.MatrixNeurons[index].DotVectorized(_architecture.MatrixWeights[index], _architecture.MatrixNeurons[index + 1]);
                _architecture.MatrixNeurons[index + 1].Sum(_architecture.MatrixBiases[index]);

                index++;

                if (index >= _architecture.Count)
                {
                    _architecture.MatrixNeurons[^1].ApplySigmoidVectorized();
                    break;
                }

                _architecture.MatrixNeurons[index].ApplyReLUVectorized();
            }
        }
        
        return _architecture.MatrixNeurons[^1];
    }

    private void RandomizeWeights()
    {
        for (var i = 0; i < _architecture.Count; i++)
        {
            float scale = float.Sqrt(2f / _architecture.MatrixWeights[i].Rows);
            _architecture.MatrixWeights[i].Randomize(-scale, scale);
        }
    }
}
