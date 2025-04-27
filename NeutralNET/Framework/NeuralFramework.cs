using NeutralNET.Matrices;
using NeutralNET.Models;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NeutralNET.Framework;

public class NeuralFramework
{
    private MatrixBatchProcessor _batchProcessor = null!;

    private Matrix[] _matrixNeurons = null!;
    private Matrix[] _matrixWeights = null!;
    private Matrix[] _matrixBiases = null!;

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
            _matrixWeights[i - 1] = new Matrix(_matrixNeurons[i - 1].Columns, _config.Architecture[i]);
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
                 ProcessBatch(gradientFramework, batch, ref loss);
             
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

    private void ProcessBatch(
        NeuralFramework gradientFramework, 
        IEnumerable<(Memory<float> Input, Memory<float> Output)> batch,
        ref float loss)
    {
        BackPropagate(gradientFramework, batch);
        Learn(gradientFramework);

        loss += Loss(batch);
    }

    private void Learn(NeuralFramework gradient)
    {
        for (var i = 0; i < Count; i++)
        {
            gradient._matrixWeights[i].Clip(-1f, 1f);
            gradient._matrixBiases[i].Clip(-1f, 1f);

            LearnInternalVectorized(_matrixWeights.AsSpan(), gradient._matrixWeights.AsSpan(), _config.LearningRate, i);
            LearnInternalVectorized(_matrixBiases.AsSpan(), gradient._matrixBiases.AsSpan(), _config.LearningRate, i);
        }
    }

    private void LearnInternalA(
    ReadOnlySpan<Matrix> matrixes,
    ReadOnlySpan<Matrix> gradientMatrixes,
    float rate,
    int index)
    {
        var totalIndex = 0;

        var a = rate;
        var weightDecay = _config.WeightDecay;
        var A = matrixes[index].Data;
        var B = gradientMatrixes[index].Data;

        var totalLength = matrixes[index].Rows * matrixes[index].Columns;

        float factor = 1.0f - rate * weightDecay;

        for (var i = 0; i < totalLength; i++)
        {
            //A[i] = A[i] - rate * (weightDecay * A[i] + B[i]);
            A[i] = A[i] * factor - rate * B[i];
        }       
    }

    private void LearnInternalVectorized(
    ReadOnlySpan<Matrix> matrixes,
    ReadOnlySpan<Matrix> gradientMatrixes,
    float rate,
    int index)
    {
        var weightDecay = _config.WeightDecay;
        float factor = 1.0f - rate * weightDecay;

        var aSpan = matrixes[index].Data.AsSpan();
        var bSpan = gradientMatrixes[index].Data.AsSpan();

        ref float aRef = ref MemoryMarshal.GetReference(aSpan);
        ref float bRef = ref MemoryMarshal.GetReference(bSpan);
        int remaining = aSpan.Length;

        var factorVec = Vector256.Create(factor);
        var rateVec = Vector256.Create(-rate);

        while (remaining >= 8)
        {
            var aVec = Vector256.LoadUnsafe(ref aRef);
            var bVec = Vector256.LoadUnsafe(ref bRef);

            var result = Avx.Add(
                Avx.Multiply(aVec, factorVec),
                Avx.Multiply(bVec, rateVec)
            );

            result.StoreUnsafe(ref aRef);
            aRef = ref Unsafe.Add(ref aRef, 8);
            bRef = ref Unsafe.Add(ref bRef, 8);
            remaining -= 8;
        }

        for (int i = 0; i < remaining; i++)
        {
            Unsafe.Add(ref aRef, i) =
                Unsafe.Add(ref aRef, i) * factor - rate * Unsafe.Add(ref bRef, i);
        }
    }

    private void BackPropagate(
        NeuralFramework gradient,
        IEnumerable<(Memory<float> Input, Memory<float> Output)> batch)
    {       
        gradient.ZeroOut();

        int rowCount = 0;
        foreach (var pair in batch)
        {
            var inputRow = pair.Input.Span;
            var outputRow = pair.Output.Span;

            inputRow.CopyTo(_matrixNeurons[0].Span);
            Forward();

            for (var j = 0; j < Count ; j++)
            {
                gradient._matrixNeurons[j].Clear();
            }

            ComputeOutputLayer(gradient, outputRow);
            PropagateToPreviousLayer(gradient);

            ++rowCount;
        }

        NormalizeGradientsVectorized(gradient, rowCount);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ComputeOutputLayer(NeuralFramework gradient, Span<float> outputRow)
    {
        var realLastMatrixNeuron = _matrixNeurons[Count].Span;
        var gradientLastMatrixNeuron = gradient._matrixNeurons[Count].Span;

        int i = 0;

        while (i <= outputRow.Length - Vector256<float>.Count)
        {
            var predVec = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(realLastMatrixNeuron), (nuint)i);
            var targetVec = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(outputRow), (nuint)i);
            var diff = Avx.Subtract(predVec, targetVec);
            Vector256.StoreUnsafe(diff, ref MemoryMarshal.GetReference(gradientLastMatrixNeuron), (nuint)i);
            i += Vector256<float>.Count;
        }

        for (; i < outputRow.Length; i++)
        {
            gradientLastMatrixNeuron[i] = realLastMatrixNeuron[i] - outputRow[i];
        }
    }

    private void PropagateToPreviousLayer(NeuralFramework gradient)
    {       
        for (int layerIdx = Count; layerIdx > 0; layerIdx--)
        {            
            var currentActivations = _matrixNeurons[layerIdx].GetRowSpan(0);
            var currentErrors = gradient._matrixNeurons[layerIdx].GetRowSpan(0);

            BackPropagateLayer(layerIdx, gradient, currentActivations, currentErrors);
        }
    }

    private void BackPropagateLayer(
        int layerIndex,
        NeuralFramework gradient,
        ReadOnlySpan<float> currentActivations,
        ReadOnlySpan<float> currentErrors)
    {
        var gradientLayerIndexNeuron = gradient._matrixNeurons[layerIndex - 1];
        var gradientLayerIndexBias = gradient._matrixBiases[layerIndex - 1];
        var gradientLayerIndexWeight = gradient._matrixWeights[layerIndex - 1];

        var realLayerIndexNeuron = _matrixNeurons[layerIndex - 1];
        var realLayerIndexWeight = _matrixWeights[layerIndex - 1];

        for (int neuronIdx = 0; neuronIdx < currentActivations.Length; neuronIdx++)
        {
            float activation = currentActivations[neuronIdx];
            float error = currentErrors[neuronIdx];

            var neuronGradient = CalculateNeuronGradient(activation, error);

            gradientLayerIndexBias.Add(0, neuronIdx, neuronGradient);

            for (int prevNeuronIdx = 0; prevNeuronIdx < realLayerIndexNeuron.Columns; prevNeuronIdx++)
            {
                float prevActivation = realLayerIndexNeuron.At(0, prevNeuronIdx);
                gradientLayerIndexWeight.Add(
                    prevNeuronIdx,
                    neuronIdx,
                    neuronGradient * prevActivation);

                gradientLayerIndexNeuron.Add(
                    0,
                    prevNeuronIdx,
                    neuronGradient * realLayerIndexWeight.At(prevNeuronIdx, neuronIdx));
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
            NormalizeArray(gradient._matrixWeights[i].Span, divisorVec, divisorScalar);
            NormalizeArray(gradient._matrixBiases[i].Span, divisorVec, divisorScalar);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeArray(Span<float> data, Vector256<float> divisorVec, float divisorScalar)
    {
        int simdLength = Vector256<float>.Count;
        int i = 0;

        while (i <= data.Length - simdLength)
        {
            var vec = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(data), (nuint)i);
            vec = Avx.Divide(vec, divisorVec);
            vec.StoreUnsafe(ref MemoryMarshal.GetReference(data), (nuint)i);
            i += simdLength;
        }

        for (; i < data.Length; i++)
        {
            data[i] /= divisorScalar;
        }
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

    private float Loss(IEnumerable<(Memory<float> Input, Memory<float> Output)> batch)
    {
        float totalLoss = 0f;
        int count = 0;
        var realFirstNeuron = _matrixNeurons[0].Span;
        var realLastNeuron = _matrixNeurons[Count];

        foreach (var pair in batch)
        {
            pair.Input.Span.CopyTo(realFirstNeuron);
            Forward();

            var outputRow = pair.Output.Span;
            var predicted = realLastNeuron.Span;
            float batchLoss = 0f;
            int j = 0;

            var lossVec = Vector256<float>.Zero;
            while (j <= outputRow.Length - Vector256<float>.Count)
            {
                var predVec = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(predicted), (nuint)j);
                var targetVec = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(outputRow), (nuint)j);
                var diff = Avx.Subtract(predVec, targetVec);
                lossVec = Avx.Add(lossVec, Avx.Multiply(diff, diff));
                j += Vector256<float>.Count;
            }

            batchLoss += Vector256.Sum(lossVec);

            for (; j < outputRow.Length; j++)
            {
                float diff = predicted[j] - outputRow[j];
                batchLoss += diff * diff;
            }

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
                    _matrixNeurons[Count].ApplySigmoid();
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
