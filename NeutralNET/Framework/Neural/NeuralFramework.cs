using NeutralNET.Activation;
using NeutralNET.Matrices;
using NeutralNET.Models;
using NeutralNET.Utils;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NeutralNET.Framework.Neural;

public unsafe class NeuralFramework<TArch> where TArch : IArchitecture<TArch>
{
    private readonly NeuralNetworkConfig _config;
    private readonly TArch _architecture;
    private readonly TArch _gradientArchitecture;
    private readonly Random _rng;

    private uint[] _trainingOutputStrideMask = null!;
    private int[] _indices;
    private int _timestep = 1;

    public NeuralFramework(NeuralNetworkConfig config)
    {
        if (config.Architecture.Length <= 0)
        {
            throw new ArgumentException("Negative or empty hidden layers.");
        }

        _config = config;
        _architecture = TArch.Create(_config.Architecture);
        _gradientArchitecture = TArch.Create(_config.Architecture);
        _rng = new Random();
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

    public NeuralForward Run(IModel model)
    {
        var trainingInput = model.TrainingInput;
        var trainingOutput = model.TrainingOutput;
        _trainingOutputStrideMask = model.TrainingOutputStrideMask;

        _architecture.MatrixNeurons[0].CopyRowFrom(trainingInput, 0);

        RandomizeWeightsBiases();
        HandleTraining(trainingInput, trainingOutput);

        model.TrainingInput = _architecture.MatrixNeurons[0];

        return Forward;
    }

    public IEnumerable<NeuralMatrix> EnumerateEpochs(IModel model)
    {
        var trainingInput = model.TrainingInput;
        var trainingOutput = model.TrainingOutput;
        _trainingOutputStrideMask = model.TrainingOutputStrideMask;

        _architecture.MatrixNeurons[0].CopyRowFrom(trainingInput, 0);

        RandomizeWeightsBiases();

        //foreach (var item in HandleTrainingEpoch(trainingInput, trainingInput))
        //{
        //    yield return item;
        //}

        foreach (var item in HandleTrainingEpoch(trainingInput, trainingOutput))
        {
            yield return item;
        }
    }

    public IEnumerable<NeuralMatrix> RunEpoch(IModel model)
    {
        var trainingInput = model.TrainingInput;
        var trainingOutput = model.TrainingOutput;
        _trainingOutputStrideMask = model.TrainingOutputStrideMask;

        _architecture.MatrixNeurons[0].CopyRowFrom(trainingInput, 0);

        RandomizeWeightsBiases();

        return HandleTrainingEpoch(trainingInput, trainingOutput);
    }

    private void HandleTraining(NeuralMatrix trainingInput, NeuralMatrix trainingOutput)
    {
        _indices = [.. Enumerable.Range(0, trainingInput.Rows)];

        var batchProcessCount = 0;
        var stopWatch = Stopwatch.StartNew();
        var orderedBatchesView = GetOrderedBatchView(trainingInput, trainingOutput);

        for (var epoch = 0; epoch < _config.Epochs; epoch++)
        {
            float loss = 0;
            var totalExamples = 0;
            ProcessOrderedBatchesView(orderedBatchesView, ref batchProcessCount, ref loss, ref totalExamples);
           
            loss /= totalExamples;

            if (epoch % orderedBatchesView.BatchCount is 0)
            {
                DisplayEpochResult(stopWatch.Elapsed, batchProcessCount, loss, epoch);
            }
        }
    }

    private IEnumerable<NeuralMatrix> HandleTrainingEpoch(NeuralMatrix trainingInput, NeuralMatrix trainingOutput)
    {
        _indices = [.. Enumerable.Range(0, trainingInput.Rows)];

        var batchProcessCount = 0;
        var stopWatch = Stopwatch.StartNew();
        var orderedBatchesView = GetOrderedBatchView(trainingInput, trainingOutput);

        for (var epoch = 0; epoch < _config.Epochs; epoch++)
        {
            float loss = 0;
            var totalExamples = 0;

            ProcessOrderedBatchesView(orderedBatchesView, ref batchProcessCount, ref loss, ref totalExamples);

            loss /= totalExamples;

            if (epoch % _config.BatchSize is 0)
            {
                DisplayEpochResult(stopWatch.Elapsed, batchProcessCount, loss, epoch);
            }

            yield return Forward();
        }
    }

    private OrderedBatchesView GetOrderedBatchView(NeuralMatrix trainingInput, NeuralMatrix trainingOutput)
    {
        var gradientArchitecture = new Architecture(_config.Architecture);
        _indices = [.. Enumerable.Range(0, trainingInput.Rows)];

        var orderedBatchesView = new OrderedBatchesView(_indices, trainingInput, trainingOutput, _config.BatchSize);

        return orderedBatchesView;
    }

    private void DisplayEpochResult(TimeSpan elapsed, int batchProcessCount, float loss, int epoch)
    {
        var batchesPerSecond = batchProcessCount / elapsed.TotalSeconds;
        var lossToPercent = 100.0 * (1.0 - Math.Min(loss, 1.0));

        var rLoss = loss > 0 ? 1 + float.Log10(loss) / 8 : 0;
        var gLoss = 1 - rLoss;

        var rReadyLoss = Math.Clamp((int)(rLoss * 255), 0, 255);
        var gReadyLoss = Math.Clamp((int)(gLoss * 255), 0, 255);

        var result = $"Epoch ({epoch}/{_config.Epochs}) Accuracy: {lossToPercent:F5}% Loss:{loss} BPS:{batchesPerSecond}/s TP:{elapsed}";
        result = result.WithColor(System.Drawing.Color.FromArgb(255, rReadyLoss, gReadyLoss, 0));

        Console.WriteLine(result);
    }


    private void ProcessOrderedBatchesView(
        OrderedBatchesView 
        orderedBatchesView, 
        ref int batchProcessCount, 
        ref float loss,
        ref int totalExamples)
    {
        if (_config.WithShuffle)
        {
            _rng.Shuffle(_indices);
        }

        foreach (var batch in orderedBatchesView)
        {
            loss += ProcessBatch(batch);
            totalExamples += batch.ActualSize;
            batchProcessCount++;
        }
    }

    private float ProcessBatch(       
        OrderedBatchView batch)
    {
        BackPropagate(batch);
        Learn();

        return Loss(batch);
    }

    private void Learn()
    {
        float beta1 = _config.Beta1;
        float beta2 = _config.Beta2;
        float epsilon = _config.Epsilon;
        float lr = _config.LearningRate;
        float wd = _config.WeightDecay;

        for (var i = 0; i < _architecture.Count; i++)
        {
            UpdateAdamMomentsVectorized(
                _architecture.MatrixMWeights[i], _architecture.MatrixVWeights[i],
                _gradientArchitecture.MatrixWeights[i], beta1, beta2);

            UpdateAdamMomentsVectorized(
                _architecture.MatrixMBiases[i], _architecture.MatrixVBiases[i],
                _gradientArchitecture.MatrixBiases[i], beta1, beta2);

            ApplyAdamUpdateVectorized(
                _architecture.MatrixWeights[i],
                _architecture.MatrixMWeights[i],
                _architecture.MatrixVWeights[i],
                lr, wd, beta1, beta2, epsilon, _timestep);

            ApplyAdamUpdateVectorized(
                _architecture.MatrixBiases[i],
                _architecture.MatrixMBiases[i],
                _architecture.MatrixVBiases[i],
                lr, wd, beta1, beta2, epsilon, _timestep);
        }

        _timestep++;
    }

    private void LearnInternalVectorized(
    NeuralMatrix[] matrixes,
    NeuralMatrix[] gradientMatrixes,
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void UpdateAdamMomentsVectorized(
    NeuralMatrix mMatrix, NeuralMatrix vMatrix, NeuralMatrix gradient,
    float beta1, float beta2)
    {
        float* mPtr = mMatrix.Pointer;
        float* vPtr = vMatrix.Pointer;
        float* gPtr = gradient.Pointer;
        float* end = mPtr + mMatrix.AllocatedLength;

        var beta1Vec = Vector256.Create(beta1);
        var beta2Vec = Vector256.Create(beta2);
        var oneMinusBeta1 = Vector256.Create(1 - beta1);
        var oneMinusBeta2 = Vector256.Create(1 - beta2);

        if (Avx2.IsSupported)
        {
            for (; mPtr != end;)
            {
                var m = Avx.LoadVector256(mPtr);
                var v = Avx.LoadVector256(vPtr);
                var g = Avx.LoadVector256(gPtr);

                var newM = Avx.Add(Avx.Multiply(beta1Vec, m),
                              Avx.Multiply(oneMinusBeta1, g));

                var gSq = Avx.Multiply(g, g);
                var newV = Avx.Add(Avx.Multiply(beta2Vec, v),
                              Avx.Multiply(oneMinusBeta2, gSq));

                newM.Store(mPtr);
                newV.Store(vPtr);

                mPtr += 8; vPtr += 8; gPtr += 8;
            }
        }
        else
        {
            // Scalar implementation
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void ApplyAdamUpdateVectorized(
        NeuralMatrix param, NeuralMatrix m, NeuralMatrix v,
        float lr, float wd, float beta1, float beta2, float epsilon, int t)
    {
        float* p = param.Pointer;
        float* mPtr = m.Pointer;
        float* vPtr = v.Pointer;
        float* end = p + param.AllocatedLength;

        float beta1T = MathF.Pow(beta1, t);
        float beta2T = MathF.Pow(beta2, t);
        var mCorrVec = Vector256.Create(1 / (1 - beta1T));
        var vCorrVec = Vector256.Create(1 / (1 - beta2T));
        var lrVec = Vector256.Create(lr);
        var epsVec = Vector256.Create(epsilon);
        var wdVec = Vector256.Create(lr * wd);

        if (Avx2.IsSupported)
        {
            for (; p != end;)
            {
                var paramVec = Avx.LoadVector256(p);
                var mVec = Avx.LoadVector256(mPtr);
                var vVec = Avx.LoadVector256(vPtr);
                
                var mHat = Avx.Multiply(mVec, mCorrVec);
                var vHat = Avx.Multiply(vVec, vCorrVec);
                
                var sqrtV = Avx.Sqrt(vHat);
                var denom = Avx.Add(sqrtV, epsVec);
                var step = Avx.Divide(mHat, denom);
                step = Avx.Multiply(lrVec, step);

                var decay = Avx.Multiply(wdVec, paramVec);
                var newParam = Avx.Subtract(Avx.Subtract(paramVec, decay), step);

                newParam.Store(p);

                p += 8; mPtr += 8; vPtr += 8;
            }
        }
    }

    private void BackPropagate(       
        OrderedBatchView batch)
    {
        _gradientArchitecture.ZeroOut();

        int rowCount = 0;
        foreach (var (input, output) in batch)
        {
            NativeMemory.Copy(input, _architecture.MatrixNeurons[0].Pointer, sizeof(float) * (nuint)batch.InputStride);

            Forward();

            for (var j = 0; j < _architecture.Count; j++)
            {
                _gradientArchitecture.MatrixNeurons[j].Clear();
            }

            ComputeOutputLayer(output);
            PropagateToPreviousLayer();

            ++rowCount;
        }

        NormalizeGradientsVectorized(rowCount);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ComputeOutputLayer(float* outputPointer)
    {
        var aPtr = _architecture.MatrixNeurons[^1].Pointer;
        var bPtr = _gradientArchitecture.MatrixNeurons[^1].Pointer;
        var cPtr = outputPointer;
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

    private void PropagateToPreviousLayer()
    {
        for (int layerIdx = _architecture.Count; layerIdx > 0; layerIdx--)
        {
            var currentActivations = _architecture.MatrixNeurons[layerIdx];
            var currentErrors = _gradientArchitecture.MatrixNeurons[layerIdx];

            BackPropagateLayerVectorized(layerIdx, currentActivations, currentErrors);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void BackPropagateLayerVectorized(
    int layerIndex,    
    NeuralMatrix currentActivations,
    NeuralMatrix currentErrors)
    {
        bool isOutputLayer = layerIndex == _architecture.Count - 1;

        var prevNeuronGradients = _gradientArchitecture.MatrixNeurons[layerIndex - 1].Pointer;
        var gradientLayerIndexBias = _gradientArchitecture.MatrixBiases[layerIndex - 1].Pointer;
        var weightsGradient = _gradientArchitecture.MatrixWeights[layerIndex - 1].Pointer;

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

            var neuronGradient = CalculateNeuronGradient(activation, error, isOutputLayer);

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
                *cPtr += neuronGradient * *aPtr;
                *dPtr += neuronGradient * *bPtr;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void NormalizeGradientsVectorized(int rowNumber)
    {
        var divisorVec = Vector256.Create((float)rowNumber);
        var divisorScalar = (float)rowNumber;

        for (var i = 0; i < _gradientArchitecture.Count; i++)
        {
            NormalizeArray(_gradientArchitecture.MatrixWeights[i], divisorVec, divisorScalar);
            NormalizeArray(_gradientArchitecture.MatrixBiases[i], divisorVec, divisorScalar);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeArray(NeuralMatrix matrix, Vector256<float> divisorVec, float divisorScalar)
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

    private float CalculateNeuronGradient(float activation, float error, bool isOutput)
    {
        float gradient = isOutput
            ? Math.Max(activation * (1 - activation), 0.01f)
            : activation > 0 ? 1f : 0f;                     

        return 2 * Math.Clamp(error, -10f, 10f) * gradient;
    }

    private float Loss(OrderedBatchView batch)
    {
        var loss = 0f;

        var realFirstNeuronMatrix = _architecture.MatrixNeurons[0];
        var realLastNeuronMatrix = _architecture.MatrixNeurons[^1];

        var aPtr = realFirstNeuronMatrix.Pointer;
        var realFirstNeuronPtr = realLastNeuronMatrix.Pointer;

        var sumMask = Vector256.Create(_trainingOutputStrideMask).AsSingle();

        foreach (var pair in batch)
        {
            var bPtr = realFirstNeuronPtr;

            NativeMemory.Copy(pair.Input, aPtr, sizeof(float) * (nuint)batch.InputStride);
            Forward();

            var cPtr = pair.Output;
            var cEnd = cPtr + batch.OutputStride;

            var predicted = realLastNeuronMatrix;
            var batchLoss = 0f;

            var lossVec = Vector256<float>.Zero;

            for (; cPtr != cEnd; bPtr += Vector256<float>.Count, cPtr += Vector256<float>.Count)
            {
                var predVec = Vector256.LoadAligned(bPtr);
                var targetVec = Vector256.LoadAligned(cPtr);
                var diff = Avx.Subtract(predVec, targetVec);
                lossVec = Avx.Add(lossVec, Avx.Multiply(diff, diff));
            }

            lossVec = Avx.And(lossVec, sumMask);
            batchLoss += Vector256.Sum(lossVec);

            loss += batchLoss;
        }
            
        return loss;
    }

    private NeuralMatrix Forward()
    {
        var index = 0;

        while (true)
        {
            _architecture.MatrixNeurons[index].DotVectorized(_architecture.MatrixWeights[index], _architecture.MatrixNeurons[index + 1]);
            _architecture.MatrixNeurons[index + 1].SumVectorized(_architecture.MatrixBiases[index]);

            index++;

            if (index >= _architecture.Count)
            {
                ActivationFunctions.ApplySigmoidVectorized(_architecture.MatrixNeurons[^1]);
                break;
            }

            ActivationFunctions.ApplyReLUVectorized(_architecture.MatrixNeurons[index]);
        }

        return _architecture.MatrixNeurons[^1];
    }

    private void RandomizeWeightsBiases()
    {
        for (var i = 0; i < _architecture.Count; i++)
        {
            float scale = MathF.Sqrt(2.0f / _architecture.MatrixWeights[i].Rows);
            _architecture.MatrixWeights[i].Randomize(-scale, scale);
            _architecture.MatrixBiases[i].Clear();
        }
    }
}
