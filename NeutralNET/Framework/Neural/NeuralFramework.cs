using NeutralNET.Activation;
using NeutralNET.Framework.Optimizers;
using NeutralNET.Matrices;
using NeutralNET.Models;
using NeutralNET.Utils;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using static NeutralNET.Activation.ActivationSelector;

namespace NeutralNET.Framework.Neural;

public unsafe class NeuralFramework<TArch> where TArch : IArchitecture<TArch>
{
    private static readonly ActivationSelector _activationSelector = new();

    private readonly NeuralTuner _neuralWinder;

    private readonly NeuralNetworkConfig _config;
    private readonly TArch _gradientArchitecture;
    private readonly IOptimizer _optimizer;
    private readonly Random _rng;

    private readonly ActivationFunction _hiddenActivation;
    private readonly ActivationFunction _outputActivation;

    private readonly DerivativeFunction _hiddenDerivative;
    private readonly DerivativeFunction _outputDerivative;

    private readonly List<BatchNormLayer> _batchNormLayers = new();

    private uint[] _trainingOutputStrideMask = null!;
    private int[] _indices;

    public readonly TArch Architecture;

    public NeuralFramework(NeuralNetworkConfig config)
    {
        if (config.Architecture.Length <= 0)
        {
            throw new ArgumentException("Negative or empty hidden layers.");
        }

        _config = config;
        _gradientArchitecture = TArch.Create(_config.Architecture);
        _rng = new Random();
        _neuralWinder = new NeuralTuner(_rng);

        _hiddenActivation = _activationSelector.GetActivation(_config.HiddenActivation);
        _outputActivation = _activationSelector.GetActivation(_config.OutputActivation);

        _hiddenDerivative = _activationSelector.GetDerivative(_config.HiddenActivation);

        _outputDerivative = _activationSelector.GetDerivative(_config.OutputActivation);

        Architecture = TArch.Create(_config.Architecture);

        for (int i = 0; i < _config.Architecture.Length - 1; i++)
        {
            _batchNormLayers.Add(new BatchNormLayer(_config.Architecture[i + 1]));
        }

        _optimizer = new OptimizerFactory<TArch>(config, Architecture, _gradientArchitecture)
            .GetOptimizer();
    }

    public void Print(string name)
    {
        Console.WriteLine($"{name} = [");

        for (var i = 0; i < Architecture.Count; i++)
        {
            Architecture.MatrixWeights[i].Print($"{nameof(Architecture.MatrixWeights)}[{i}]");
            Architecture.MatrixBiases[i].Print($"{nameof(Architecture.MatrixWeights)}[{i}]");
        }

        Console.WriteLine("]");
    }

    public NeuralForward Run(IModel model)
    {
        var trainingInput = model.TrainingInput;
        var trainingOutput = model.TrainingOutput;
        _trainingOutputStrideMask = model.TrainingOutput.StrideMasks;

        RandomizeWeightsBiases();
        HandleTraining(model);

        model.TrainingInput = Architecture.MatrixNeurons[0];

        return Forward;
    }

    public NeuralForward Run(IDynamicModel model)
    {
        _trainingOutputStrideMask = MatrixUtils.GetStrideMask(1);

        RandomizeWeightsBiases();
        HandleTraining(model);

        return Forward;
    }

    public IEnumerable<NeuralMatrix> EnumerateEpochs(IModel model)
    {
        var trainingInput = model.TrainingInput;
        var trainingOutput = model.TrainingOutput;
        _trainingOutputStrideMask = model.TrainingOutput.StrideMasks;

        RandomizeWeightsBiases();

        foreach (var item in HandleTrainingEpoch(trainingInput, trainingInput))
        {
            yield return item;
        }

        foreach (var item in HandleTrainingEpoch(trainingInput, trainingOutput))
        {
            yield return item;
        }
    }

    public IEnumerable<NeuralMatrix> RunEpoch(IModel model)
    {
        var trainingInput = model.TrainingInput;
        var trainingOutput = model.TrainingOutput;
        _trainingOutputStrideMask = model.TrainingOutput.StrideMasks;

        RandomizeWeightsBiases();

        return HandleTrainingEpoch(trainingInput, trainingOutput);
    }

    public NeuralMatrix Forward()
    {
        var index = 0;

        while (true)
        {
            Architecture.MatrixNeurons[index].DotVectorized(Architecture.MatrixWeights[index], Architecture.MatrixNeurons[index + 1]);
            Architecture.MatrixNeurons[index + 1].SumVectorized(Architecture.MatrixBiases[index]);

            index++;

            if (index < _batchNormLayers.Count)
            {
                _neuralWinder.ApplyBatchNormForward(Architecture.MatrixNeurons[index + 1], _batchNormLayers[index]);
            }

            if (index < Architecture.Count - 1)
            {
                _neuralWinder.ApplyDropout(Architecture.MatrixNeurons[index], _config.DropoutRate);
            }

            if (index >= Architecture.Count)
            {
                _outputActivation(Architecture.MatrixNeurons[^1]);
                break;
            }

            _hiddenActivation(Architecture.MatrixNeurons[index]);
        }

        return Architecture.MatrixNeurons[^1];
    }

    private void HandleTraining(IModel model)
    {
        var batchProcessCount = 0;
        var stopWatch = Stopwatch.StartNew();
        var orderedBatchesView = GetOrderedBatchView(model.TrainingInput, model.TrainingOutput);

        for (var epoch = 0; epoch < _config.Epochs; epoch++)
        {
            float loss = 0;
            var totalExamples = 0;
            ProcessOrderedBatchesView(orderedBatchesView, ref batchProcessCount, ref loss, ref totalExamples);

            loss /= totalExamples;

            DisplayEpochResult(stopWatch.Elapsed, batchProcessCount, loss, epoch);
        }
    }

    private void HandleTraining(IDynamicModel model)
    {
        var batchProcessCount = 0;
        var stopWatch = Stopwatch.StartNew();
        var orderedBatchesView = new InfiniteBatchesView(model, _config.BatchSize);

        for (var epoch = 0; epoch < _config.Epochs; epoch++)
        {
            float loss = 0;
            var totalExamples = 0;
            ProcessOrderedBatchesView(orderedBatchesView.Take(1024), ref batchProcessCount, ref loss, ref totalExamples);

            loss /= totalExamples;

            if (epoch % 100 is 0)
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

            DisplayEpochResult(stopWatch.Elapsed, batchProcessCount, loss, epoch);

            yield return Forward();
        }
    }

    private BaseBatchView GetOrderedBatchView(NeuralMatrix trainingInput, NeuralMatrix trainingOutput)
    {
        _indices = [.. Enumerable.Range(0, trainingInput.Rows)];

        var orderedBatchesView = new FiniteBatchesView(_indices, trainingInput, trainingOutput, _config.BatchSize);

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

        var result = $"Epoch ({epoch, 6}/{_config.Epochs, -6}) Accuracy: {lossToPercent:F5}% Loss:{loss, 15:G13} BPS:{batchesPerSecond:F4}/s TP:{elapsed}";
        result = result.WithColor(System.Drawing.Color.FromArgb(255, rReadyLoss, gReadyLoss, 0));

        Console.WriteLine(result);
    }


    private void ProcessOrderedBatchesView(
        IEnumerable<OrderedBatchView> orderedBatchesView,
        ref int batchProcessCount,
        ref float loss,
        ref int totalExamples)
    {
        if (_config.WithShuffle && _config.Model is not null)
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
        _optimizer.Learn();

        return Loss(batch);
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

        if (Avx512F.IsSupported)
        {
            var factorVec = Vector512.Create(factor);
            var rateVec = Vector512.Create(-_config.LearningRate);

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
                aPtr[i] = aPtr[i] * factor - _config.LearningRate * bPtr[i];
            }
        }
    }

    private void BackPropagate(
        OrderedBatchView batch)
    {
        _gradientArchitecture.ZeroOut();

        int rowCount = 0;
        foreach (var trainingPair in batch)
        {
            NativeMemory.Copy(trainingPair.Input, Architecture.MatrixNeurons[0].Pointer, sizeof(float) * (nuint)batch.BatchesView.InputStride);

            Forward();

            for (var j = 0; j < Architecture.Count; j++)
            {
                _gradientArchitecture.MatrixNeurons[j].Clear();
            }

            ComputeOutputLayer(trainingPair.Output);
            PropagateToPreviousLayer();

            ++rowCount;
        }

        NormalizeGradientsVectorized(rowCount);
        ClipGradients();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ComputeOutputLayer(float* trainingOutputPointer)
    {
        var archOutputPtr = Architecture.MatrixNeurons[^1].Pointer;
        var gradOutputErrorPtr = _gradientArchitecture.MatrixNeurons[^1].Pointer;
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

    private void PropagateToPreviousLayer()
    {
        for (int layerIdx = Architecture.Count; layerIdx > 0; layerIdx--)
        {
            var currentArchNeurons = Architecture.MatrixNeurons[layerIdx];
            var currentGradErrors = _gradientArchitecture.MatrixNeurons[layerIdx];

            BackPropagateLayerVectorized(layerIdx, currentArchNeurons, currentGradErrors);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void BackPropagateLayerVectorized(
    int layerIndex,
    NeuralMatrix currentActivations,
    NeuralMatrix currentErrors)
    {
        var previousLayerIndex = layerIndex - 1; 
        var isOutputLayer = layerIndex == (Architecture.Count - 1);

        var prevArchNeurons = Architecture.MatrixNeurons[previousLayerIndex];
        var prevArchWeightsPtr = Architecture.MatrixWeights[previousLayerIndex].Pointer;

        var prevGradNeurons = _gradientArchitecture.MatrixNeurons[previousLayerIndex].Pointer;
        var prevGradWeights = _gradientArchitecture.MatrixWeights[previousLayerIndex].Pointer;
        var prevGradBiases = _gradientArchitecture.MatrixBiases[previousLayerIndex].Pointer;

        var prevArchNeuronsPtr = prevArchNeurons.Pointer;
        var prevArchNeuronsPtrEnd = prevArchNeuronsPtr + prevArchNeurons.ColumnsStride;

        for (var i = 0; i < currentActivations.UsedColumns; i++)
        {
            var activation = currentActivations.Pointer[i];
            var error = currentErrors.Pointer[i];

            var neuronGradient = CalculateNeuronGradient(activation, error, isOutputLayer);

            prevGradBiases[i] += neuronGradient;

            AccumulateVectorizedGradients(prevArchNeuronsPtr, prevArchNeuronsPtrEnd, ref prevArchWeightsPtr, ref prevGradWeights, prevGradNeurons, neuronGradient);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateVectorizedGradients(
        float* prevArchNeuronsPtr,
        float* prevArchNeuronsPtrEnd,
        ref float* prevArchWeightsPtr,
        ref float* prevGradWeights,
        float* prevGradNeurons,
        float neuronGradient)
    {       
        if (Avx512F.IsSupported)
        {
            var neuronGradientsVec = Vector512.Create(neuronGradient);

            for (; prevArchNeuronsPtr != prevArchNeuronsPtrEnd;
                prevArchNeuronsPtr += NeuralMatrix.Alignment,
                prevArchWeightsPtr += NeuralMatrix.Alignment,
                prevGradWeights += NeuralMatrix.Alignment,
                prevGradNeurons += NeuralMatrix.Alignment)
            {
                var prevArchNeuronsVector = Vector512.LoadAligned(prevArchNeuronsPtr);
                var prevArchWeightsVector = Vector512.LoadAligned(prevArchWeightsPtr);

                var wGrad = Avx512F.Multiply(neuronGradientsVec, prevArchNeuronsVector);
                var pGrad = Avx512F.Multiply(neuronGradientsVec, prevArchWeightsVector);

                var existingWGrad = Vector512.LoadAligned(prevGradWeights);
                var existingPGrad = Vector512.LoadAligned(prevGradNeurons);

                var grad = Avx512F.Add(existingWGrad, wGrad);
                grad.StoreAligned(prevGradWeights);

                grad = Avx512F.Add(existingPGrad, pGrad);
                grad.StoreAligned(prevGradNeurons);
            }
        }
        else
        {
            for (; prevArchNeuronsPtr != prevArchNeuronsPtrEnd; ++prevArchNeuronsPtr, ++prevArchWeightsPtr, ++prevGradWeights, ++prevGradNeurons)
            {
                *prevGradWeights += *prevArchNeuronsPtr * neuronGradient;
                *prevGradNeurons += *prevArchWeightsPtr * neuronGradient;
            }
        }
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void NormalizeGradientsVectorized(int rowNumber)
    {
        var divisorVec = Vector512.Create((float)rowNumber);
        var divisorScalar = (float)rowNumber;

        for (var i = 0; i < _gradientArchitecture.Count; i++)
        {
            NormalizeArray(_gradientArchitecture.MatrixWeights[i], divisorVec, divisorScalar);
            NormalizeArray(_gradientArchitecture.MatrixBiases[i], divisorVec, divisorScalar);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void NormalizeArray(NeuralMatrix matrix, Vector512<float> divisorVec, float divisorScalar)
    {
        var ptr = matrix.Pointer;
        float* end = ptr + matrix.AllocatedLength;

        if (Avx512F.IsSupported)
        {
            for (; ptr != end; ptr += NeuralMatrix.Alignment)
            {
                var vec = Vector512.LoadAligned(ptr);
                vec = Avx512F.Divide(vec, divisorVec);
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float CalculateNeuronGradient(float activation, float error, bool isOutput)
    {
        var derivativeFn = isOutput ? _outputDerivative : _hiddenDerivative;
        var gradient = derivativeFn(activation);

        return 2 * Math.Clamp(error, -100f, 100f) * gradient;
    }

    private float Loss(OrderedBatchView batch)
    {
        var loss = 0f;

        var realFirstNeuronMatrix = Architecture.MatrixNeurons[0];
        var realLastNeuronMatrix = Architecture.MatrixNeurons[^1];

        var aPtr = realFirstNeuronMatrix.Pointer;
        var realLastNeuronPtr = realLastNeuronMatrix.Pointer;

        var sumMask = Vector512.Create(_trainingOutputStrideMask).AsSingle();

        foreach (var pair in batch)
        {
            var bPtr = realLastNeuronPtr;

            NativeMemory.Copy(pair.Input, aPtr, sizeof(float) * (nuint)batch.BatchesView.InputStride);
            Forward();

            var cPtr = pair.Output;
            var cEnd = cPtr + batch.BatchesView.OutputStride;

            var predicted = realLastNeuronMatrix;
            var batchLoss = 0f;

            var lossVec = Vector512<float>.Zero;

            for (; cPtr != cEnd; bPtr += NeuralMatrix.Alignment, cPtr += NeuralMatrix.Alignment)
            {
                var predVec = Vector512.LoadAligned(bPtr);
                var targetVec = Vector512.LoadAligned(cPtr);
                var diff = Avx512F.Subtract(predVec, targetVec);
                lossVec = Avx512F.Add(lossVec, Avx512F.Multiply(diff, diff));
            }

            lossVec = Avx512F.And(lossVec.AsInt32(), sumMask.AsInt32()).AsSingle();
            batchLoss += Vector512.Sum(lossVec);

            loss += batchLoss;
        }

        return loss;
    }

    public void ClipGradients(float maxNorm = 1.0f)
    {
        for (int i = 0; i < Architecture.Count; i++)
        {
            _gradientArchitecture.MatrixWeights[i].Clip(-maxNorm, maxNorm);
            _gradientArchitecture.MatrixBiases[i].Clip(-maxNorm, maxNorm);
        }
    }

    private void RandomizeWeightsBiases()
    {
        for (int i = 0; i < Architecture.Count; i++)
        {
            int fan_in = Architecture.MatrixWeights[i].Rows;
            float stddev = MathF.Sqrt(2.0f / fan_in);
            Architecture.MatrixWeights[i].RandomizeGaussian(0, stddev);
            Architecture.MatrixBiases[i].Clear();
        }
    }
}
