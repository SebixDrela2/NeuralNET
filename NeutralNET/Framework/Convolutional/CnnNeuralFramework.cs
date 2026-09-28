using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text.RegularExpressions;
using NeutralNET.Activation;
using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Framework.Connected.Optimizers;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Framework.Convolutional.Native;
using NeutralNET.GPU;
using NeutralNET.Matrices;
using Tensorflow.Keras.Layers;
using static NeutralNET.Activation.ActivationSelector;

namespace NeutralNET.Framework.Neural.CNN;

using static ConvRenter;
using static NeuralRenter;

/// <summary>
/// Zero-GC CNN framework with full object and buffer pooling, pluggable optimizers,
/// and low-latency P/Invoke CUDA/cuBLAS GPU matrix acceleration.
/// </summary>
public sealed unsafe class CnnNeuralFramework
{
    public const bool EnableGpu = true;
    private const int Avx256Size = 8;
    private const int Avx512Size = 16;

    private static readonly bool IsAvx512Supported = Avx512F.IsSupported;
    private static readonly bool IsAvx2Supported = Avx2.IsSupported;

    private readonly CnnArchitectureConfig _cnnConfig;
    private readonly ActivationSelector _activationSelector = new();
    private CnnSize _input;

    private readonly List<ActivationType> _convActivationTypes;
    private readonly List<ActivationFunction> _denseActivations;
    private readonly List<DerivativeFunction> _denseDerivatives;
    private readonly List<ICnnOptimizer> _convOptimizers;
    private readonly List<ICnnOptimizer> _denseOptimizers;

    private readonly List<NeuralMatrix> _denseLayerMatrixes = [];
    private readonly List<DenseHyperParameters> _denseHyperParameters = [];
    private readonly List<ReverseDenseHyperParamers> _reverseDenseHyperParameters = [];
    private readonly List<ConvHyperParameters> _convHyperParameters = [];
    private readonly List<CublasConvAllocations> _cublasConvAllocations = [];
    private readonly List<CublasDenseAllocations> _cublasDenseAllocations = [];
    private readonly List<CublasReverseDenseAllocations> _cublasReverseDenseAllocations = [];

    private CnnMatrix _pooledOutputGrad;
    private NeuralMatrix _outputGrad;

    private NeuralMatrix? _flattenedInput;

    private readonly Random _rng;
    private readonly int _maxBatch;

    public PerfCounter? Perf { get; set; }

    public CnnNeuralFramework(NeuralNetworkConfig baseConfig, CnnArchitectureConfig cnnConfig,
        int batchSize, int inputChannels, int inputHeight, int inputWidth)
    {
        _cnnConfig = cnnConfig;
        _rng = new Random();
        _input = new CnnSize(batchSize, inputChannels, inputHeight, inputWidth);
        _maxBatch = batchSize;

        var convCount = cnnConfig.ConvLayers.Count;

        _convHyperParameters = [with(convCount)];
        _convActivationTypes = [with(convCount)];
        _convOptimizers = [with(convCount)];

        SetupCnnConvParameters(cnnConfig);

        if (EnableGpu)
        {
            SetupCublasForCnn(cnnConfig);
        }

        int flattenedSize = ComputeFlattenedSize(cnnConfig);
        int[] denseArch = [flattenedSize, .. cnnConfig.DenseArchitecture];
        int denseCount = denseArch.Length - 1;

        _denseHyperParameters = [with(denseCount)];
        _denseActivations = [with(denseCount)];
        _denseDerivatives = [with(denseCount)];
        _denseOptimizers = [with(denseCount)];

        SetupDenseArchitecture(denseArch, cnnConfig);
        SetupReverseDenseHyperParameters();

        if (EnableGpu)
        {
            SetUpCublasForDense();
            SetupCublasDenseReversed();
        }
    }

    private void SetupCublasForCnn(CnnArchitectureConfig cnnConfig)
    {
        for (int i = 0; i < cnnConfig.ConvLayers.Count; i++)
        {
            var gradPatchMat = GetGradPatchMat(i);
            var dW = GetDw(i);
            var convolution = GetConvolution(i);

            _cublasConvAllocations.Add(new CublasConvAllocations(
                gradPatchMat, dW, convolution));
        }

        CublasContext GetGradPatchMat(int i)
        {
            var preGradMatrix = _convHyperParameters[i].PreGradMatrix;
            var flattenedWeights = _convHyperParameters[i].FlattenedWeights;
            var gradPatchMath = _convHyperParameters[i].GradPatchMat;

            var patches = preGradMatrix.Rows;
            var inDim = _convHyperParameters[i].ColInput.UsedColumns;
            var filters = _convHyperParameters[i].PreGrad.Channels;

            var context = new CublasContext(
                new CublasTransitions(CublasOperation.NonTranspose, CublasOperation.NonTranspose),
                new CublasItem<int>(patches, inDim, filters),
                new CublasItem<int>(preGradMatrix.ColumnsStride, flattenedWeights.ColumnsStride, gradPatchMath.ColumnsStride));

            return context;
        }

        CublasContext GetDw(int i)
        {
            var preGradMatrix = _convHyperParameters[i].PreGradMatrix;
            var colInput = _convHyperParameters[i].ColInput;
            var dW = _convHyperParameters[i].DWeights;

            var filters = _convHyperParameters[i].PreGrad.Channels;
            var inDim = _convHyperParameters[i].ColInput.UsedColumns;
            var patches = preGradMatrix.Rows;

            var context = new CublasContext(
                new CublasTransitions(CublasOperation.Transpose, CublasOperation.NonTranspose),
                new CublasItem<int>(filters, inDim, patches),
                new CublasItem<int>(preGradMatrix.ColumnsStride, colInput.ColumnsStride, dW.ColumnsStride));

            return context;
        }

        CublasContext GetConvolution(int i)
        {
            var flattenedWeights = _convHyperParameters[i].FlattenedWeights;
            var colInput = _convHyperParameters[i].ColInput;

            var patches = colInput.Rows;
            var filters = flattenedWeights.Rows;
            var innerDim = colInput.UsedColumns;

            var convolution = _convHyperParameters[i].Convolution;

            var context = new CublasContext(
                new CublasTransitions(CublasOperation.NonTranspose, CublasOperation.Transpose),
                new CublasItem<int>(patches, filters, innerDim),
                new CublasItem<int>(colInput.ColumnsStride, flattenedWeights.ColumnsStride, convolution.ColumnsStride));

            return context;
        }
    }

    private void SetupCnnConvParameters(CnnArchitectureConfig cnnConfig)
    {
        var cnnSize = _input;

        for (int i = 0; i < cnnConfig.ConvLayers.Count; i++)
        {
            cnnSize = SetupCnnConvParametersForLayer(cnnConfig, cnnSize, i);
        }

        var lastPooled = _convHyperParameters[^1].Input!;
        int featureDim = lastPooled.Channels * lastPooled.Height * lastPooled.Width;

        _flattenedInput = RentNeural(lastPooled.Batch, featureDim);
        _pooledOutputGrad = RentCnn(lastPooled.Batch, lastPooled.Channels, lastPooled.Height, lastPooled.Width);
    }

    private CnnSize SetupCnnConvParametersForLayer(CnnArchitectureConfig cnnConfig, CnnSize prevInput, int i)
    {
        CnnLayerConfig? layer = cnnConfig.ConvLayers[i];
        var fanIn = prevInput.Channels * layer.KernelHeight * layer.KernelWidth;
        var stddev = MathF.Sqrt(2.0f / fanIn);
        var weights = RentCnn(layer.Filters, prevInput.Channels, layer.KernelHeight, layer.KernelWidth);

        for (int f = 0; f < layer.Filters; f++)
        {
            for (int c = 0; c < prevInput.Channels; c++)
            {
                for (int y = 0; y < layer.KernelHeight; y++)
                {
                    for (int x = 0; x < layer.KernelWidth; x++)
                    {
                        weights[f, c, y, x] = NextGaussianFloat(0, stddev);
                    }
                }
            }
        }

        var biases = RentCnn(1, layer.Filters, 1, 1);

        for (int f = 0; f < layer.Filters; f++)
        {
            biases[0, f, 0, 0] = NextGaussianFloat(0, 0.1f);
        }

        var flattenedWeights = FlattenConvWeights(weights);
        var convOutSz = GetCnnSize(prevInput.BatchSize, weights.Batch, prevInput.Height, prevInput.Width, layer);
        var preAct = RentCnn(convOutSz);
        var postAct = RentCnn(convOutSz);
        var gradInput = RentCnn(convOutSz);
        var preGrad = RentCnn(convOutSz);
        var preGradMatrix = GetPreGradMatrix(preGrad);
        var input = GetInput(postAct, layer.PoolSize);
        var inputGrad = RentCnn(input.Batch, input.Channels, input.Height, input.Width);

        int nextH = layer.UseMaxPool ? convOutSz.Height / layer.PoolSize : convOutSz.Height;
        int nextW = layer.UseMaxPool ? convOutSz.Width / layer.PoolSize : convOutSz.Width;
        var nextLayer = new CnnSize(prevInput.BatchSize, weights.Batch, nextH, nextW);

        var colInput = GetColInput(prevInput, layer.KernelHeight, layer.KernelWidth, layer.Stride, layer.Padding);
        var gradPatchMat = RentNeural(preGradMatrix.Rows, colInput.UsedColumns);
        var poolIndices = GetPoolIndices(postAct, layer.PoolSize);

        var dW = GetDWeights(colInput, preGrad);
        var dB = RentNeural(preGrad.Channels, 1);
        var convolution = RentNeural(colInput.Rows, flattenedWeights.Rows);
        var pooled = GetPooled(layer, preAct);

        ConvHyperParameters convHyperParam = new(
            input, colInput, weights, flattenedWeights, biases,
            preAct, postAct, poolIndices, gradInput, preGrad,
            preGradMatrix, dW, dB, convolution, inputGrad,
            gradPatchMat, pooled);

        convHyperParam.Init(i);

        _convHyperParameters.Add(convHyperParam);
        _convActivationTypes.Add(layer.Activation);

        SetUpOptimizerPerCnnLayer(dW);

        prevInput = nextLayer;

        return prevInput;
    }

    private void SetUpOptimizerPerCnnLayer(NeuralMatrix dW)
    {
        int innerDim = dW.Rows;
        int filters = dW.UsedColumns;

        var mWeights = RentNeural(innerDim, filters);
        var vWeights = RentNeural(innerDim, filters);
        var mBiases = RentNeural(1, filters);
        var vBiases = RentNeural(1, filters);

        var adamParameters = new AdamHyperLayerParameters(mWeights, vWeights, mBiases, vBiases);
        var opt = CnnOptimizerFactory.Create(_cnnConfig.OptimizerConfig, adamParameters, null!);

        _convOptimizers.Add(opt);
    }

    private void SetUpCublasForDense()
    {
        for (var i = 0; i < _denseHyperParameters.Count; i++)
        {
            var current = i == 0 ? _flattenedInput! : _denseLayerMatrixes[i - 1];
            var result = _denseLayerMatrixes[i];

            var layerResult = GetLayer(current, result, i);

            _cublasDenseAllocations.Add(new CublasDenseAllocations(layerResult));
        }

        CublasContext GetLayer(NeuralMatrix current, NeuralMatrix result, int i)
        {
            var weights = _denseHyperParameters[i].Weights;

            int batchSize = current.Rows;
            int inFeatures = current.UsedColumns;
            int outFeatures = weights.Rows;

            var context = new CublasContext(
                new CublasTransitions(CublasOperation.NonTranspose, CublasOperation.Transpose),
                new CublasItem<int>(batchSize, outFeatures, inFeatures),
                new CublasItem<int>(current.ColumnsStride, weights.ColumnsStride, result.ColumnsStride));

            return context;
        }
    }

    private void SetupCublasDenseReversed()
    {
        for (int i = _denseHyperParameters.Count - 1; i >= 0; i--)
        {
            var gradOutput = i == _denseHyperParameters.Count - 1
                ? _outputGrad
                : _reverseDenseHyperParameters[i + 1].GradInput;
            var inputToLayer = (i == 0) ? _flattenedInput! : _denseHyperParameters[i - 1].PostAct;

            var dW = GetDW(gradOutput, inputToLayer, i);
            var gradInput = GetGradInput(gradOutput, i);

            _cublasReverseDenseAllocations.Add(new CublasReverseDenseAllocations(dW, gradInput));
        }

        _cublasReverseDenseAllocations.Reverse();

        CublasContext GetDW(NeuralMatrix gradOutput, NeuralMatrix inputToLayer, int i)
        {
            int batch = gradOutput.Rows;
            int outDim = gradOutput.UsedColumns;
            int inDim = inputToLayer.UsedColumns;

            var gradPre = _reverseDenseHyperParameters[i].GradPre;
            var dW = _reverseDenseHyperParameters[i].DWeights;

            var context = new CublasContext(
                new CublasTransitions(CublasOperation.Transpose, CublasOperation.NonTranspose),
                new CublasItem<int>(inDim, outDim, batch),
                new CublasItem<int>(inputToLayer.ColumnsStride, gradPre.ColumnsStride, dW.ColumnsStride));

            return context;
        }

        CublasContext GetGradInput(NeuralMatrix gradOutput, int i)
        {
            int batch = gradOutput.Rows;
            var weights = _denseHyperParameters[i].Weights;
            var weightOutDim = weights.Rows;
            var weightInDim = weights.UsedColumns;
            var gradInput = _reverseDenseHyperParameters[i].GradInput;
            var gradPre = _reverseDenseHyperParameters[i].GradPre;

            var context = new CublasContext(
                new CublasTransitions(CublasOperation.NonTranspose, CublasOperation.NonTranspose),
                new CublasItem<int>(batch, weightInDim, weightOutDim),
                new CublasItem<int>(gradPre.ColumnsStride, weights.ColumnsStride, gradInput.ColumnsStride));

            return context;
        }
    }

    private void SetupDenseArchitecture(int[] denseArch, CnnArchitectureConfig cnnConfig)
    {
        var actSize = _input.BatchSize;

        for (int i = 0; i < denseArch.Length - 1; i++)
        {
            var inputSize = denseArch[i];
            var outputSize = denseArch[i + 1];
            float stddev = MathF.Sqrt(2.0f / inputSize);

            var weights = RentNeural(outputSize, inputSize);
            for (int outIdx = 0; outIdx < outputSize; outIdx++)
            {
                for (int inIdx = 0; inIdx < inputSize; inIdx++)
                {
                    weights.At(outIdx, inIdx) = NextGaussianFloat(0, stddev);
                }
            }

            var biases = RentNeural(1, outputSize);
            for (int j = 0; j < outputSize; j++)
            {
                biases.At(0, j) = NextGaussianFloat(0, 0.1f);
            }

            var preAct = RentNeural(actSize, weights.Rows);
            var postAct = RentNeural(actSize, weights.Rows);

            _denseHyperParameters.Add(new(weights, biases, preAct, postAct));

            weights.DisplayName = $"Dense_Weights[{i}]";
            biases.DisplayName = $"Dense_Biases[{i}]";
            preAct.DisplayName = $"Dense_PreAct[{i}]";
            postAct.DisplayName = $"Dense_PostAct[{i}]";

            ActivationType actType = (i == denseArch.Length - 2)
                ? cnnConfig.OutputActivation
                : cnnConfig.DenseHiddenActivation;

            var act = _activationSelector.GetActivation(actType);
            var der = _activationSelector.GetDerivative(actType);

            _denseActivations.Add(act);
            _denseDerivatives.Add(der);
        }

        var probabilities = _denseHyperParameters[^1].PostAct;
        int rows = probabilities.Rows;
        int cols = probabilities.UsedColumns;

        _outputGrad = RentNeural(rows, cols);
    }

    private void SetupReverseDenseHyperParameters()
    {
        var gradOutput = _outputGrad;

        for (int i = _denseHyperParameters.Count - 1; i >= 0; i--)
        {
            var inputToLayer = (i == 0) ? _flattenedInput! : _denseHyperParameters[i - 1].PostAct;

            int batch = gradOutput.Rows;
            int outDim = gradOutput.UsedColumns;
            int inDim = inputToLayer.UsedColumns;

            var weights = _denseHyperParameters[i].Weights;
            int weightInDim = weights.UsedColumns;

            var gradPre = RentNeural(batch, outDim);
            var dW = RentNeural(inDim, outDim);
            var dB = RentNeural(1, outDim);
            var gradInput = RentNeural(batch, weightInDim);

            _reverseDenseHyperParameters.Add(new(gradPre, dW, dB, gradInput));

            gradOutput = gradInput;
        }

        _reverseDenseHyperParameters.Reverse();

        var current = _flattenedInput!;

        for (int i = 0; i < _denseHyperParameters.Count; i++)
        {
            int batchSize = current.Rows;
            int inFeatures = current.UsedColumns;
            int outFeatures = _denseHyperParameters[i].Weights.Rows;

            var layerMatrix = RentNeural(batchSize, outFeatures);

            _denseLayerMatrixes.Add(layerMatrix);

            var dW = _reverseDenseHyperParameters[i].DWeights;

            int iSize = dW.Rows;
            int oSize = dW.UsedColumns;

            var mWeights = RentNeural(iSize, oSize);
            var vWeights = RentNeural(iSize, oSize);
            var mBiases = RentNeural(1, oSize);
            var vBiases = RentNeural(1, oSize);

            var adamParameters = new AdamHyperLayerParameters(mWeights, vWeights, mBiases, vBiases);
            var opt = CnnOptimizerFactory.Create(_cnnConfig.OptimizerConfig, null!, adamParameters);

            _denseOptimizers.Add(opt);

            current = layerMatrix;
        }
    }

    public CnnMatrix[] GetConvLayerOutput(CnnMatrix input)
    {
        CnnMatrix[] output = new CnnMatrix[_cnnConfig.ConvLayers.Count];
        CnnMatrix current = input;

        for (int i = 0; i < _cnnConfig.ConvLayers.Count; i++)
        {
            ConvForward(current, i);

            var layer = _cnnConfig.ConvLayers[i];
            var convPreAct = _convHyperParameters[i].PreAct;
            var pooled = _convHyperParameters[i].Pooled;

            var pAct = convPreAct.Pointer;
            var totalElements = convPreAct.Batch * convPreAct.Channels * convPreAct.Height * convPreAct.Width;

            ApplyActivationVectorized(pAct, totalElements, layer.Activation);
            MaxPoolForwardInPlace(convPreAct, pooled, layer.PoolSize);

            output[i] = pooled;
        }

        return output;
    }

    public void SaveData<TEnum>(TEnum key, Stream stream) where TEnum : struct, Enum
    {
        using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);

        writer.Write(Convert.ToInt32(key));
        writer.Write(_convHyperParameters.Count);
        writer.Write(_denseHyperParameters.Count);

        for (int i = 0; i < _convHyperParameters.Count; i++)
        {
            SaveCnnMatrix(writer, _convHyperParameters[i].Weights);
            SaveCnnMatrix(writer, _convHyperParameters[i].Biases);
        }

        for (int i = 0; i < _denseHyperParameters.Count; i++)
        {
            SaveNeuralMatrix(writer, _denseHyperParameters[i].Weights);
            SaveNeuralMatrix(writer, _denseHyperParameters[i].Biases);
        }

        writer.Flush();
    }

    public void SaveData<TEnum>(TEnum key, string directoryPath) where TEnum : struct, Enum
    {
        Directory.CreateDirectory(directoryPath);
        var filePath = Path.Combine(directoryPath, $"{typeof(TEnum).Name}_{key}.bin");
        using var stream = File.Create(filePath);
        SaveData(key, stream);
    }

    public bool LoadData<TEnum>(TEnum key, Stream stream)
        where TEnum : struct, Enum
    {
        using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);

        var savedEnumKey = reader.ReadInt32();
        if (savedEnumKey != Convert.ToInt32(key))
        {
            Console.WriteLine($"Mismatch enum key in model file. Expected: '{key}', Found: '{savedEnumKey}'.");
            return false;
        }

        int convCount = reader.ReadInt32();
        int denseCount = reader.ReadInt32();

        if (convCount != _convHyperParameters.Count || denseCount != _denseHyperParameters.Count)
        {
            Console.WriteLine($"Layer count mismatch between framework and checkpoint file.");
            return false;
        }

        for (int i = 0; i < _convHyperParameters.Count; i++)
        {
            LoadCnnMatrix(reader, _convHyperParameters[i].Weights);
            LoadCnnMatrix(reader, _convHyperParameters[i].Biases);
        }

        for (int i = 0; i < _denseHyperParameters.Count; i++)
        {
            LoadNeuralMatrix(reader, _denseHyperParameters[i].Weights);
            LoadNeuralMatrix(reader, _denseHyperParameters[i].Biases);
        }

        return true;
    }

    public bool LoadData<TEnum>(TEnum key, string directoryPath) where TEnum : struct, Enum
    {
        var filePath = Path.Combine(directoryPath, $"{typeof(TEnum).Name}_{key}.bin");

        if (!File.Exists(filePath)) return false;

        using (var stream = File.OpenRead(filePath))
        {
            if (LoadData(key, stream)) return true;
        }

        Console.Write($"Override exsisting file? \e[s");
        while (true)
        {
            Console.Write("\e[u (y/n)\e[K\e[u");
            switch (Console.ReadLine()?.Trim().ToLower())
            {
                case "y":
                    File.Delete(filePath);
                    Console.WriteLine("\e[u\r\e[K");
                    return false;
                case "n":
                    throw new InvalidOperationException();
            }
        }
    }

    private static void SaveCnnMatrix(BinaryWriter writer, CnnMatrix matrix)
    {
        writer.Write(matrix.Batch);
        writer.Write(matrix.Channels);
        writer.Write(matrix.Height);
        writer.Write(matrix.Width);

        int totalElements = matrix.UnsafeSize;
        float* ptr = matrix.Pointer;
        for (int i = 0; i < totalElements; i++)
        {
            writer.Write(ptr[i]);
        }
    }

    private static void LoadCnnMatrix(BinaryReader reader, CnnMatrix matrix)
    {
        int b = reader.ReadInt32();
        int c = reader.ReadInt32();
        int h = reader.ReadInt32();
        int w = reader.ReadInt32();

        if (b != matrix.Batch || c != matrix.Channels || h != matrix.Height || w != matrix.Width)
        {
            throw new InvalidOperationException("CnnMatrix dimensions mismatch when loading standard layer.");
        }

        int totalElements = matrix.UnsafeSize;
        float* ptr = matrix.Pointer;
        for (int i = 0; i < totalElements; i++)
        {
            ptr[i] = reader.ReadSingle();
        }
    }

    private static void SaveNeuralMatrix(BinaryWriter writer, NeuralMatrix matrix)
    {
        writer.Write(matrix.Rows);
        writer.Write(matrix.UsedColumns);

        float* ptr = matrix.Pointer;
        int stride = matrix.ColumnsStride;

        for (int r = 0; r < matrix.Rows; r++)
        {
            float* rowPtr = ptr + r * stride;
            for (int c = 0; c < matrix.UsedColumns; c++)
            {
                writer.Write(rowPtr[c]);
            }
        }
    }

    private static void LoadNeuralMatrix(BinaryReader reader, NeuralMatrix matrix)
    {
        int rows = reader.ReadInt32();
        int cols = reader.ReadInt32();

        if (rows != matrix.Rows || cols != matrix.UsedColumns)
        {
            throw new InvalidOperationException("NeuralMatrix shape mismatch during loading.");
        }

        float* ptr = matrix.Pointer;
        int stride = matrix.ColumnsStride;

        for (int r = 0; r < matrix.Rows; r++)
        {
            float* rowPtr = ptr + r * stride;
            for (int c = 0; c < matrix.UsedColumns; c++)
            {
                rowPtr[c] = reader.ReadSingle();
            }
        }
    }

    public void Dispose()
    {
        foreach (var b in _convHyperParameters) b.Dispose();
        foreach (var w in _denseHyperParameters) w.Dispose();
        foreach (var l in _denseLayerMatrixes) l.Dispose();
        foreach (var r in _reverseDenseHyperParameters) r.Dispose();
        foreach (var opt in _convOptimizers) opt.Dispose();
        foreach (var opt in _denseOptimizers) opt.Dispose();
        foreach (var alloc in _cublasConvAllocations) alloc.Dispose();
        foreach (var alloc in _cublasDenseAllocations) alloc.Dispose();
        foreach (var alloc in _cublasReverseDenseAllocations) alloc.Dispose();

        _pooledOutputGrad.Dispose();
        _outputGrad.Dispose();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float NextGaussianFloat(float mean, float stddev)
    {
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = 1.0 - _rng.NextDouble();

        return (float)(mean + stddev * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
    }

    private int ComputeFlattenedSize(CnnArchitectureConfig config)
    {
        var w = _input.Width;
        var h = _input.Height;
        var channels = _input.Channels;

        foreach (var layer in config.ConvLayers)
        {
            int paddedW = w + (2 * layer.Padding);
            int paddedH = h + (2 * layer.Padding);

            w = (paddedW - layer.KernelWidth) / layer.Stride + 1;
            h = (paddedH - layer.KernelHeight) / layer.Stride + 1;

            channels = layer.Filters;

            if (layer.UseMaxPool)
            {
                h /= layer.PoolSize;
                w /= layer.PoolSize;
            }
        }

        return channels * h * w;
    }

    public NeuralMatrix Forward(CnnMatrix input)
    {
        CnnMatrix? current = input;

        using (Perf?.Measure("Forward.SetBatchLimitAll"))
            SetBatchLimitAll(input.Batch);

        for (int layerIdx = 0; layerIdx < _cnnConfig.ConvLayers.Count; layerIdx++)
        {
            var layer = _cnnConfig.ConvLayers[layerIdx];
            var prefix = $"Forward.Conv[{layerIdx}]";

            using (Perf?.Measure($"{prefix}.ConvForward"))
                ConvForward(current, layerIdx);

            var convPreAct = _convHyperParameters[layerIdx].PreAct;
            var pooled = _convHyperParameters[layerIdx].Pooled;

            var pAct = convPreAct.Pointer;
            var totalElements = convPreAct.Batch * convPreAct.Channels * convPreAct.Height * convPreAct.Width;

            using (Perf?.Measure($"{prefix}.Activation"))
                ApplyActivationVectorized(pAct, totalElements, layer.Activation);

            using (Perf?.Measure($"{prefix}.MaxPool"))
                MaxPoolForwardInPlace(convPreAct, pooled, layer.PoolSize);

            current = pooled;
        }

        var lastPooled = _convHyperParameters[^1].Pooled;

        using (Perf?.Measure("Forward.Flatten"))
            Flatten(lastPooled);

        using (Perf?.Measure("Forward.DenseForward"))
            DenseForward(storeIntermediates: false);

        return _denseLayerMatrixes[^1];
    }

    private static void ApplyActivationVectorized(float* ptr, int count, ActivationType activation)
    {
        int i = 0;
        switch (activation)
        {
            case ActivationType.ReLU:
                if (IsAvx512Supported)
                {
                    var vZero = Vector512<float>.Zero;
                    int vecLimit = count - (count % Avx512Size);

                    for (; i < vecLimit; i += Avx512Size)
                    {
                        var vSrc = Vector512.Load(ptr + i);
                        Vector512.Max(vSrc, vZero).Store(ptr + i);
                    }
                }
                else if (IsAvx2Supported)
                {
                    var vZero = Vector256<float>.Zero;
                    int vecLimit = count - (count % Avx256Size);

                    for (; i < vecLimit; i += Avx256Size)
                    {
                        var vSrc = Avx.LoadVector256(ptr + i);
                        Avx.Max(vSrc, vZero).Store(ptr + i);
                    }
                }

                for (; i < count; i++)
                {
                    if (ptr[i] < 0f) ptr[i] = 0f;
                }
                break;

            case ActivationType.LeakyReLU:
                const float alpha = 0.01f;

                if (IsAvx512Supported)
                {
                    var vAlpha = Vector512.Create(alpha);
                    int vecLimit = count - (count % Avx512Size);

                    for (; i < vecLimit; i += Avx512Size)
                    {
                        var vSrc = Vector512.Load(ptr + i);
                        var vScaled = vSrc * vAlpha;
                        Vector512.Max(vSrc, vScaled).Store(ptr + i);
                    }
                }
                else if (IsAvx2Supported)
                {
                    var vAlpha = Vector256.Create(alpha);
                    int vecLimit = count - (count % Avx256Size);

                    for (; i < vecLimit; i += Avx256Size)
                    {
                        var vSrc = Avx.LoadVector256(ptr + i);
                        var vScaled = Avx.Multiply(vSrc, vAlpha);
                        Avx.Max(vSrc, vScaled).Store(ptr + i);
                    }
                }

                for (; i < count; i++)
                {
                    if (ptr[i] < 0f) ptr[i] *= alpha;
                }
                break;

            default:
                break;
        }
    }

    public float TrainBatch(CnnMatrix input, NeuralMatrix target, float learningRate)
    {
        input.DisplayName = "MainInput";
        target.DisplayName = "MainExpected";

        CnnMatrix current = input;

        using (Perf?.Measure("Train.SetBatchLimitAll"))
            SetBatchLimitAll(current.Batch);

        using (Perf?.Measure("Train.ForwardPoolingPass"))
            ForwardPoolingPass(ref current);

        float loss;
        using (Perf?.Measure("Train.ComputeLoss"))
            loss = ComputeCrossEntropyLoss(target);

        using (Perf?.Measure("Train.LossGradient"))
            LossGradientVectorized(target);

        using (Perf?.Measure("Train.DenseBackward"))
            DenseBackWardClipped(learningRate);

        using (Perf?.Measure("Train.BulkMemoryCopy"))
            BulkMemoryCopy();

        using (Perf?.Measure("Train.ConvBackward"))
            PerformConvolutionBackwardPass();

        return float.IsNaN(loss) || float.IsInfinity(loss) || loss > 100f ? 10.0f : loss;
    }

    private void SetBatchLimitAll(int limit)
    {
        if (limit <= 0)
            throw new InvalidOperationException($"SetBatchLimitAll: non-positive batch {limit}.");
        if (limit > _maxBatch)
            throw new InvalidOperationException(
                $"SetBatchLimitAll: requested batch {limit} exceeds the maximum " +
                $"the framework was constructed for ({_maxBatch}).");

        _input.BatchSize = limit;

        foreach (var elem in _convHyperParameters)
        {
            elem.SetBatchLimit(limit);
        }

        foreach (var (_, _, preAct, postAct) in _denseHyperParameters)
        {
            preAct.SetRowSize(limit);
            postAct.SetRowSize(limit);
        }

        foreach (var m in _denseLayerMatrixes)
        {
            m.SetRowSize(limit);
        }

        foreach (var r in _reverseDenseHyperParameters)
        {
            r.GradPre.SetRowSize(limit);
            r.GradInput.SetRowSize(limit);
        }

        _flattenedInput!.SetRowSize(limit);
        _pooledOutputGrad.SetBatch(limit);
        _outputGrad.SetRowSize(limit);
    }

    private void PerformConvolutionBackwardPass()
    {
        var currentGrad = _pooledOutputGrad;

        for (int layerIdx = _cnnConfig.ConvLayers.Count - 1; layerIdx >= 0; layerIdx--)
        {
            var layer = _cnnConfig.ConvLayers[layerIdx];
            var cnvParams = _convHyperParameters[layerIdx];

            currentGrad = ProcessSingleConvLayerBackward(
                currentGrad,
                layerIdx, layer,
                cnvParams
            );
        }
    }

    private CnnMatrix ProcessSingleConvLayerBackward(
        CnnMatrix currentGrad,
        int layerIdx,
        CnnLayerConfig layer,
        ConvHyperParameters cnvParams)
    {
        CnnMatrix preAct = cnvParams.PreAct;
        CnnMatrix postAct = cnvParams.PostAct;
        NeuralMatrix colInput = cnvParams.ColInput;
        CnnMatrix input = cnvParams.Input;
        NeuralMatrix indices = cnvParams.PoolIndices;
        CnnMatrix gradInput = cnvParams.GradInput;
        CnnMatrix preGrad = cnvParams.PreGrad;
        NeuralMatrix preGradMatrix = cnvParams.PreGradMatrix;
        NeuralMatrix dW = cnvParams.DWeights;
        NeuralMatrix dB = cnvParams.DBiases;
        CnnMatrix inputGrad = cnvParams.InputGrad;
        NeuralMatrix gradPatchMat = cnvParams.GradPatchMat;

        var prefix = $"ConvBack[{layerIdx}]";

        using (Perf?.Measure($"{prefix}.Clear"))
        {
            dW.Clear();
            dB.Clear();
            gradInput.Clear();
        }

        using (Perf?.Measure($"{prefix}.MaxPoolBackward"))
            BackPropagateThroughPool(currentGrad, layer, gradInput, indices);

        using (Perf?.Measure($"{prefix}.PreGradient"))
            ComputePreGradient(layer, preGrad, postAct, gradInput);

        using (Perf?.Measure($"{prefix}.ConvertPregrad"))
            ConvertPregradToMatrix(preGrad, preGradMatrix);

        var patches = preGradMatrix.Rows;
        var filters = preGrad.Channels;
        var inDim = colInput.UsedColumns;

        using (Perf?.Measure($"{prefix}.WeightGrad"))
            ComputeWeightGradient(colInput, preGradMatrix, dW, patches, filters, inDim, layerIdx);

        using (Perf?.Measure($"{prefix}.BiasGrad"))
            ComputeBiasGradient(preGradMatrix, dB, patches, filters);

        using (Perf?.Measure($"{prefix}.AdamUpdate"))
        {
            _convOptimizers[layerIdx].Update(
                cnvParams.Weights,
                cnvParams.Biases,
                dW,
                dB);
        }

        using (Perf?.Measure($"{prefix}.InputGrad"))
        {
            var flattenedWeights = cnvParams.FlattenedWeights;
            ComputeGradientWithRespectToInput(flattenedWeights, preGradMatrix, gradPatchMat, patches, filters, inDim, layerIdx);
        }

        using (Perf?.Measure($"{prefix}.Col2Im"))
            inputGrad.Col2Im(gradPatchMat);

        return inputGrad;
    }

    private void ComputeGradientWithRespectToInput(
        NeuralMatrix flattenedWeightMatrix,
        NeuralMatrix preGradMatrix,
        NeuralMatrix gradPatchMat,
        int patches,
        int filters,
        int inDim,
        int layerIdx)
    {
        if (EnableGpu)
        {
            var allocation = _cublasConvAllocations[layerIdx].GradPatchMat;

            GpuMatrixOps.RowMajorSgemmHostStaged(
                allocation,
                patches,
                inDim,
                filters,
                preGradMatrix.Pointer,
                flattenedWeightMatrix.Pointer,
                gradPatchMat.Pointer);
        }
        else
        {
            float* pGradPatch = gradPatchMat.Pointer;
            float* pPreGradMat = preGradMatrix.Pointer;
            float* pWeightMat = flattenedWeightMatrix.Pointer;

            int gradPatchStride = gradPatchMat.ColumnsStride;
            int weightMatStride = flattenedWeightMatrix.ColumnsStride;
            int preGradMatStride = preGradMatrix.ColumnsStride;

            for (int patch = 0; patch < patches; patch++)
            {
                float* rowPreGradMat = pPreGradMat + patch * preGradMatStride;
                float* rowGradPatch = pGradPatch + patch * gradPatchStride;

                Unsafe.InitBlockUnaligned(rowGradPatch, 0, (uint)(inDim * sizeof(float)));

                for (int f = 0; f < filters; f++)
                {
                    float gv = rowPreGradMat[f];
                    if (gv == 0f) continue;

                    float* rowW = pWeightMat + f * weightMatStride;

                    int i = 0;

                    if (Avx512F.IsSupported)
                    {
                        var vg = Vector512.Create(gv);
                        int limit = inDim - (inDim % Avx512Size);
                        for (; i < limit; i += Avx512Size)
                        {
                            var vW = Vector512.LoadUnsafe(ref rowW[i]);
                            var vGP = Vector512.LoadUnsafe(ref rowGradPatch[i]);
                            var vRes = Avx512F.FusedMultiplyAdd(vg, vW, vGP);
                            Vector512.StoreUnsafe(vRes, ref rowGradPatch[i]);
                        }
                    }
                    else if (Avx2.IsSupported)
                    {
                        var vg = Vector256.Create(gv);
                        int limit = inDim - (inDim % Avx256Size);
                        for (; i < limit; i += Avx256Size)
                        {
                            var vW = Avx.LoadVector256(rowW + i);
                            var vGP = Avx.LoadVector256(rowGradPatch + i);
                            var vRes = Fma.MultiplyAdd(vg, vW, vGP);
                            Avx.Store(rowGradPatch + i, vRes);
                        }
                    }

                    for (; i < inDim; i++)
                        rowGradPatch[i] += gv * rowW[i];
                }
            }
        }
    }

    private void ComputeWeightGradient(
        NeuralMatrix colInput,
        NeuralMatrix preGradMatrix,
        NeuralMatrix dW,
        int patches,
        int filters,
        int inDim,
        int layerIdx)
    {
        if (EnableGpu)
        {
            var allocation = _cublasConvAllocations[layerIdx].DWeights;

            GpuMatrixOps.RowMajorSgemmHostStaged(
                allocation,
                filters, inDim, patches,
                preGradMatrix.Pointer,
                colInput.Pointer,
                dW.Pointer);
        }
        else
        {
            float* pdW = dW.Pointer;
            int dWStride = dW.ColumnsStride;

            float* pColIn = colInput.Pointer;
            int colInStride = colInput.ColumnsStride;
            float* pPreGradMat = preGradMatrix.Pointer;
            int preGradMatStride = preGradMatrix.ColumnsStride;

            for (int patch = 0; patch < patches; patch++)
            {
                float* rowColIn = pColIn + patch * colInStride;
                float* rowPreGrad = pPreGradMat + patch * preGradMatStride;

                for (int f = 0; f < filters; f++)
                {
                    float gv = rowPreGrad[f];
                    if (gv == 0f) continue;

                    float* rowDW = pdW + f * dWStride;
                    int i = 0;

                    if (Avx512F.IsSupported)
                    {
                        var vg = Vector512.Create(gv);
                        int vecLimit = inDim - (inDim % Avx512Size);
                        for (; i < vecLimit; i += Avx512Size)
                        {
                            var vIn = Vector512.Load(rowColIn + i);
                            var vDW = Vector512.Load(rowDW + i);
                            vDW = Avx512F.FusedMultiplyAdd(vg, vIn, vDW);
                            vDW.Store(rowDW + i);
                        }
                    }
                    else if (Avx2.IsSupported)
                    {
                        var vg = Vector256.Create(gv);
                        int vecLimit = inDim - (inDim % Avx256Size);
                        for (; i < vecLimit; i += Avx256Size)
                        {
                            var vIn = Avx.LoadVector256(rowColIn + i);
                            var vDW = Avx.LoadVector256(rowDW + i);
                            vDW = Fma.MultiplyAdd(vg, vIn, vDW);
                            Avx.Store(rowDW + i, vDW);
                        }
                    }

                    for (; i < inDim; i++)
                    {
                        rowDW[i] += gv * rowColIn[i];
                    }
                }
            }
        }
    }

    private static void ComputeBiasGradient(NeuralMatrix preGradMatrix, NeuralMatrix dB, int patches, int filters)
    {
        var pdB = dB.Pointer;
        var pPreGradMat = preGradMatrix.Pointer;
        var preGradMatStride = preGradMatrix.ColumnsStride;

        for (int patch = 0; patch < patches; patch++)
        {
            float* rowPreGradMat = pPreGradMat + patch * preGradMatStride;
            int f = 0;

            if (IsAvx512Supported)
            {
                int vecLimit = filters - (filters % Avx512Size);
                for (; f < vecLimit; f += Avx512Size)
                {
                    var vDB = Vector512.Load(pdB + f);
                    var vGrad = Vector512.Load(rowPreGradMat + f);
                    (vDB + vGrad).Store(pdB + f);
                }
            }
            else if (IsAvx2Supported)
            {
                int vecLimit = filters - (filters % Avx256Size);
                for (; f < vecLimit; f += Avx256Size)
                {
                    var vDB = Avx.LoadVector256(pdB + f);
                    var vGrad = Avx.LoadVector256(rowPreGradMat + f);
                    (vDB + vGrad).Store(pdB + f);
                }
            }

            for (; f < filters; f++)
            {
                pdB[f] += rowPreGradMat[f];
            }
        }
    }

    private static void ConvertPregradToMatrix(CnnMatrix preGrad, NeuralMatrix preGradMatrix)
    {
        int outH = preGrad.Height;
        int outW = preGrad.Width;
        int patches = preGrad.Batch * outH * outW;
        int filters = preGrad.Channels;
        var pPreGrad = preGrad.Pointer;
        var pPreGradMat = preGradMatrix.Pointer;
        var preGradMatStride = preGradMatrix.ColumnsStride;

        var spatialSize = outH * outW;
        var batchStride = filters * spatialSize;

        for (int b = 0; b < preGrad.Batch; b++)
        {
            float* pBatch = pPreGrad + b * batchStride;
            int patchBase = b * spatialSize;

            for (int f = 0; f < filters; f++)
            {
                float* pFilterSrc = pBatch + f * spatialSize;
                for (int spatialIdx = 0; spatialIdx < spatialSize; spatialIdx++)
                {
                    pPreGradMat[(patchBase + spatialIdx) * preGradMatStride + f] = pFilterSrc[spatialIdx];
                }
            }
        }
    }

    private void ComputePreGradient(CnnLayerConfig layer, CnnMatrix preGrad, CnnMatrix postAct, CnnMatrix convGrad)
    {
        ApplyDerivativeDirect(convGrad, postAct, preGrad, layer.Activation);
    }

    private static void ApplyDerivativeDirect(CnnMatrix gradient, CnnMatrix postAct, CnnMatrix dest, ActivationType type)
    {
        int totalElements = gradient.UnsafeSize;
        float* pGrad = gradient.Pointer;
        float* pPost = postAct.Pointer;
        float* pDst = dest.Pointer;

        int i = 0;

        if (type == ActivationType.ReLU)
        {
            if (IsAvx512Supported)
            {
                Vector512<float> vZero = Vector512<float>.Zero;
                int limit = totalElements - (totalElements % Avx512Size);
                for (; i < limit; i += Avx512Size)
                {
                    Vector512<float> g = Vector512.Load(pGrad + i);
                    Vector512<float> p = Vector512.Load(pPost + i);
                    Vector512<float> mask = Vector512.GreaterThan(p, vZero);
                    (g & mask).Store(pDst + i);
                }
            }
            else if (IsAvx2Supported)
            {
                int end = totalElements & (Avx256Size - 1);
                for (; i < end; i += Avx256Size)
                {
                    Avx.And(
                        Avx.LoadVector256(pGrad + i),
                        Avx.CompareGreaterThan(
                            Avx.LoadVector256(pPost + i),
                            Vector256<float>.Zero
                        )
                    ).Store(pDst + i);
                }
            }
        }

        for (; i < totalElements; i++)
        {
            float p = pPost[i];
            float g = pGrad[i];
            pDst[i] = type switch
            {
                ActivationType.ReLU => p <= 0 ? 0 : g,
                ActivationType.LeakyReLU => p <= 0 ? 0.01f * g : g,
                ActivationType.Sigmoid => g * p * (1.0f - p),
                ActivationType.Tanh => g * (1.0f - p * p),
                _ => g
            };
        }
    }

    private void BackPropagateThroughPool(CnnMatrix currentGrad, CnnLayerConfig layer, CnnMatrix gradInput, NeuralMatrix indices)
    {
        MaxPoolBackward(currentGrad, gradInput, indices, layer.PoolSize);
    }

    private void BulkMemoryCopy()
    {
        var denseGradInput = _reverseDenseHyperParameters[0].GradInput;
        float* pDenseGrad = denseGradInput.Pointer;
        float* pPooledGrad = _pooledOutputGrad.Pointer;

        int denseStride = denseGradInput.ColumnsStride;
        int spatialDim = _pooledOutputGrad.Channels * _pooledOutputGrad.Height * _pooledOutputGrad.Width;

        for (int b = 0; b < _pooledOutputGrad.Batch; b++)
        {
            float* srcRow = pDenseGrad + b * denseStride;
            float* dstRow = pPooledGrad + b * spatialDim;
            nuint bytesToCopy = (nuint)spatialDim * sizeof(float);
            NativeMemory.Copy(srcRow, dstRow, bytesToCopy);
        }
    }

    private void DenseBackWardClipped(float learningRate)
    {
        DenseBackward(learningRate, skipLastDerivative: true);

        if (_convHyperParameters[^1].Input is null)
        {
            throw new InvalidOperationException("_lastPooledOutput is null.");
        }
    }

    private void LossGradientVectorized(NeuralMatrix target)
    {
        var probabilities = _denseHyperParameters[^1].PostAct;
        int rows = probabilities.Rows;
        int cols = probabilities.UsedColumns;

        float* pProb = probabilities.Pointer;
        float* pTarg = target.Pointer;
        float* pGrad = _outputGrad.Pointer;

        int probStride = probabilities.ColumnsStride;
        int targStride = target.ColumnsStride;
        int gradStride = _outputGrad.ColumnsStride;

        Vector512<float> vInvBatch512 = Vector512.Create(1.0f / rows);
        Vector256<float> vInvBatch256 = Vector256.Create(1.0f / rows);
        float invBatch = 1.0f / rows;

        for (int r = 0; r < rows; r++)
        {
            float* rowP = pProb + r * probStride;
            float* rowT = pTarg + r * targStride;
            float* rowG = pGrad + r * gradStride;

            int c = 0;
            if (IsAvx512Supported)
            {
                int vecLimit = cols - (cols % Avx512Size);
                for (; c < vecLimit; c += Avx512Size)
                {
                    var vP = Vector512.Load(rowP + c);
                    var vT = Vector512.Load(rowT + c);
                    var vDiff = vP - vT;
                    (vDiff * vInvBatch512).Store(rowG + c);
                }
            }
            else if (IsAvx2Supported)
            {
                int vecLimit = cols - (cols % Avx256Size);
                for (; c < vecLimit; c += Avx256Size)
                {
                    var vP = Avx.LoadVector256(rowP + c);
                    var vT = Avx.LoadVector256(rowT + c);
                    var vDiff = Avx.Subtract(vP, vT);
                    Avx.Multiply(vDiff, vInvBatch256).Store(rowG + c);
                }
            }

            for (; c < cols; c++)
            {
                rowG[c] = (rowP[c] - rowT[c]) * invBatch;
            }
        }
    }

    private void ForwardPoolingPass(ref CnnMatrix current)
    {
        for (var layerIdx = 0; layerIdx < _cnnConfig.ConvLayers.Count; layerIdx++)
        {
            var layer = _cnnConfig.ConvLayers[layerIdx];
            var input = _convHyperParameters[layerIdx].Input;
            var prefix = $"FwdPool[{layerIdx}]";

            using (Perf?.Measure($"{prefix}.CopyInput"))
                input.CopyFrom(current);

            ConvForward(current, layerIdx);

            var preAct = _convHyperParameters[layerIdx].PreAct;
            var postAct = _convHyperParameters[layerIdx].PostAct;

            using (Perf?.Measure($"{prefix}.CopyPostAct"))
                postAct.CopyFrom(preAct);

            using (Perf?.Measure($"{prefix}.Activation"))
                ApplyActivation(postAct, layer.Activation);

            var poolIndices = _convHyperParameters[layerIdx].PoolIndices;

            using (Perf?.Measure($"{prefix}.MaxPoolForward"))
                MaxPoolForward(postAct, poolIndices, input, layer.PoolSize);

            current = input;
        }

        using (Perf?.Measure("FwdPool.Flatten"))
            Flatten(current);

        DenseForward(storeIntermediates: true);
    }

    private void ConvForward(CnnMatrix current, int layerIdx)
    {
        var colInput = _convHyperParameters[layerIdx].ColInput;
        var layer = _cnnConfig.ConvLayers[layerIdx];
        var weights = _convHyperParameters[layerIdx].Weights;
        var flattenedWeights = _convHyperParameters[layerIdx].FlattenedWeights;
        var preAct = _convHyperParameters[layerIdx].PreAct;
        var biases = _convHyperParameters[layerIdx].Biases;
        var convolution = _convHyperParameters[layerIdx].Convolution;

        var prefix = $"Conv[{layerIdx}]";

        using (Perf?.Measure($"{prefix}.Im2Col"))
            current.Im2Col(colInput, layer.KernelHeight, layer.KernelWidth, layer.Stride, layer.Padding);

        using (Perf?.Measure($"{prefix}.UpdateFlattenWeights"))
            UpdateFlattenConvWeights(weights, flattenedWeights);

        using (Perf?.Measure($"{prefix}.ComputeConvolution"))
            ComputeConvolution(colInput, flattenedWeights, convolution, layerIdx);

        using (Perf?.Measure($"{prefix}.AddBias"))
            AddBias(convolution, biases);

        using (Perf?.Measure($"{prefix}.FillToCnnMatrix"))
            FillToCnnMatrix(convolution, preAct);
    }

    private void UpdateFlattenConvWeights(CnnMatrix weights, NeuralMatrix flattenedWeights)
    {
        var innerDim = flattenedWeights.UsedColumns;
        var filters = flattenedWeights.Rows;

        var pSrc = weights.Pointer;
        var pDst = flattenedWeights.Pointer;
        var dstStride = flattenedWeights.ColumnsStride;

        if (dstStride == innerDim)
        {
            nuint totalBytes = (nuint)((long)filters * innerDim * sizeof(float));
            NativeMemory.Copy(pSrc, pDst, totalBytes);
            return;
        }

        nuint bytesPerRow = (nuint)((long)innerDim * sizeof(float));

        for (int f = 0; f < filters; f++)
        {
            float* srcRow = pSrc + (f * innerDim);
            float* dstRow = pDst + (f * dstStride);
            NativeMemory.Copy(srcRow, dstRow, bytesPerRow);
        }
    }

    private NeuralMatrix FlattenConvWeights(CnnMatrix weights)
    {
        var innerDim = weights.Channels * weights.Height * weights.Width;
        var filters = weights.Batch;
        var weightMat = RentNeural(filters, innerDim);

        return weightMat;
    }

    private void ComputeConvolution(NeuralMatrix colInput, NeuralMatrix flattenedWeights, NeuralMatrix convolution, int layerIdx)
    {
        int patches = colInput.Rows;
        int filters = flattenedWeights.Rows;
        int innerDim = colInput.UsedColumns;

        if (EnableGpu)
        {
            var allocation = _cublasConvAllocations[layerIdx].Convolution;

            GpuMatrixOps.RowMajorSgemmHostStaged(
                allocation,
                patches, filters, innerDim,
                colInput.Pointer,
                flattenedWeights.Pointer,
                convolution.Pointer);
        }
        else
        {
            float* colPtr = colInput.Pointer;
            float* weightPtr = flattenedWeights.Pointer;
            float* resPtr = convolution.Pointer;
            int colStride = colInput.ColumnsStride;
            int weightStride = flattenedWeights.ColumnsStride;
            int resStride = convolution.ColumnsStride;

            for (int patch = 0; patch < patches; patch++)
            {
                float* colRow = colPtr + patch * colStride;
                float* resRow = resPtr + patch * resStride;

                for (int f = 0; f < filters; f++)
                {
                    float* weightRow = weightPtr + f * weightStride;
                    float sum = 0;
                    int inner = 0;

                    if (Avx512F.IsSupported)
                    {
                        Vector512<float> sumVec = Vector512<float>.Zero;
                        int vectorizable = innerDim - (innerDim % Avx512Size);
                        for (; inner < vectorizable; inner += Avx512Size)
                        {
                            var colVec = Avx512F.LoadAlignedVector512(colRow + inner);
                            var weightVec = Avx512F.LoadAlignedVector512(weightRow + inner);
                            sumVec = Avx512F.FusedMultiplyAdd(colVec, weightVec, sumVec);
                        }
                        sum = Vector512.Sum(sumVec);
                    }
                    else if (Avx2.IsSupported)
                    {
                        Vector256<float> sumVec = Vector256<float>.Zero;
                        int vectorizable = innerDim - (innerDim % Avx256Size);
                        for (; inner < vectorizable; inner += Avx256Size)
                        {
                            var colVec = Avx2.LoadAlignedVector256(colRow + inner);
                            var weightVec = Avx2.LoadAlignedVector256(weightRow + inner);
                            sumVec = Fma.MultiplyAdd(colVec, weightVec, sumVec);
                        }
                        sum = Vector256.Sum(sumVec);
                    }

                    for (; inner < innerDim; inner++)
                    {
                        sum += colRow[inner] * weightRow[inner];
                    }

                    resRow[f] = sum;
                }
            }
        }
    }

    private void AddBias(NeuralMatrix result, CnnMatrix biases)
    {
        int patches = result.Rows;
        int filters = biases.Channels;
        int resStride = result.ColumnsStride;
        float* resPtr = result.Pointer;
        float* biasPtr = biases.Pointer;

        for (int patch = 0; patch < patches; patch++)
        {
            float* row = resPtr + patch * resStride;
            int f = 0;

            if (IsAvx512Supported)
            {
                int limit = filters - (filters % Avx512Size);
                for (; f < limit; f += Avx512Size)
                {
                    (Vector512.Load(row + f) + Vector512.Load(biasPtr + f)).Store(row + f);
                }
            }
            else if (IsAvx2Supported)
            {
                int limit = filters - (filters % Avx256Size);
                for (; f < limit; f += Avx256Size)
                {
                    (Avx.LoadVector256(row + f) + Avx.LoadVector256(biasPtr + f)).Store(row + f);
                }
            }

            for (; f < filters; f++)
            {
                row[f] += biasPtr[f];
            }
        }
    }

    private void FillToCnnMatrix(NeuralMatrix result, CnnMatrix preAct)
    {
        float* pSrc = result.Pointer;
        float* pDst = preAct.Pointer;

        int srcStride = result.ColumnsStride;
        int spatialSize = preAct.Height * preAct.Width;
        int batchStrideDst = preAct.Channels * spatialSize;

        for (int b = 0; b < preAct.Batch; b++)
        {
            int batchOffsetSrc = b * spatialSize;
            float* pDstBatch = pDst + (b * batchStrideDst);

            for (int f = 0; f < preAct.Channels; f++)
            {
                float* pDstChannel = pDstBatch + (f * spatialSize);
                for (int spatialIdx = 0; spatialIdx < spatialSize; spatialIdx++)
                {
                    pDstChannel[spatialIdx] = pSrc[(batchOffsetSrc + spatialIdx) * srcStride + f];
                }
            }
        }
    }

    private CnnSize GetCnnSize(int batchSize, int filters, int height, int width, CnnLayerConfig layer)
    {
        int outH = (height + 2 * layer.Padding - layer.KernelHeight) / layer.Stride + 1;
        int outW = (width + 2 * layer.Padding - layer.KernelWidth) / layer.Stride + 1;

        return new(batchSize, filters, outH, outW);
    }

    private void MaxPoolForwardInPlace(CnnMatrix preAct, CnnMatrix pooled, int poolSize)
    {
        int batch = preAct.Batch;
        int channels = preAct.Channels;
        int inH = preAct.Height;
        int inW = preAct.Width;

        int outH = inH / poolSize;
        int outW = inW / poolSize;

        float* pIn = preAct.Pointer;
        float* pOut = pooled.Pointer;

        int spatialInSize = inH * inW;
        int spatialOutSize = outH * outW;
        int numSlices = batch * channels;

        for (int slice = 0; slice < numSlices; slice++)
        {
            float* sliceIn = pIn + (slice * spatialInSize);
            float* sliceOut = pOut + (slice * spatialOutSize);

            for (int oh = 0; oh < outH; oh++)
            {
                int yStart = oh * poolSize;
                for (int ow = 0; ow < outW; ow++)
                {
                    int xStart = ow * poolSize;
                    float maxVal = float.NegativeInfinity;

                    for (int dy = 0; dy < poolSize; dy++)
                    {
                        float* rowPtr = sliceIn + ((yStart + dy) * inW);
                        for (int dx = 0; dx < poolSize; dx++)
                        {
                            float val = rowPtr[xStart + dx];
                            if (val > maxVal) maxVal = val;
                        }
                    }

                    sliceOut[oh * outW + ow] = maxVal;
                }
            }
        }
    }

    private void MaxPoolForward(CnnMatrix postAct, NeuralMatrix indices, CnnMatrix pooled, int poolSize)
    {
        int batch = postAct.Batch;
        int channels = postAct.Channels;
        int inH = postAct.Height;
        int inW = postAct.Width;

        int outH = inH / poolSize;
        int outW = inW / poolSize;

        float* pIn = postAct.Pointer;
        float* pOut = pooled.Pointer;
        float* pIdx = indices.Pointer;

        int spatialInSize = inH * inW;
        int spatialOutSize = outH * outW;
        int numSlices = batch * channels;

        // Fast path for the common poolSize == 2 case.
        if (poolSize == 2)
        {
            for (int slice = 0; slice < numSlices; slice++)
            {
                float* sliceIn = pIn + (slice * spatialInSize);
                float* sliceOut = pOut + (slice * spatialOutSize);
                float* sliceIdx = pIdx + (slice * spatialOutSize);

                for (int oh = 0; oh < outH; oh++)
                {
                    int y0 = oh * 2;
                    int y1 = y0 + 1;
                    float* row0 = sliceIn + y0 * inW;
                    float* row1 = sliceIn + y1 * inW;
                    int outBase = oh * outW;
                    int idxRow0 = y0 * inW;
                    int idxRow1 = y1 * inW;

                    for (int ow = 0; ow < outW; ow++)
                    {
                        int x = ow * 2;
                        float a = row0[x];
                        float b = row0[x + 1];
                        float c = row1[x];
                        float d = row1[x + 1];

                        // Pairwise max tree
                        float ab = a > b ? a : b;
                        float cd = c > d ? c : d;
                        bool topWins = ab > cd;
                        float maxVal = topWins ? ab : cd;

                        // Index resolution: 2 comparisons per output pixel total
                        int maxIdx;
                        if (topWins)
                        {
                            maxIdx = (a > b) ? idxRow0 + x : idxRow0 + x + 1;
                        }
                        else
                        {
                            maxIdx = (c > d) ? idxRow1 + x : idxRow1 + x + 1;
                        }

                        int outOffset = outBase + ow;
                        sliceOut[outOffset] = maxVal;
                        sliceIdx[outOffset] = maxIdx;
                    }
                }
            }
            return;
        }

        // Generic path for poolSize != 2 (unchanged semantics, minor cleanup).
        for (int slice = 0; slice < numSlices; slice++)
        {
            float* sliceIn = pIn + (slice * spatialInSize);
            float* sliceOut = pOut + (slice * spatialOutSize);
            float* sliceIdx = pIdx + (slice * spatialOutSize);

            int outIdx = 0;

            for (int oh = 0; oh < outH; oh++)
            {
                int yStart = oh * poolSize;

                for (int ow = 0; ow < outW; ow++)
                {
                    int xStart = ow * poolSize;

                    float maxVal = float.NegativeInfinity;
                    int maxIdx = 0;

                    for (int dy = 0; dy < poolSize; dy++)
                    {
                        int y = yStart + dy;
                        float* rowPtr = sliceIn + y * inW;
                        int rowBase = y * inW;

                        for (int dx = 0; dx < poolSize; dx++)
                        {
                            int x = xStart + dx;
                            float val = rowPtr[x];

                            if (val > maxVal)
                            {
                                maxVal = val;
                                maxIdx = rowBase + x;
                            }
                        }
                    }

                    sliceOut[outIdx] = maxVal;
                    sliceIdx[outIdx] = maxIdx;
                    outIdx++;
                }
            }
        }
    }

    private CnnMatrix MaxPoolBackward(CnnMatrix gradOutput, CnnMatrix gradInput, NeuralMatrix indices, int poolSize)
    {
        int batch = gradInput.Batch;
        int channels = gradInput.Channels;
        int inH = gradInput.Height;
        int inW = gradInput.Width;

        int outH = gradOutput.Height;
        int outW = gradOutput.Width;

        int totalInputElements = batch * channels * inH * inW;

        float* pGradOut = gradOutput.Pointer;
        float* pGradIn = gradInput.Pointer;
        float* pIndices = indices.Pointer;

        int spatialInSize = inH * inW;
        int spatialOutSize = outH * outW;

        int numSlices = batch * channels;

        for (int slice = 0; slice < numSlices; slice++)
        {
            float* sliceGradOut = pGradOut + (slice * spatialOutSize);
            float* sliceGradIn = pGradIn + (slice * spatialInSize);
            float* sliceIndices = pIndices + (slice * spatialOutSize);

            for (int i = 0; i < spatialOutSize; i++)
            {
                int maxIdx = (int)sliceIndices[i];
                sliceGradIn[maxIdx] += sliceGradOut[i];
            }
        }

        return gradInput;
    }

    private void DenseForward(bool storeIntermediates)
    {
        var current = _flattenedInput!;

        for (int i = 0; i < _denseHyperParameters.Count; i++)
        {
            var weights = _denseHyperParameters[i].Weights;
            var biases = _denseHyperParameters[i].Biases;

            int batchSize = current.Rows;
            int inFeatures = current.UsedColumns;
            int outFeatures = weights.Rows;

            var result = _denseLayerMatrixes[i];

            var prefix = $"DenseFwd[{i}]";

            if (EnableGpu)
            {
                var allocation = _cublasDenseAllocations[i].Layer;

                using (Perf?.Measure($"{prefix}.Gemm"))
                {
                    GpuMatrixOps.RowMajorSgemmHostStaged(
                        allocation,
                        batchSize, outFeatures, inFeatures,
                        current.Pointer,
                        weights.Pointer,
                        result.Pointer);
                }

                using (Perf?.Measure($"{prefix}.BiasAdd"))
                {
                    for (int b = 0; b < batchSize; b++)
                    {
                        float* row = result.Pointer + b * result.ColumnsStride;
                        for (int f = 0; f < outFeatures; f++)
                        {
                            row[f] += biases.Pointer[f];
                        }
                    }
                }
            }
            else
            {
                using (Perf?.Measure($"{prefix}.Gemm"))
                {
                    float* inPtr = current.Pointer;
                    float* weightPtr = weights.Pointer;
                    float* biasPtr = biases.Pointer;
                    float* resPtr = result.Pointer;

                    int inStride = current.ColumnsStride;
                    int weightStride = weights.ColumnsStride;
                    int resStride = result.ColumnsStride;

                    for (int r = 0; r < batchSize; r++)
                    {
                        float* inRow = inPtr + r * inStride;
                        float* resRow = resPtr + r * resStride;

                        int outNeuron = 0;

                        if (Avx512F.IsSupported)
                        {
                            int vecInFeatures = inFeatures - (inFeatures % Avx512Size);

                            for (; outNeuron <= outFeatures - 4; outNeuron += 4)
                            {
                                float* w0 = weightPtr + (outNeuron + 0) * weightStride;
                                float* w1 = weightPtr + (outNeuron + 1) * weightStride;
                                float* w2 = weightPtr + (outNeuron + 2) * weightStride;
                                float* w3 = weightPtr + (outNeuron + 3) * weightStride;

                                Vector512<float> acc0 = Vector512<float>.Zero;
                                Vector512<float> acc1 = Vector512<float>.Zero;
                                Vector512<float> acc2 = Vector512<float>.Zero;
                                Vector512<float> acc3 = Vector512<float>.Zero;

                                int inIdx = 0;
                                for (; inIdx < vecInFeatures; inIdx += Avx512Size)
                                {
                                    var vIn = Vector512.Load(inRow + inIdx);

                                    acc0 = Avx512F.FusedMultiplyAdd(vIn, Vector512.Load(w0 + inIdx), acc0);
                                    acc1 = Avx512F.FusedMultiplyAdd(vIn, Vector512.Load(w1 + inIdx), acc1);
                                    acc2 = Avx512F.FusedMultiplyAdd(vIn, Vector512.Load(w2 + inIdx), acc2);
                                    acc3 = Avx512F.FusedMultiplyAdd(vIn, Vector512.Load(w3 + inIdx), acc3);
                                }

                                resRow[outNeuron + 0] = Vector512.Sum(acc0) + biasPtr[outNeuron + 0];
                                resRow[outNeuron + 1] = Vector512.Sum(acc1) + biasPtr[outNeuron + 1];
                                resRow[outNeuron + 2] = Vector512.Sum(acc2) + biasPtr[outNeuron + 2];
                                resRow[outNeuron + 3] = Vector512.Sum(acc3) + biasPtr[outNeuron + 3];

                                for (; inIdx < inFeatures; inIdx++)
                                {
                                    float val = inRow[inIdx];
                                    resRow[outNeuron + 0] += val * w0[inIdx];
                                    resRow[outNeuron + 1] += val * w1[inIdx];
                                    resRow[outNeuron + 2] += val * w2[inIdx];
                                    resRow[outNeuron + 3] += val * w3[inIdx];
                                }
                            }

                            for (; outNeuron < outFeatures; outNeuron++)
                            {
                                float* wRow = weightPtr + outNeuron * weightStride;
                                Vector512<float> acc = Vector512<float>.Zero;

                                int inIdx = 0;
                                for (; inIdx < vecInFeatures; inIdx += Avx512Size)
                                {
                                    var vIn = Vector512.Load(inRow + inIdx);
                                    var vW = Vector512.Load(wRow + inIdx);
                                    acc = Avx512F.FusedMultiplyAdd(vIn, vW, acc);
                                }

                                float sum = Vector512.Sum(acc);
                                for (; inIdx < inFeatures; inIdx++)
                                {
                                    sum += inRow[inIdx] * wRow[inIdx];
                                }

                                resRow[outNeuron] = sum + biasPtr[outNeuron];
                            }
                        }
                        else if (Avx2.IsSupported)
                        {
                            int vecInFeatures = inFeatures - (inFeatures % Avx256Size);

                            for (; outNeuron <= outFeatures - 4; outNeuron += 4)
                            {
                                float* w0 = weightPtr + (outNeuron + 0) * weightStride;
                                float* w1 = weightPtr + (outNeuron + 1) * weightStride;
                                float* w2 = weightPtr + (outNeuron + 2) * weightStride;
                                float* w3 = weightPtr + (outNeuron + 3) * weightStride;

                                Vector256<float> acc0 = Vector256<float>.Zero;
                                Vector256<float> acc1 = Vector256<float>.Zero;
                                Vector256<float> acc2 = Vector256<float>.Zero;
                                Vector256<float> acc3 = Vector256<float>.Zero;

                                int inIdx = 0;
                                for (; inIdx < vecInFeatures; inIdx += Avx256Size)
                                {
                                    var vIn = Avx2.LoadVector256(inRow + inIdx);

                                    acc0 = Fma.MultiplyAdd(vIn, Avx2.LoadVector256(w0 + inIdx), acc0);
                                    acc1 = Fma.MultiplyAdd(vIn, Avx2.LoadVector256(w1 + inIdx), acc1);
                                    acc2 = Fma.MultiplyAdd(vIn, Avx2.LoadVector256(w2 + inIdx), acc2);
                                    acc3 = Fma.MultiplyAdd(vIn, Avx2.LoadVector256(w3 + inIdx), acc3);
                                }

                                resRow[outNeuron + 0] = Vector256.Sum(acc0) + biasPtr[outNeuron + 0];
                                resRow[outNeuron + 1] = Vector256.Sum(acc1) + biasPtr[outNeuron + 1];
                                resRow[outNeuron + 2] = Vector256.Sum(acc2) + biasPtr[outNeuron + 2];
                                resRow[outNeuron + 3] = Vector256.Sum(acc3) + biasPtr[outNeuron + 3];

                                for (; inIdx < inFeatures; inIdx++)
                                {
                                    float val = inRow[inIdx];
                                    resRow[outNeuron + 0] += val * w0[inIdx];
                                    resRow[outNeuron + 1] += val * w1[inIdx];
                                    resRow[outNeuron + 2] += val * w2[inIdx];
                                    resRow[outNeuron + 3] += val * w3[inIdx];
                                }
                            }

                            for (; outNeuron < outFeatures; outNeuron++)
                            {
                                float* wRow = weightPtr + outNeuron * weightStride;
                                Vector256<float> acc = Vector256<float>.Zero;

                                int inIdx = 0;
                                for (; inIdx < vecInFeatures; inIdx += Avx256Size)
                                {
                                    var vIn = Avx2.LoadVector256(inRow + inIdx);
                                    var vW = Avx2.LoadVector256(wRow + inIdx);
                                    acc = Fma.MultiplyAdd(vIn, vW, acc);
                                }

                                float sum = Vector256.Sum(acc);
                                for (; inIdx < inFeatures; inIdx++)
                                {
                                    sum += inRow[inIdx] * wRow[inIdx];
                                }

                                resRow[outNeuron] = sum + biasPtr[outNeuron];
                            }
                        }
                        else
                        {
                            for (; outNeuron < outFeatures; outNeuron++)
                            {
                                float* wRow = weightPtr + outNeuron * weightStride;
                                float sum = 0f;

                                for (int inIdx = 0; inIdx < inFeatures; inIdx++)
                                {
                                    sum += inRow[inIdx] * wRow[inIdx];
                                }

                                resRow[outNeuron] = sum + biasPtr[outNeuron];
                            }
                        }
                    }
                }
            }

            if (storeIntermediates)
            {
                _denseHyperParameters[i].PreAct.CopyFrom(result);
            }

            using (Perf?.Measure($"{prefix}.Activation"))
                _denseActivations[i](result);

            if (storeIntermediates)
            {
                _denseHyperParameters[i].PostAct.CopyFrom(result);
            }

            current = result;
        }
    }

    private void DenseBackward(float learningRate, bool skipLastDerivative = false)
    {
        var gradOutput = _outputGrad;

        for (int i = _denseHyperParameters.Count - 1; i >= 0; i--)
        {
            var prefix = $"DenseBack[{i}]";

            var preAct = _denseHyperParameters[i].PreAct;
            var inputToLayer = (i == 0) ? _flattenedInput : _denseHyperParameters[i - 1].PostAct;

            if (inputToLayer == null)
            {
                throw new InvalidOperationException($"inputToLayer is null for layer {i}.");
            }

            int batch = gradOutput.Rows;
            int outDim = gradOutput.UsedColumns;
            int inDim = inputToLayer.UsedColumns;

            var gradPre = _reverseDenseHyperParameters[i].GradPre;

            float* pGradOut = gradOutput.Pointer;
            float* pGradPre = gradPre.Pointer;
            float* pPreAct = preAct.Pointer;
            int strideGradOut = gradOutput.ColumnsStride;
            int strideGradPre = gradPre.ColumnsStride;
            int stridePreAct = preAct.ColumnsStride;

            var derivativeFn = _denseDerivatives[i];
            bool skipDeriv = skipLastDerivative && (i == _denseHyperParameters.Count - 1);

            using (Perf?.Measure($"{prefix}.Derivative"))
            {
                for (int r = 0; r < batch; r++)
                {
                    float* rowGO = pGradOut + r * strideGradOut;
                    float* rowGP = pGradPre + r * strideGradPre;
                    float* rowPA = pPreAct + r * stridePreAct;

                    if (skipDeriv)
                    {
                        NativeMemory.Copy(rowGO, rowGP, (nuint)(outDim * sizeof(float)));
                    }
                    else
                    {
                        for (int c = 0; c < outDim; c++)
                        {
                            rowGP[c] = rowGO[c] * derivativeFn(rowPA[c]);
                        }
                    }
                }
            }

            var dW = _reverseDenseHyperParameters[i].DWeights;
            dW.Clear();

            if (EnableGpu)
            {
                var allocation = _cublasReverseDenseAllocations[i].DWeights;

                using (Perf?.Measure($"{prefix}.DWeights"))
                {
                    GpuMatrixOps.RowMajorSgemmHostStaged(
                        allocation,
                        inDim, outDim, batch,
                        inputToLayer.Pointer,
                        gradPre.Pointer,
                        dW.Pointer);
                }
            }
            else
            {
                using (Perf?.Measure($"{prefix}.DWeights"))
                {
                    float* pIn = inputToLayer.Pointer;
                    float* pDW = dW.Pointer;
                    int strideIn = inputToLayer.ColumnsStride;
                    int strideDW = dW.ColumnsStride;

                    for (int r = 0; r < batch; r++)
                    {
                        float* rowIn = pIn + r * strideIn;
                        float* rowGP = pGradPre + r * strideGradPre;

                        for (int cIn = 0; cIn < inDim; cIn++)
                        {
                            float xVal = rowIn[cIn];
                            if (xVal == 0f) continue;

                            float* rowDW = pDW + cIn * strideDW;
                            int cOut = 0;

                            if (Avx512F.IsSupported)
                            {
                                var vX = Vector512.Create(xVal);
                                int limit = outDim - (outDim % 16);
                                for (; cOut < limit; cOut += 16)
                                {
                                    var vGP = Vector512.Load(rowGP + cOut);
                                    var vDW = Vector512.Load(rowDW + cOut);
                                    Avx512F.FusedMultiplyAdd(vX, vGP, vDW).Store(rowDW + cOut);
                                }
                            }
                            else if (Avx2.IsSupported)
                            {
                                var vX = Vector256.Create(xVal);
                                int limit = outDim - (outDim % 8);
                                for (; cOut < limit; cOut += 8)
                                {
                                    var vGP = Avx.LoadVector256(rowGP + cOut);
                                    var vDW = Avx.LoadVector256(rowDW + cOut);
                                    Fma.MultiplyAdd(vX, vGP, vDW).Store(rowDW + cOut);
                                }
                            }

                            for (; cOut < outDim; cOut++)
                            {
                                rowDW[cOut] += xVal * rowGP[cOut];
                            }
                        }
                    }
                }
            }

            var dB = _reverseDenseHyperParameters[i].DBiases;

            dB.Clear();
            float* pDB = dB.Pointer;

            using (Perf?.Measure($"{prefix}.DBiases"))
            {
                for (int r = 0; r < batch; r++)
                {
                    float* rowGP = pGradPre + r * strideGradPre;
                    int cOut = 0;
                    if (IsAvx512Supported)
                    {
                        int limit = outDim - (outDim % 16);
                        for (; cOut < limit; cOut += 16)
                        {
                            (Vector512.Load(pDB + cOut) + Vector512.Load(rowGP + cOut)).Store(pDB + cOut);
                        }
                    }
                    else if (IsAvx2Supported)
                    {
                        int limit = outDim - (outDim % 8);
                        for (; cOut < limit; cOut += 8)
                        {
                            (Avx.LoadVector256(pDB + cOut) + Avx.LoadVector256(rowGP + cOut)).Store(pDB + cOut);
                        }
                    }
                    for (; cOut < outDim; cOut++)
                    {
                        pDB[cOut] += rowGP[cOut];
                    }
                }
            }

            using (Perf?.Measure($"{prefix}.AdamUpdate"))
                _denseOptimizers[i].Update(_denseHyperParameters[i].Weights, _denseHyperParameters[i].Biases, dW, dB);

            var weights = _denseHyperParameters[i].Weights;
            int weightOutDim = weights.Rows;
            int weightInDim = weights.UsedColumns;
            var gradInput = _reverseDenseHyperParameters[i].GradInput;

            if (EnableGpu)
            {
                var allocation = _cublasReverseDenseAllocations[i].GradInput;

                using (Perf?.Measure($"{prefix}.GradInput"))
                {
                    GpuMatrixOps.RowMajorSgemmHostStaged(
                        allocation,
                        batch, weightInDim, weightOutDim,
                        gradPre.Pointer,
                        weights.Pointer,
                        gradInput.Pointer);
                }
            }
            else   // CPU path
            {
                using (Perf?.Measure($"{prefix}.GradInput"))
                {
                    float* pWeights = weights.Pointer;
                    float* pGradInput = gradInput.Pointer;
                    int strideWeights = weights.ColumnsStride;
                    int strideGradInput = gradInput.ColumnsStride;

                    for (int r = 0; r < batch; r++)
                    {
                        float* rowGP = pGradPre + r * strideGradPre;
                        float* rowGI = pGradInput + r * strideGradInput;

                        for (int cOut = 0; cOut < weightOutDim; cOut++)
                        {
                            float gVal = rowGP[cOut];
                            if (gVal == 0f) continue;

                            float* rowW = pWeights + cOut * strideWeights;
                            int cIn = 0;

                            if (Avx512F.IsSupported)
                            {
                                var vG = Vector512.Create(gVal);
                                int limit = weightInDim - (weightInDim % 16);
                                for (; cIn < limit; cIn += 16)
                                {
                                    var vW = Vector512.LoadUnsafe(ref rowW[cIn]);
                                    var vGI = Vector512.LoadUnsafe(ref rowGI[cIn]);
                                    var vOut = Avx512F.FusedMultiplyAdd(vG, vW, vGI);
                                    Vector512.StoreUnsafe(vOut, ref rowGI[cIn]);
                                }
                            }
                            else if (Avx2.IsSupported)
                            {
                                var vG = Vector256.Create(gVal);
                                int limit = weightInDim - (weightInDim % 8);
                                for (; cIn < limit; cIn += 8)
                                {
                                    var vW = Avx.LoadVector256(rowW + cIn);
                                    var vGI = Avx.LoadVector256(rowGI + cIn);
                                    var vOut = Fma.MultiplyAdd(vG, vW, vGI);
                                    Avx.Store(rowGI + cIn, vOut);
                                }
                            }

                            for (; cIn < weightInDim; cIn++)
                            {
                                rowGI[cIn] += gVal * rowW[cIn];
                            }
                        }
                    }
                }
            }

            gradOutput = gradInput;
        }
    }

    private void ApplyActivation(CnnMatrix matrix, ActivationType type)
    {
        if (type == ActivationType.Identity) return;

        int totalElements = matrix.Batch * matrix.Channels * matrix.Height * matrix.Width;
        float* ptr = matrix.Pointer;

        int i = 0;

        if (type == ActivationType.ReLU)
        {
            if (IsAvx512Supported)
            {
                int vSize = Vector512<float>.Count;
                int vectorizable = totalElements - (totalElements % vSize);
                Vector512<float> zero = Vector512<float>.Zero;
                for (; i < vectorizable; i += vSize)
                {
                    Vector512<float> vec = Vector512.Load(ptr + i);
                    Vector512.Max(vec, zero).Store(ptr + i);
                }
            }
            else if (IsAvx2Supported)
            {
                int vSize = Vector256<float>.Count;
                int vectorizable = totalElements - (totalElements % vSize);
                Vector256<float> zero = Vector256<float>.Zero;
                for (; i < vectorizable; i += vSize)
                {
                    Vector256<float> vec = Avx.LoadVector256(ptr + i);
                    Avx.Max(vec, zero).Store(ptr + i);
                }
            }
        }
        else if (type == ActivationType.LeakyReLU)
        {
            if (IsAvx512Supported)
            {
                int vSize = Vector512<float>.Count;
                int vectorizable = totalElements - (totalElements % vSize);
                Vector512<float> zero = Vector512<float>.Zero;
                Vector512<float> alpha = Vector512.Create(0.01f);
                for (; i < vectorizable; i += vSize)
                {
                    Vector512<float> vec = Vector512.Load(ptr + i);
                    Vector512<float> scaled = Vector512.Multiply(vec, alpha);
                    Vector512.ConditionalSelect(Vector512.GreaterThan(vec, zero), vec, scaled).Store(ptr + i);
                }
            }
            else if (IsAvx2Supported)
            {
                int vSize = Vector256<float>.Count;
                int vectorizable = totalElements - (totalElements % vSize);
                Vector256<float> zero = Vector256<float>.Zero;
                Vector256<float> alpha = Vector256.Create(0.01f);
                for (; i < vectorizable; i += vSize)
                {
                    Vector256<float> vec = Avx.LoadVector256(ptr + i);
                    Vector256<float> scaled = Avx.Multiply(vec, alpha);
                    var mask = Avx.Compare(vec, zero, FloatComparisonMode.OrderedGreaterThanNonSignaling);
                    Avx.BlendVariable(scaled, vec, mask).Store(ptr + i);
                }
            }
        }

        for (; i < totalElements; i++)
        {
            float val = ptr[i];
            ptr[i] = type switch
            {
                ActivationType.ReLU => val < 0 ? 0 : val,
                ActivationType.LeakyReLU => val < 0 ? 0.01f * val : val,
                ActivationType.Sigmoid => 1.0f / (1.0f + MathF.Exp(-val)),
                ActivationType.Tanh => MathF.Tanh(val),
                _ => val
            };
        }
    }

    private void Flatten(CnnMatrix input)
    {
        int featureDim = input.Channels * input.Height * input.Width;

        float* pSrc = input.Pointer;
        float* pDst = _flattenedInput!.Pointer;

        int srcStride = featureDim;
        int dstStride = _flattenedInput.ColumnsStride;

        if (srcStride == dstStride)
        {
            var totalBytes = (nuint)(input.Batch * featureDim * sizeof(float));
            NativeMemory.Copy(pSrc, pDst, totalBytes);
        }
        else
        {
            var bytesPerBatch = (nuint)featureDim * sizeof(float);

            for (int b = 0; b < input.Batch; b++)
            {
                var srcBatch = pSrc + (b * srcStride);
                var dstBatch = pDst + (b * dstStride);

                NativeMemory.Copy(srcBatch, dstBatch, bytesPerBatch);
            }
        }
    }

    private float ComputeCrossEntropyLoss(NeuralMatrix targets)
    {
        var predictions = _denseHyperParameters[^1].PostAct;
        var rows = predictions.Rows;
        var cols = predictions.UsedColumns;
        const float eps = 1e-7f;
        float totalLoss = 0f;

        float* pPred = predictions.Pointer;
        float* pTarg = targets.Pointer;

        int predStride = predictions.ColumnsStride;
        int targStride = targets.ColumnsStride;

        int badPredCount = 0;
        int emptyTargetRows = 0;

        for (int r = 0; r < rows; r++)
        {
            float* predRow = pPred + r * predStride;
            float* targRow = pTarg + r * targStride;

            float rowLoss = 0f;
            bool rowHasValidTarget = false;

            for (int c = 0; c < cols; c++)
            {
                float p = predRow[c];

                if (!(p >= 0f && p <= 1f))
                {
                    badPredCount++;
                    continue;
                }

                float pVal = p < eps ? eps : (p > 1f - eps ? 1f - eps : p);
                float tVal = targRow[c];

                if (tVal > 0f)
                {
                    rowHasValidTarget = true;
                    rowLoss -= tVal * MathF.Log(pVal);
                }
            }

            if (!rowHasValidTarget)
            {
                emptyTargetRows++;
                rowLoss = MathF.Log(cols);
            }

            totalLoss += rowLoss;
        }

        if (badPredCount > 0 || emptyTargetRows > 0)
        {
            Console.WriteLine(
                $"[CE] badPredictions={badPredCount} emptyTargetRows={emptyTargetRows} " +
                $"(batch={rows}, classes={cols})");
        }

        return totalLoss / rows;
    }

    /// <summary>
    /// Dumps the full architecture: per-layer config, recomputed shapes,
    /// and the actual live buffer dimensions held by the framework.
    /// </summary>
    public void DumpArchitecture(string tag = "")
    {
        var sb = new System.Text.StringBuilder();
        string hr = new string('=', 100);

        sb.AppendLine(hr);
        sb.AppendLine($"CNN ARCHITECTURE DUMP  {(string.IsNullOrEmpty(tag) ? "" : $"[{tag}]")}");
        sb.AppendLine(hr);

        sb.AppendLine($"baseConfig       : OptimizerConfig={_cnnConfig.OptimizerConfig}");
        sb.AppendLine($"input (_input)   : batch={_input.BatchSize} ch={_input.Channels} h={_input.Height} w={_input.Width}");
        sb.AppendLine($"convLayers       : {_cnnConfig.ConvLayers.Count}");
        sb.AppendLine($"denseArch (cfg)  : [{string.Join(", ", _cnnConfig.DenseArchitecture)}]");
        sb.AppendLine($"hiddenAct        : {_cnnConfig.DenseHiddenActivation}");
        sb.AppendLine($"outputAct        : {_cnnConfig.OutputActivation}");
        sb.AppendLine($"SIMD             : AVX512={IsAvx512Supported} AVX2={IsAvx2Supported} GPU={EnableGpu}");
        sb.AppendLine();

        int h = _input.Height, w = _input.Width, ch = _input.Channels;
        int batch = _input.BatchSize;

        sb.AppendLine("---- LAYER CONFIG & RECOMPUTED SHAPES ----");
        for (int i = 0; i < _cnnConfig.ConvLayers.Count; i++)
        {
            var L = _cnnConfig.ConvLayers[i];
            int paddedH = h + 2 * L.Padding;
            int paddedW = w + 2 * L.Padding;
            int outH = (paddedH - L.KernelHeight) / L.Stride + 1;
            int outW = (paddedW - L.KernelWidth) / L.Stride + 1;
            int pooledH = L.UseMaxPool ? outH / L.PoolSize : outH;
            int pooledW = L.UseMaxPool ? outW / L.PoolSize : outW;

            sb.AppendLine($"  conv[{i}]:");
            sb.AppendLine($"    in       : {batch}x{ch}x{h}x{w}");
            sb.AppendLine($"    kernel   : {L.KernelHeight}x{L.KernelWidth}  stride={L.Stride}  pad={L.Padding}");
            sb.AppendLine($"    filters  : {L.Filters}");
            sb.AppendLine($"    pool     : use={L.UseMaxPool} size={L.PoolSize}");
            sb.AppendLine($"    act      : {L.Activation}");
            sb.AppendLine($"    convOut  : {batch}x{L.Filters}x{outH}x{outW}");
            sb.AppendLine($"    pooled   : {batch}x{L.Filters}x{pooledH}x{pooledW}");
            int patchSize = ch * L.KernelHeight * L.KernelWidth;
            int totalPatches = batch * outH * outW;
            sb.AppendLine($"    colInput : ({totalPatches},{patchSize})");
            sb.AppendLine($"    wgtMat   : ({L.Filters},{patchSize})");

            ch = L.Filters;
            h = pooledH;
            w = pooledW;
        }

        int flatSize = ch * h * w;
        sb.AppendLine($"  flattenedSize (recomputed) : {flatSize}   ({batch}x{flatSize})");

        sb.AppendLine();
        sb.AppendLine("---- LIVE convHyperParameters ----");
        for (int i = 0; i < _convHyperParameters.Count; i++)
        {
            var p = _convHyperParameters[i];
            sb.AppendLine($"  conv[{i}]:");
            sb.AppendLine($"    Input              : {Fmt(p.Input)}");
            sb.AppendLine($"    ColInput           : {Fmt(p.ColInput)}");
            sb.AppendLine($"    Weights (Cnn)      : {Fmt(p.Weights)}");
            sb.AppendLine($"    FlattenedWeights   : {Fmt(p.FlattenedWeights)}");
            sb.AppendLine($"    Biases             : {Fmt(p.Biases)}");
            sb.AppendLine($"    PreAct             : {Fmt(p.PreAct)}");
            sb.AppendLine($"    PostAct            : {Fmt(p.PostAct)}");
            sb.AppendLine($"    PoolIndices        : {Fmt(p.PoolIndices)}");
        }

        sb.AppendLine();
        sb.AppendLine("---- LIVE denseHyperParameters ----");
        for (int i = 0; i < _denseHyperParameters.Count; i++)
        {
            var p = _denseHyperParameters[i];
            sb.AppendLine($"  dense[{i}]:");
            sb.AppendLine($"    Weights   : {Fmt(p.Weights)}");
            sb.AppendLine($"    Biases    : {Fmt(p.Biases)}");
            sb.AppendLine($"    PreAct    : {Fmt(p.PreAct)}");
            sb.AppendLine($"    PostAct   : {Fmt(p.PostAct)}");
            sb.AppendLine($"    act       : {_denseActivations[i].Method.Name}");
        }

        sb.AppendLine();
        sb.AppendLine($"  _flattenedInput  : {Fmt(_flattenedInput)}");
        sb.AppendLine($"  _lastPooledOut   : {Fmt(_convHyperParameters[^1].Input)}");

        sb.AppendLine();
        sb.AppendLine("---- CONSISTENCY CHECKS ----");

        int chkH = _input.Height, chkW = _input.Width, chkC = _input.Channels;
        for (int i = 0; i < _cnnConfig.ConvLayers.Count; i++)
        {
            var L = _cnnConfig.ConvLayers[i];
            var p = _convHyperParameters[i];

            int inH = chkH, inW = chkW, inC = chkC;
            int padH = chkH + 2 * L.Padding;
            int padW = chkW + 2 * L.Padding;
            int cOutH = (padH - L.KernelHeight) / L.Stride + 1;
            int cOutW = (padW - L.KernelWidth) / L.Stride + 1;
            int pOutH = L.UseMaxPool ? cOutH / L.PoolSize : cOutH;
            int pOutW = L.UseMaxPool ? cOutW / L.PoolSize : cOutW;
            int patchSize = inC * L.KernelHeight * L.KernelWidth;
            int totalPatches = _input.BatchSize * cOutH * cOutW;

            bool ok = true;
            ok &= Check(sb, $"conv[{i}].Input          expects {_input.BatchSize}x{inC}x{inH}x{inW}",
                        p.Input.Batch == _input.BatchSize && p.Input.Channels == inC
                        && p.Input.Height == inH && p.Input.Width == inW, p.Input);
            ok &= Check(sb, $"conv[{i}].PooledOutput   expects {_input.BatchSize}x{L.Filters}x{pOutH}x{pOutW}",
                        p.ColInput.Rows == totalPatches && p.ColInput.UsedColumns == patchSize, p.ColInput);
            ok &= Check(sb, $"conv[{i}].Weights        expects {L.Filters}x{inC}x{L.KernelHeight}x{L.KernelWidth}",
                        p.Weights.Batch == L.Filters && p.Weights.Channels == inC
                        && p.Weights.Height == L.KernelHeight && p.Weights.Width == L.KernelWidth, p.Weights);
            ok &= Check(sb, $"conv[{i}].FlattenedWeights expects ({L.Filters},{patchSize})",
                        p.FlattenedWeights.Rows == L.Filters && p.FlattenedWeights.UsedColumns == patchSize, p.FlattenedWeights);
            ok &= Check(sb, $"conv[{i}].Biases         expects (1,{L.Filters},1,1)",
                        p.Biases.Batch == 1 && p.Biases.Channels == L.Filters
                        && p.Biases.Height == 1 && p.Biases.Width == 1, p.Biases);
            ok &= Check(sb, $"conv[{i}].PreAct         expects {_input.BatchSize}x{L.Filters}x{cOutH}x{cOutW}",
                        p.PreAct.Batch == _input.BatchSize && p.PreAct.Channels == L.Filters
                        && p.PreAct.Height == cOutH && p.PreAct.Width == cOutW, p.PreAct);
            ok &= Check(sb, $"conv[{i}].PostAct        expects {_input.BatchSize}x{L.Filters}x{cOutH}x{cOutW}",
                        p.PostAct.Batch == _input.BatchSize && p.PostAct.Channels == L.Filters
                        && p.PostAct.Height == cOutH && p.PostAct.Width == cOutW, p.PostAct);
            ok &= Check(sb, $"conv[{i}].PoolIndices    expects ({_input.BatchSize * L.Filters * pOutH * pOutW},1)",
                        p.PoolIndices.Rows == _input.BatchSize * L.Filters * pOutH * pOutW
                        && p.PoolIndices.UsedColumns == 1, p.PoolIndices);

            if (ok) sb.AppendLine($"    conv[{i}] OK");

            chkC = L.Filters;
            chkH = pOutH;
            chkW = pOutW;
        }

        int denseIn = flatSize;
        for (int i = 0; i < _denseHyperParameters.Count; i++)
        {
            var p = _denseHyperParameters[i];
            int denseOut = i < _cnnConfig.DenseArchitecture.Length
                ? _cnnConfig.DenseArchitecture[i]
                : p.Weights.Rows;

            bool ok = true;
            ok &= Check(sb, $"dense[{i}].Weights expects ({denseOut},{denseIn})",
                        p.Weights.Rows == denseOut && p.Weights.UsedColumns == denseIn, p.Weights);
            ok &= Check(sb, $"dense[{i}].Biases  expects (1,{denseOut})",
                        p.Biases.Rows == 1 && p.Biases.UsedColumns == denseOut, p.Biases);
            ok &= Check(sb, $"dense[{i}].PreAct  expects ({_input.BatchSize},{denseOut})",
                        p.PreAct.Rows >= _input.BatchSize && p.PreAct.UsedColumns == denseOut, p.PreAct);
            ok &= Check(sb, $"dense[{i}].PostAct expects ({_input.BatchSize},{denseOut})",
                        p.PostAct.Rows >= _input.BatchSize && p.PostAct.UsedColumns == denseOut, p.PostAct);

            if (ok) sb.AppendLine($"    dense[{i}] OK");
            denseIn = denseOut;
        }

        sb.AppendLine(hr);
        Console.WriteLine(sb.ToString());
    }

    private static string Fmt(CnnMatrix? m)
        => m is null
            ? "<null>"
            : $"Cnn({m.Batch}x{m.Channels}x{m.Height}x{m.Width})  elements={m.UnsafeSize}";

    private static string Fmt(NeuralMatrix? m)
        => m is null
            ? "<null>"
            : $"Nrl({m.Rows}x{m.UsedColumns})  stride={m.ColumnsStride}  elements={m.UnsafeSize}";

    private static bool Check(System.Text.StringBuilder sb, string label, bool ok, CnnMatrix m)
    {
        sb.AppendLine($"  {(ok ? "✔" : "✘")} {label}   actual={Fmt(m)}");
        return ok;
    }

    private static bool Check(System.Text.StringBuilder sb, string label, bool ok, NeuralMatrix m)
    {
        sb.AppendLine($"  {(ok ? "✔" : "✘")} {label}   actual={Fmt(m)}");
        return ok;
    }
}
