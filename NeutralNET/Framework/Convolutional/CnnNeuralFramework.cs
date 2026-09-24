using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NeutralNET.Activation;
using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Framework.Convolutional;
using NeutralNET.GPU;
using NeutralNET.Matrices;
using static NeutralNET.Activation.ActivationSelector;

namespace NeutralNET.Framework.Neural.CNN;

/// <summary>
/// Zero-GC CNN framework with full object and buffer pooling, pluggable optimizers,
/// and low-latency P/Invoke CUDA/cuBLAS GPU matrix acceleration.
/// </summary>
public sealed unsafe class CnnNeuralFramework
{
    private const bool EnableGpu = true;
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

    private readonly List<DenseHyperParameters> _denseHyperParameters = [];
    private readonly List<ConvHyperParameters> _convHyperParameters = [];

    private NeuralMatrix? _flattenedInput;
    private CnnMatrix? _lastPooledOutput;

    private readonly Random _rng;
    private readonly int _maxBatch;

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

        int flattenedSize = ComputeFlattenedSize(cnnConfig);
        int[] denseArch = [flattenedSize, .. cnnConfig.DenseArchitecture];
        int denseCount = denseArch.Length - 1;

        _denseHyperParameters = [with(denseCount)];
        _denseActivations = [with(denseCount)];
        _denseDerivatives = [with(denseCount)];
        _denseOptimizers = [with(denseCount)];

        SetupDenseArchitecture(denseArch, cnnConfig);
    }

    private void SetupCnnConvParameters(CnnArchitectureConfig cnnConfig)
    {
        var prevInput = _input;

        for (int i = 0; i < cnnConfig.ConvLayers.Count; i++)
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
            var input = GetInput(postAct, layer.PoolSize);

            int nextH = layer.UseMaxPool ? convOutSz.Height / layer.PoolSize : convOutSz.Height;
            int nextW = layer.UseMaxPool ? convOutSz.Width / layer.PoolSize : convOutSz.Width;
            var nextLayer = new CnnSize(prevInput.BatchSize, weights.Batch, nextH, nextW);

            var colInput = GetColInput(prevInput, layer.KernelHeight, layer.KernelWidth, layer.Stride, layer.Padding);
            var poolIndices = GetPoolIndices(postAct, layer.PoolSize);

            _convHyperParameters.Add(new(input, colInput, weights, flattenedWeights, biases, preAct, postAct, poolIndices));

            input.DisplayName = $"Conv_Input[{i}]";
            colInput.DisplayName = $"Conv_ColInput[{i}]";
            weights.DisplayName = $"Conv_Weights[{i}]";
            flattenedWeights.DisplayName = $"Conv_FlattenedWeights[{i}]";
            biases.DisplayName = $"Conv_Biases[{i}]";
            preAct.DisplayName = $"Conv_PreAct[{i}]";
            postAct.DisplayName = $"Conv_PostAct[{i}]";
            poolIndices.DisplayName = $"Conv_PoolIndices[{i}]";

            _convActivationTypes.Add(layer.Activation);

            var opt = CnnOptimizerFactory.Create(_cnnConfig.OptimizerConfig);
            _convOptimizers.Add(opt);

            prevInput = nextLayer;

            NeuralMatrix GetPoolIndices(CnnMatrix postAct, int poolSize)
            {
                int batch = postAct.Batch;
                int channels = postAct.Channels;
                int inH = postAct.Height;
                int inW = postAct.Width;

                int outH = inH / poolSize;
                int outW = inW / poolSize;

                return RentNeural(batch * channels * outH * outW, 1);
            }

            NeuralMatrix GetColInput(CnnSize input, int kernelH, int kernelW, int stride, int padding)
            {
                int paddedH = input.Height + 2 * padding;
                int paddedW = input.Width + 2 * padding;

                int outH = (paddedH - kernelH) / stride + 1;
                int outW = (paddedW - kernelW) / stride + 1;
                int patchSize = input.Channels * kernelH * kernelW;
                int totalPatches = input.BatchSize * outH * outW;

                return RentNeural(totalPatches, patchSize);
            }

            CnnMatrix GetInput(CnnMatrix postAct, int poolSize)
            {
                int batch = postAct.Batch;
                int channels = postAct.Channels;
                int inH = postAct.Height;
                int inW = postAct.Width;

                int outH = inH / poolSize;
                int outW = inW / poolSize;

                return RentCnn(batch, channels, outH, outW);
            }
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

            var opt = CnnOptimizerFactory.Create(cnnConfig.OptimizerConfig);
            _denseOptimizers.Add(opt);
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

            var pAct = convPreAct.Pointer;
            var totalElements = convPreAct.Batch * convPreAct.Channels * convPreAct.Height * convPreAct.Width;
            ApplyActivationVectorized(pAct, totalElements, layer.Activation);

            CnnMatrix next;
            if (layer.UseMaxPool)
            {
                next = MaxPoolForwardInPlace(convPreAct, layer.PoolSize);
            }
            else
            {
                next = convPreAct;
            }

            current = next;
            output[i] = current;
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
        ClearIntermediates();

        foreach (var b in _convHyperParameters) b.Dispose();
        foreach (var w in _denseHyperParameters) w.Dispose();
        foreach (var opt in _convOptimizers) opt.Dispose();
        foreach (var opt in _denseOptimizers) opt.Dispose();
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
        var needsDispose = false;

        SetBatchLimitAll(input.Batch);

        for (int layerIdx = 0; layerIdx < _cnnConfig.ConvLayers.Count; layerIdx++)
        {
            var layer = _cnnConfig.ConvLayers[layerIdx];
            ConvForward(current, layerIdx);
            var convOut = _convHyperParameters[layerIdx].PreAct;

            if (needsDispose)
            {
                current.Dispose();
            }

            var pAct = convOut.Pointer;
            var totalElements = convOut.Batch * convOut.Channels * convOut.Height * convOut.Width;

            ApplyActivationVectorized(pAct, totalElements, layer.Activation);

            if (layer.UseMaxPool)
            {
                var pooled = MaxPoolForwardInPlace(convOut, layer.PoolSize);
                current = pooled;
            }
            else
            {
                current = convOut;
            }

            needsDispose = true;
        }

        var flat = Flatten(current);

        if (needsDispose)
        {
            current.Dispose();
        }

        NeuralMatrix denseOut = DenseForward(flat, storeIntermediates: false);
        flat.Dispose();

        return denseOut;
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

    public float Train(CnnMatrix input, NeuralMatrix target, float learningRate)
    {
        ClearIntermediates();

        input.DisplayName = "MainInput";
        target.DisplayName = "MainExpected";

        CnnMatrix current = input;

        SetBatchLimitAll(current.Batch);

        var probabilities = ForwardPoolingPass(ref current);
        var loss = ComputeCrossEntropyLoss(probabilities, target);
        var grad = GetVectorizedLossGradients(target, probabilities);
        var denseGrad = DenseBackWardClipped(learningRate, grad);
        var currentGrad = BulkMemoryCopy(denseGrad);

        currentGrad = GetConvolutionBackwardPass(currentGrad);
        currentGrad?.Dispose();
        ClearIntermediates();

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
            elem.Input.Batch = limit;
        }

        foreach (var (_, _, preAct, postAct) in _denseHyperParameters)
        {
            preAct.SetRowSize(limit);
            postAct.SetRowSize(limit);
        }
    }

    private CnnMatrix GetConvolutionBackwardPass(CnnMatrix currentGrad)
    {
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

        return currentGrad;
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
        CnnMatrix inputTensor = cnvParams.Input;
        NeuralMatrix indices = cnvParams.PoolIndices;

        using var convGrad = BackPropagateThroughPool(currentGrad, layer, postAct, indices);
        using var preGrad = ComputePreGradient(layer, preAct, postAct, convGrad);

        using var preGradMatrix = ConvertPregradToMatrix(preGrad);

        var patches = preGradMatrix.Rows;
        var filters = preGrad.Channels;
        var inDim = colInput.UsedColumns;

        using var dW = ComputeWeightGradient(colInput, preGradMatrix, patches, filters, inDim);
        using var dB = ComputeBiasGradient(preGradMatrix, patches, filters);

        _convOptimizers[layerIdx].UpdateConvWeights(
            cnvParams.Weights,
            cnvParams.Biases,
            dW,
            dB
        );

        var flattenedWeights = cnvParams.FlattenedWeights;
        using var gradPatchMat = ComputeGradientWithRespectToInput(flattenedWeights, preGradMatrix, patches, filters, inDim);

        var inputGrad = RentCnn(inputTensor.Batch, inputTensor.Channels, inputTensor.Height, inputTensor.Width);
        inputGrad.Col2Im(gradPatchMat);

        return inputGrad;
    }

    private static NeuralMatrix ComputeGradientWithRespectToInput(
    NeuralMatrix weightMat,
    NeuralMatrix preGradMatrix,
    int patches,
    int filters,
    int inDim)
    {
        var gradPatchMat = RentNeural(patches, inDim);
        gradPatchMat.DisplayName = "GradPatchMat";

        if (EnableGpu)                       // ← un-inverted
        {
            GpuMatrixOps.RowMajorSgemmHostStaged(
                GpuMatrixOps.CublasOperation.NonTranspose,
                GpuMatrixOps.CublasOperation.NonTranspose,
                patches, inDim, filters,
                1.0f,
                preGradMatrix.Pointer, preGradMatrix.ColumnsStride,
                weightMat.Pointer, weightMat.ColumnsStride,
                0.0f,
                gradPatchMat.Pointer, gradPatchMat.ColumnsStride);
        }
        else
        {
            float* pGradPatch = gradPatchMat.Pointer;
            float* pPreGradMat = preGradMatrix.Pointer;
            float* pWeightMat = weightMat.Pointer;

            int gradPatchStride = gradPatchMat.ColumnsStride;
            int weightMatStride = weightMat.ColumnsStride;
            int preGradMatStride = preGradMatrix.ColumnsStride;

            // gradPatch[p, i] = Σ_f preGrad[p, f] * weight[f, i]
            // Vectorize over `i` (contiguous in both weight[f,:] and gradPatch[p,:])
            // and broadcast the scalar preGrad[p, f] into every lane.
            for (int patch = 0; patch < patches; patch++)
            {
                float* rowPreGradMat = pPreGradMat + patch * preGradMatStride;
                float* rowGradPatch = pGradPatch + patch * gradPatchStride;

                // Each row accumulates into gradPatch[p, 0..inDim-1] from zero.
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

        return gradPatchMat;
    }

    private static NeuralMatrix ComputeWeightGradient(
    NeuralMatrix colInput,
    NeuralMatrix preGradMatrix,
    int patches,
    int filters,
    int inDim)
    {
        NeuralMatrix dW;

        if (EnableGpu)
        {
            dW = RentNeural(filters, inDim);
            // dW = preGradMatrixᵀ · colInput  →  (filters, inDim)
            GpuMatrixOps.RowMajorSgemmHostStaged(
                GpuMatrixOps.CublasOperation.Transpose,
                GpuMatrixOps.CublasOperation.NonTranspose,
                filters, inDim, patches,
                1.0f,
                preGradMatrix.Pointer, preGradMatrix.ColumnsStride,
                colInput.Pointer, colInput.ColumnsStride,
                0.0f,
                dW.Pointer, dW.ColumnsStride);
        }
        else
        {
            dW = RentNeural(inDim, filters);

            float* pdW = dW.Pointer;
            int dWStride = dW.ColumnsStride;

            float* pColIn = colInput.Pointer;
            int colInStride = colInput.ColumnsStride;
            float* pPreGradMat = preGradMatrix.Pointer;
            int preGradMatStride = preGradMatrix.ColumnsStride;

            // dW[f, i] = Σ_p preGrad[p, f] * colInput[p, i]
            // Outer loop over patches keeps rowColIn hot across all filters.
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

        return dW;
    }

    private static NeuralMatrix ComputeBiasGradient(NeuralMatrix preGradMatrix, int patches, int filters)
    {
        var dB = RentNeural(1, filters);
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

        return dB;
    }

    private static NeuralMatrix ConvertPregradToMatrix(CnnMatrix preGrad)
    {
        int outH = preGrad.Height;
        int outW = preGrad.Width;
        int patches = preGrad.Batch * outH * outW;
        int filters = preGrad.Channels;

        var preGradMatrix = RentNeural(patches, filters);
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

        return preGradMatrix;
    }

    private CnnMatrix ComputePreGradient(CnnLayerConfig layer, CnnMatrix preAct, CnnMatrix postAct, CnnMatrix convGrad)
    {
        var preGrad = RentCnn(preAct.Batch, preAct.Channels, preAct.Height, preAct.Width);
        ApplyDerivativeDirect(convGrad, postAct, preGrad, layer.Activation);
        return preGrad;
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

    private CnnMatrix BackPropagateThroughPool(CnnMatrix currentGrad, CnnLayerConfig layer, CnnMatrix postAct, NeuralMatrix indices)
    {
        if (layer.UseMaxPool)
        {
            var convGrad = MaxPoolBackward(currentGrad, postAct, indices, layer.PoolSize);
            currentGrad.Dispose();

            return convGrad;
        }

        return currentGrad;
    }

    private CnnMatrix BulkMemoryCopy(NeuralMatrix denseGrad)
    {
        var lastPooled = _lastPooledOutput!;
        var pooledGrad = RentCnn(lastPooled.Batch, lastPooled.Channels, lastPooled.Height, lastPooled.Width);

        float* pDenseGrad = denseGrad.Pointer;
        float* pPooledGrad = pooledGrad.Pointer;

        int denseStride = denseGrad.ColumnsStride;
        int spatialDim = lastPooled.Channels * lastPooled.Height * lastPooled.Width;

        for (int b = 0; b < lastPooled.Batch; b++)
        {
            float* srcRow = pDenseGrad + b * denseStride;
            float* dstRow = pPooledGrad + b * spatialDim;
            nuint bytesToCopy = (nuint)spatialDim * sizeof(float);

            NativeMemory.Copy(srcRow, dstRow, bytesToCopy);
        }

        denseGrad.Dispose();
        return pooledGrad;
    }

    private NeuralMatrix DenseBackWardClipped(float learningRate, NeuralMatrix grad)
    {
        var denseGrad = DenseBackward(grad, learningRate, skipLastDerivative: true);

        if (_lastPooledOutput == null)
        {
            throw new InvalidOperationException("_lastPooledOutput is null.");
        }

        return denseGrad;
    }

    private static NeuralMatrix GetVectorizedLossGradients(NeuralMatrix target, NeuralMatrix probabilities)
    {
        int rows = probabilities.Rows;
        int cols = probabilities.UsedColumns;
        var grad = RentNeural(rows, cols);

        float* pProb = probabilities.Pointer;
        float* pTarg = target.Pointer;
        float* pGrad = grad.Pointer;

        int probStride = probabilities.ColumnsStride;
        int targStride = target.ColumnsStride;
        int gradStride = grad.ColumnsStride;

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

        return grad;
    }

    private NeuralMatrix ForwardPoolingPass(ref CnnMatrix current)
    {
        for (var layerIdx = 0; layerIdx < _cnnConfig.ConvLayers.Count; layerIdx++)
        {
            var layer = _cnnConfig.ConvLayers[layerIdx];
            var input = _convHyperParameters[layerIdx].Input;
            input.CopyFrom(current);

            ConvForward(current, layerIdx);

            var preAct = _convHyperParameters[layerIdx].PreAct;
            var postAct = _convHyperParameters[layerIdx].PostAct;

            postAct.CopyFrom(preAct);
            ApplyActivation(postAct, layer.Activation);

            var poolIndices = _convHyperParameters[layerIdx].PoolIndices;
            MaxPoolForward(postAct, poolIndices, input, layer.PoolSize);
            current = input;
        }

        _lastPooledOutput = current;

        var flat = Flatten(current);
        _flattenedInput = flat;

        using (DenseForward(flat, storeIntermediates: true)) { }

        return _denseHyperParameters[^1].PostAct;  //_densePostAct[^1];
    }

    private void ConvForward(CnnMatrix current, int layerIdx)
    {
        var colInput = _convHyperParameters[layerIdx].ColInput;
        var layer = _cnnConfig.ConvLayers[layerIdx];
        var weights = _convHyperParameters[layerIdx].Weights;
        var flattenedWeights = _convHyperParameters[layerIdx].FlattenedWeights;
        var preAct = _convHyperParameters[layerIdx].PreAct;
        var biases = _convHyperParameters[layerIdx].Biases;

        current.Im2Col(colInput, layer.KernelHeight, layer.KernelWidth, layer.Stride, layer.Padding);
        UpdateFlattenConvWeights(weights, flattenedWeights);
        var result = ComputeConvolution(colInput, flattenedWeights);
        AddBias(result, biases);

        FillToCnnMatrix(result, preAct);

        result.Dispose();
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

    private NeuralMatrix ComputeConvolution(NeuralMatrix colInput, NeuralMatrix weightMat)
    {
        int patches = colInput.Rows;
        int filters = weightMat.Rows;
        int innerDim = colInput.UsedColumns;

        var result = RentNeural(patches, filters);

        if (EnableGpu)
        {
            GpuMatrixOps.RowMajorSgemmHostStaged(
                GpuMatrixOps.CublasOperation.NonTranspose,
                GpuMatrixOps.CublasOperation.Transpose,
                patches, filters, innerDim,
                1.0f,
                colInput.Pointer, colInput.ColumnsStride,
                weightMat.Pointer, weightMat.ColumnsStride,
                0.0f,
                result.Pointer, result.ColumnsStride);
        }
        else
        {
            float* colPtr = colInput.Pointer;
            float* weightPtr = weightMat.Pointer;
            float* resPtr = result.Pointer;
            int colStride = colInput.ColumnsStride;
            int weightStride = weightMat.ColumnsStride;
            int resStride = result.ColumnsStride;

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

        return result;
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

    private CnnMatrix MaxPoolForwardInPlace(CnnMatrix input, int poolSize)
    {
        int batch = input.Batch;
        int channels = input.Channels;
        int inH = input.Height;
        int inW = input.Width;

        int outH = inH / poolSize;
        int outW = inW / poolSize;

        var pooled = RentCnn(batch, channels, outH, outW);

        float* pIn = input.Pointer;
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

        return pooled;
    }

    private void MaxPoolForward(CnnMatrix postAct, NeuralMatrix indices, CnnMatrix pooled, int poolSize)
    {
        int batch = postAct.Batch;
        int channels = postAct.Channels;
        int inH = postAct.Height;
        int inW = postAct.Width;

        int outH = inH / poolSize;
        int outW = inW / poolSize;

        // FIX: runtime shape validation. In Release, AssertSameSize is a no-op,
        // so add an explicit check here to catch silent truncation.
        if (pooled.Batch != batch ||
            pooled.Channels != channels ||
            pooled.Height != outH ||
            pooled.Width != outW)
        {
            throw new InvalidOperationException(
                $"MaxPoolForward: destination shape ({pooled.Batch},{pooled.Channels},{pooled.Height},{pooled.Width}) " +
                $"does not match expected ({batch},{channels},{outH},{outW}).");
        }

        float* pIn = postAct.Pointer;
        float* pOut = pooled.Pointer;
        float* pIdx = indices.Pointer;

        int spatialInSize = inH * inW;
        int spatialOutSize = outH * outW;
        int numSlices = batch * channels;

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
                        float* rowPtr = sliceIn + (y * inW);

                        for (int dx = 0; dx < poolSize; dx++)
                        {
                            int x = xStart + dx;
                            float val = rowPtr[x];

                            if (val > maxVal)
                            {
                                maxVal = val;
                                maxIdx = y * inW + x;
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

    private CnnMatrix MaxPoolBackward(CnnMatrix gradOutput, CnnMatrix postAct, NeuralMatrix indices, int poolSize)
    {
        int batch = postAct.Batch;
        int channels = postAct.Channels;
        int inH = postAct.Height;
        int inW = postAct.Width;
        int outH = gradOutput.Height;
        int outW = gradOutput.Width;

        var gradInput = RentCnn(batch, channels, inH, inW);

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

    private NeuralMatrix DenseForward(NeuralMatrix input, bool storeIntermediates)
    {
        var current = input;

        for (int i = 0; i < _denseHyperParameters.Count; i++)
        {
            var weights = _denseHyperParameters[i].Weights;
            var biases = _denseHyperParameters[i].Biases;

            int batchSize = current.Rows;
            int inFeatures = current.UsedColumns;
            int outFeatures = weights.Rows;

            var result = RentNeural(batchSize, outFeatures);

            if (EnableGpu)
            {
                GpuMatrixOps.RowMajorSgemmHostStaged(
                    GpuMatrixOps.CublasOperation.NonTranspose,
                    GpuMatrixOps.CublasOperation.Transpose,
                    batchSize, outFeatures, inFeatures,
                    1.0f,
                    current.Pointer, current.ColumnsStride,
                    weights.Pointer, weights.ColumnsStride,
                    0.0f,
                    result.Pointer, result.ColumnsStride);

                for (int b = 0; b < batchSize; b++)
                {
                    float* row = result.Pointer + b * result.ColumnsStride;
                    for (int f = 0; f < outFeatures; f++)
                    {
                        row[f] += biases.Pointer[f];
                    }
                }
            }
            else
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

            if (storeIntermediates)
            {
                _denseHyperParameters[i].PreAct.CopyFrom(result);
            }

            _denseActivations[i](result);

            if (storeIntermediates)
            {
                _denseHyperParameters[i].PostAct.CopyFrom(result);
            }

            if (!ReferenceEquals(current, input))
            {
                current.Dispose();
            }

            current = result;
        }

        return current;
    }

    private NeuralMatrix DenseBackward(NeuralMatrix gradOutput, float learningRate, bool skipLastDerivative = false)
    {
        for (int i = _denseHyperParameters.Count - 1; i >= 0; i--)
        {
            var preAct = _denseHyperParameters[i].PreAct;
            var inputToLayer = (i == 0) ? _flattenedInput : _denseHyperParameters[i - 1].PostAct;

            if (inputToLayer == null)
            {
                throw new InvalidOperationException($"inputToLayer is null for layer {i}.");
            }

            int batch = gradOutput.Rows;
            int outDim = gradOutput.UsedColumns;
            int inDim = inputToLayer.UsedColumns;

            var gradPre = RentNeural(batch, outDim);

            float* pGradOut = gradOutput.Pointer;
            float* pGradPre = gradPre.Pointer;
            float* pPreAct = preAct.Pointer;
            int strideGradOut = gradOutput.ColumnsStride;
            int strideGradPre = gradPre.ColumnsStride;
            int stridePreAct = preAct.ColumnsStride;

            var derivativeFn = _denseDerivatives[i];
            bool skipDeriv = skipLastDerivative && (i == _denseHyperParameters.Count - 1);

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

            var dW = RentNeural(inDim, outDim);

            if (EnableGpu)
            {
                GpuMatrixOps.RowMajorSgemmHostStaged(
                    GpuMatrixOps.CublasOperation.Transpose,
                    GpuMatrixOps.CublasOperation.NonTranspose,
                    inDim, outDim, batch,
                    1.0f,
                    inputToLayer.Pointer, inputToLayer.ColumnsStride,
                    gradPre.Pointer, gradPre.ColumnsStride,
                    0.0f,
                    dW.Pointer, dW.ColumnsStride);
            }
            else
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

            var dB = RentNeural(1, outDim);
            dB.Clear();
            float* pDB = dB.Pointer;

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

            _denseOptimizers[i].UpdateDenseWeights(_denseHyperParameters[i].Weights, _denseHyperParameters[i].Biases, dW, dB);

            var weights = _denseHyperParameters[i].Weights;
            int weightOutDim = weights.Rows;
            int weightInDim = weights.UsedColumns;
            var gradInput = RentNeural(batch, weightInDim);

            if (EnableGpu)
            {
                GpuMatrixOps.RowMajorSgemmHostStaged(
                    GpuMatrixOps.CublasOperation.NonTranspose,
                    GpuMatrixOps.CublasOperation.NonTranspose,
                    batch, weightInDim, weightOutDim,
                    1.0f,
                    gradPre.Pointer, gradPre.ColumnsStride,
                    weights.Pointer, weights.ColumnsStride,
                    0.0f,
                    gradInput.Pointer, gradInput.ColumnsStride);
            }
            else   // CPU path
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
                                // LoadUnsafe / StoreUnsafe: unaligned-safe.
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
                                // Avx.LoadVector256 / Avx.Store are the unaligned variants.
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

            gradOutput.Dispose();
            dW.Dispose();
            dB.Dispose();
            gradPre.Dispose();

            gradOutput = gradInput;
        }

        return gradOutput;
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

    private static NeuralMatrix Flatten(CnnMatrix input)
    {
        int featureDim = input.Channels * input.Height * input.Width;
        var flat = RentNeural(input.Batch, featureDim);

        float* pSrc = input.Pointer;
        float* pDst = flat.Pointer;

        int srcStride = featureDim;
        int dstStride = flat.ColumnsStride;

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

        return flat;
    }

    private float ComputeCrossEntropyLoss(NeuralMatrix predictions, NeuralMatrix targets)
    {
        var rows = predictions.Rows;
        var cols = predictions.UsedColumns;
        var eps = 1e-7f;
        var totalLoss = 0f;

        var pPred = predictions.Pointer;
        var pTarg = targets.Pointer;

        var predStride = predictions.ColumnsStride;
        var targStride = targets.ColumnsStride;

        for (var r = 0; r < rows; r++)
        {
            var predRow = pPred + r * predStride;
            var targRow = pTarg + r * targStride;

            var rowLoss = 0f;
            var rowHasValidTarget = false;

            for (int c = 0; c < cols; c++)
            {
                if (float.IsNaN(predRow[c]) || float.IsInfinity(predRow[c]) || predRow[c] < 0f || predRow[c] > 1f)
                {
                    predRow[c] = 1.0f / cols;
                }

                var pVal = Math.Clamp(predRow[c], eps, 1.0f - eps);
                var tVal = targRow[c];

                if (tVal > 0f)
                {
                    rowHasValidTarget = true;
                    var logVal = MathF.Log(pVal);

                    if (!float.IsNaN(logVal) && !float.IsInfinity(logVal))
                    {
                        rowLoss -= tVal * logVal;
                    }
                }
            }

            if (!rowHasValidTarget)
            {
                rowLoss = MathF.Log(cols);
            }

            if (float.IsNaN(rowLoss) || float.IsInfinity(rowLoss) || rowLoss > 100f)
            {
                rowLoss = 10.0f;
            }

            totalLoss += rowLoss;
        }

        var avgLoss = totalLoss / rows;

        return float.IsNaN(avgLoss) || float.IsInfinity(avgLoss) || avgLoss > 100f ? 10.0f : avgLoss;
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
        sb.AppendLine($"  _lastPooledOut   : {Fmt(_lastPooledOutput)}");

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

    private void ClearIntermediates()
    {
        if (_flattenedInput?.Pointer != null)
        {
            _flattenedInput.Dispose();
            _flattenedInput = default;
        }

        _lastPooledOutput = null;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static NeuralMatrix RentNeural(int rows, int cols, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
        => NeuralMatrix.GetOrCreate(rows, cols, fp, ln);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static CnnMatrix RentCnn(int batch, int channels, int h, int w, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
        => CnnMatrix.GetOrCreate(batch, channels, h, w, fp, ln);

    private static CnnMatrix RentCnn(CnnSize cnnSize, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
        => CnnMatrix.GetOrCreate(cnnSize.BatchSize, cnnSize.Channels, cnnSize.Height, cnnSize.Width, fp, ln);
}
