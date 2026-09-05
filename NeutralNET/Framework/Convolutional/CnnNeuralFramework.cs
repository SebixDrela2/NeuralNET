using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NeutralNET.Activation;
using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;
using NeutralNET.GPU;
using static NeutralNET.Activation.ActivationSelector;

namespace NeutralNET.Framework.Neural.CNN;

/// <summary>
/// Zero‑GC CNN framework with full object and buffer pooling, pluggable optimizers,
/// and low-latency P/Invoke CUDA/cuBLAS GPU matrix acceleration.
/// </summary>
public sealed unsafe class CnnNeuralFramework<TArch> : IDisposable
    where TArch : IArchitecture<TArch>
{
    private const float GradientClipNorm = 0.5f;
    private const int Avx256Size = 8;
    private const int Avx512Size = 16;

    private readonly NeuralNetworkConfig _baseConfig;
    private readonly CnnArchitectureConfig _cnnConfig;
    private readonly ActivationSelector _activationSelector = new();
    private readonly int _inputHeight;
    private readonly int _inputWidth;
    private readonly int _inputChannels;

    private readonly List<CnnMatrix> _convWeights = [];
    private readonly List<CnnMatrix> _convBiases = [];
    private readonly List<ActivationType> _convActivationTypes = [];

    private readonly List<NeuralMatrix> _denseWeights = [];
    private readonly List<NeuralMatrix> _denseBiases = [];
    private readonly List<ActivationFunction> _denseActivations = [];
    private readonly List<DerivativeFunction> _denseDerivatives = [];
    private readonly List<ICnnOptimizer> _convOptimizers = [];
    private readonly List<ICnnOptimizer> _denseOptimizers = [];

    private readonly List<CnnMatrix> _convInputs = [];
    private readonly List<CnnMatrix> _convPreAct = [];
    private readonly List<CnnMatrix> _convPostAct = [];
    private readonly List<NeuralMatrix> _colInputs = [];
    private readonly List<NeuralMatrix> _weightMatrices = [];
    private readonly List<NeuralMatrix> _poolIndices = [];
    private readonly List<NeuralMatrix> _densePreAct = [];
    private readonly List<NeuralMatrix> _densePostAct = [];

    private NeuralMatrix? _flattenedInput;
    private CnnMatrix? _lastPooledOutput;

    private readonly Random _rng;

    public CnnNeuralFramework(NeuralNetworkConfig baseConfig, CnnArchitectureConfig cnnConfig,
        int inputHeight, int inputWidth, int inputChannels)
    {
        _baseConfig = baseConfig;
        _cnnConfig = cnnConfig;
        _rng = new Random();
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _inputChannels = inputChannels;

        SetupCnnWeightsBiases(cnnConfig);
        SetupDenseArchitecture(cnnConfig);
    }

    private void SetupCnnWeightsBiases(CnnArchitectureConfig cnnConfig)
    {
        int inChannels = _inputChannels;

        foreach (var layer in cnnConfig.ConvLayers)
        {
            var fanIn = inChannels * layer.KernelHeight * layer.KernelWidth;
            var stddev = MathF.Sqrt(2.0f / fanIn);
            var weights = RentCnn(layer.Filters, inChannels, layer.KernelHeight, layer.KernelWidth);

            for (int f = 0; f < layer.Filters; f++)
            {
                for (int c = 0; c < inChannels; c++)
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

            _convWeights.Add(weights);
            _convBiases.Add(biases);
            _convActivationTypes.Add(layer.Activation);

            var opt = CnnOptimizerFactory.Create(_cnnConfig.OptimizerConfig);
            _convOptimizers.Add(opt);

            inChannels = layer.Filters;
        }
    }

    private void SetupDenseArchitecture(CnnArchitectureConfig cnnConfig)
    {
        var flattenedSize = ComputeFlattenedSize(_cnnConfig);
        int[] denseArch = [flattenedSize, .. _cnnConfig.DenseArchitecture];

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

            _denseWeights.Add(weights);
            _denseBiases.Add(biases);

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

    public void Dispose()
    {
        ClearIntermediates();

        foreach (var w in _convWeights) w.Dispose();
        foreach (var b in _convBiases) b.Dispose();
        foreach (var w in _denseWeights) w.Dispose();
        foreach (var b in _denseBiases) b.Dispose();
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
        var h = _inputHeight;
        var w = _inputWidth;
        var channels = _inputChannels;

        foreach (var layer in config.ConvLayers)
        {
            int paddedH = h + 2 * layer.Padding;
            int paddedW = w + 2 * layer.Padding;

            h = (paddedH - layer.KernelHeight) / layer.Stride + 1;
            w = (paddedW - layer.KernelWidth) / layer.Stride + 1;

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

        for (int layerIdx = 0; layerIdx < _cnnConfig.ConvLayers.Count; layerIdx++)
        {
            var layer = _cnnConfig.ConvLayers[layerIdx];
            var (convOut, colInput, weightMat) = ConvForward(current, layerIdx);

            colInput.Dispose();
            weightMat.Dispose();

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
                convOut.Dispose();
                current = pooled;
            }
            else
            {
                current = convOut;
            }

            needsDispose = true;
        }

        var flat = CnnNeuralFramework<TArch>.Flatten(current);

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
                if (Avx512F.IsSupported)
                {
                    var vZero = Vector512<float>.Zero;
                    int vecLimit = count - (count % Avx512Size);

                    for (; i < vecLimit; i += Avx512Size)
                    {
                        var vSrc = Vector512.Load(ptr + i);
                        Vector512.Max(vSrc, vZero).Store(ptr + i);
                    }
                }
                else if (Avx2.IsSupported)
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

                if (Avx512F.IsSupported)
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
                else if (Avx2.IsSupported)
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
        CnnMatrix current = input;

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

    private CnnMatrix GetConvolutionBackwardPass(CnnMatrix currentGrad)
    {
        for (int layerIdx = _cnnConfig.ConvLayers.Count - 1; layerIdx >= 0; layerIdx--)
        {
            var layer = _cnnConfig.ConvLayers[layerIdx];
            var preAct = _convPreAct[layerIdx];
            var postAct = _convPostAct[layerIdx];
            var colInput = _colInputs[layerIdx];
            var weightMat = _weightMatrices[layerIdx];
            var inputTensor = _convInputs[layerIdx];
            var indices = _poolIndices[layerIdx];

            currentGrad = ProcessSingleConvLayerBackward(
                currentGrad,
                layerIdx, layer,
                preAct, postAct, colInput, weightMat, inputTensor, indices);
        }

        return currentGrad;
    }

    private CnnMatrix ProcessSingleConvLayerBackward(
        CnnMatrix currentGrad,
        int layerIdx,
        CnnLayerConfig layer,
        CnnMatrix preAct,
        CnnMatrix postAct,
        NeuralMatrix colInput,
        NeuralMatrix weightMat,
        CnnMatrix inputTensor,
        NeuralMatrix indices)
    {
        var convGrad = BackPropagateThroughPool(currentGrad, layer, postAct, indices);
        ClipCnnGradient(convGrad, GradientClipNorm);

        var preGrad = ComputePreGradient(layer, preAct, postAct, convGrad);
        convGrad.Dispose();

        var preGradMatrix = ConvertPregradToMatrix(preGrad);

        var patches = preGradMatrix.Rows;
        var filters = preGrad.Channels;
        var inDim = colInput.UsedColumns;

        ClipGradients(preGradMatrix, GradientClipNorm);

        var dW = ComputeWeightGradient(colInput, preGradMatrix, patches, filters, inDim);
        ClipGradients(dW, GradientClipNorm);

        var dB = ComputeBiasGradient(preGradMatrix, patches, filters);
        ClipGradients(dB, GradientClipNorm);

        _convOptimizers[layerIdx].UpdateConvWeights(_convWeights[layerIdx], _convBiases[layerIdx], dW, dB);

        var gradPatchMat = ComputeGradientWithRespectToInput(weightMat, preGradMatrix, patches, filters, inDim);
        ClipGradients(gradPatchMat, GradientClipNorm);

        var inputGrad = RentCnn(inputTensor.Batch, inputTensor.Channels, inputTensor.Height, inputTensor.Width);
        inputGrad.Col2Im(gradPatchMat, layer.KernelHeight, layer.KernelWidth, layer.Stride, layer.Padding, 1.0f);
        ClipCnnGradient(inputGrad, GradientClipNorm);

        preGrad.Dispose();
        preGradMatrix.Dispose();
        dW.Dispose();
        dB.Dispose();
        gradPatchMat.Dispose();

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

        GpuMatrixOps.ComputeGradientWithRespectToInputGpu(
            weightMat.Pointer,
            preGradMatrix.Pointer,
            gradPatchMat.Pointer,
            patches, filters, inDim,
            weightMat.ColumnsStride,
            preGradMatrix.ColumnsStride,
            gradPatchMat.ColumnsStride);

        return gradPatchMat;
    }

    private static NeuralMatrix ComputeWeightGradient(
        NeuralMatrix colInput,
        NeuralMatrix preGradMatrix,
        int patches,
        int filters,
        int inDim)
    {
        var dW = RentNeural(filters, inDim);

        GpuMatrixOps.ComputeWeightGradientGpu(
            colInput.Pointer,
            preGradMatrix.Pointer,
            dW.Pointer,
            patches, filters, inDim,
            colInput.ColumnsStride,
            preGradMatrix.ColumnsStride,
            dW.ColumnsStride);

        return dW;
    }

    private static NeuralMatrix ComputeBiasGradient(NeuralMatrix preGradMatrix, int patches, int filters)
    {
        var dB = RentNeural(1, filters);
        var pdB = dB.Pointer;
        var pPreGradMat = preGradMatrix.Pointer;
        var preGradMatStride = preGradMatrix.ColumnsStride;
        new Span<float>(pdB, filters).Clear();

        for (int patch = 0; patch < patches; patch++)
        {
            float* rowPreGradMat = pPreGradMat + patch * preGradMatStride;
            int f = 0;

            if (Avx512F.IsSupported)
            {
                int vecLimit = filters - (filters % Avx512Size);
                for (; f < vecLimit; f += Avx512Size)
                {
                    var vDB = Vector512.Load(pdB + f);
                    var vGrad = Vector512.Load(rowPreGradMat + f);
                    (vDB + vGrad).Store(pdB + f);
                }
            }
            else if (Avx2.IsSupported)
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

        ClipCnnGradient(preGrad, GradientClipNorm);

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
            if (Avx512F.IsSupported)
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
            else if (Avx2.IsSupported)
            {
                Vector256<float> vZero = Vector256<float>.Zero;
                int limit = totalElements - (totalElements % Avx256Size);
                for (; i < limit; i += Avx256Size)
                {
                    Vector256<float> g = Avx.LoadVector256(pGrad + i);
                    Vector256<float> p = Avx.LoadVector256(pPost + i);
                    Vector256<float> mask = Avx.Compare(p, vZero, FloatComparisonMode.OrderedGreaterThanNonSignaling);
                    Avx.And(g, mask).Store(pDst + i);
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
        ClipGradients(grad, GradientClipNorm);
        var denseGrad = DenseBackward(grad, learningRate, skipLastDerivative: true);
        ClipGradients(denseGrad, GradientClipNorm);

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
            if (Avx512F.IsSupported)
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
            else if (Avx2.IsSupported)
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
        for (int layerIdx = 0; layerIdx < _cnnConfig.ConvLayers.Count; layerIdx++)
        {
            var layer = _cnnConfig.ConvLayers[layerIdx];
            _convInputs.Add(current);

            var (convPreAct, colInput, weightMat) = ConvForward(current, layerIdx);
            _convPreAct.Add(convPreAct);
            _colInputs.Add(colInput);
            _weightMatrices.Add(weightMat);

            var convPostAct = RentCnn(convPreAct.Batch, convPreAct.Channels, convPreAct.Height, convPreAct.Width);
            convPostAct.CopyFrom(convPreAct);
            ApplyActivation(convPostAct, layer.Activation);
            _convPostAct.Add(convPostAct);

            CnnMatrix pooled;
            if (layer.UseMaxPool)
            {
                pooled = MaxPoolForward(convPostAct, layer.PoolSize, out var indices);
                _poolIndices.Add(indices);
            }
            else
            {
                pooled = convPostAct;
                _poolIndices.Add(RentNeural(1, 1));
            }

            current = pooled;
        }

        _convInputs.Add(current);
        _lastPooledOutput = current;

        var flat = CnnNeuralFramework<TArch>.Flatten(current);
        _flattenedInput = flat;

        using (DenseForward(flat, storeIntermediates: true)) { }

        return _densePostAct[^1];
    }

    private void ClipGradients(NeuralMatrix gradient, float maxNorm)
    {
        int totalElements = gradient.UnsafeSize;
        if (totalElements == 0) return;

        float* ptr = gradient.Pointer;
        if (ptr == null) return;

        bool hasError = false;
        for (int i = 0; i < totalElements; i++)
        {
            if (float.IsNaN(ptr[i]) || float.IsInfinity(ptr[i]))
            {
                hasError = true;
                break;
            }
        }

        if (hasError)
        {
            new Span<float>(ptr, totalElements).Clear();
            return;
        }

        float norm = 0f;
        for (int i = 0; i < totalElements; i++)
        {
            norm += ptr[i] * ptr[i];
        }
        norm = MathF.Sqrt(norm);

        if (norm > maxNorm && norm > 0f)
        {
            float scale = maxNorm / norm;
            for (int i = 0; i < totalElements; i++)
            {
                ptr[i] *= scale;
            }
        }
    }

    private unsafe void ClipCnnGradient(CnnMatrix gradient, float maxNorm)
    {
        int totalElements = gradient.UnsafeSize;
        if (totalElements == 0) return;

        float* ptr = gradient.Pointer;
        if (ptr == null) return;

        bool hasError = false;
        for (int i = 0; i < totalElements; i++)
        {
            if (float.IsNaN(ptr[i]) || float.IsInfinity(ptr[i]))
            {
                hasError = true;
                break;
            }
        }

        if (hasError)
        {
            new Span<float>(ptr, totalElements).Clear();
            return;
        }

        float norm = 0f;
        for (int i = 0; i < totalElements; i++)
        {
            norm += ptr[i] * ptr[i];
        }
        norm = MathF.Sqrt(norm);

        if (norm > maxNorm && norm > 0f)
        {
            float scale = maxNorm / norm;
            for (int i = 0; i < totalElements; i++)
            {
                ptr[i] *= scale;
            }
        }
    }

    private (CnnMatrix preAct, NeuralMatrix colInput, NeuralMatrix weightMat) ConvForward(CnnMatrix current, int layerIdx)
    {
        var layer = _cnnConfig.ConvLayers[layerIdx];
        var weights = _convWeights[layerIdx];
        var biases = _convBiases[layerIdx];

        var colInput = current.Im2Col(layer.KernelHeight, layer.KernelWidth, layer.Stride, layer.Padding);
        var weightMat = CreateWeightMatrix(weights);
        var result = ComputeConvolution(colInput, weightMat);
        AddBias(result, biases);

        var preAct = ReshapeToCnnMatrix(result, current.Batch, weights.Batch, current.Height, current.Width, layer);

        result.Dispose();

        return (preAct, colInput, weightMat);
    }

    private NeuralMatrix CreateWeightMatrix(CnnMatrix weights)
    {
        var innerDim = weights.Channels * weights.Height * weights.Width;
        var filters = weights.Batch;
        var weightMat = RentNeural(filters, innerDim);

        var pSrc = weights.Pointer;
        var pDst = weightMat.Pointer;
        var dstStride = weightMat.ColumnsStride;

        if (dstStride == innerDim)
        {
            long totalBytes = (long)filters * innerDim * sizeof(float);
            Buffer.MemoryCopy(pSrc, pDst, totalBytes, totalBytes);
            return weightMat;
        }

        long bytesPerRow = (long)innerDim * sizeof(float);

        for (int f = 0; f < filters; f++)
        {
            float* srcRow = pSrc + (f * innerDim);
            float* dstRow = pDst + (f * dstStride);
            Buffer.MemoryCopy(srcRow, dstRow, bytesPerRow, bytesPerRow);
        }

        return weightMat;
    }

    private NeuralMatrix ComputeConvolution(NeuralMatrix colInput, NeuralMatrix weightMat)
    {
        int patches = colInput.Rows;
        int filters = weightMat.Rows;
        int innerDim = colInput.UsedColumns;

        var result = RentNeural(patches, filters);

        GpuMatrixOps.ComputeConvolutionGpu(
            colInput.Pointer,
            weightMat.Pointer,
            result.Pointer,
            patches, filters, innerDim,
            colInput.ColumnsStride, weightMat.ColumnsStride, result.ColumnsStride);

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

            if (Avx512F.IsSupported)
            {
                int limit = filters - (filters % Avx512Size);
                for (; f < limit; f += Avx512Size)
                {
                    (Vector512.Load(row + f) + Vector512.Load(biasPtr + f)).Store(row + f);
                }
            }
            else if (Avx2.IsSupported)
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

    private CnnMatrix ReshapeToCnnMatrix(NeuralMatrix result, int batchSize, int filters, int height, int width, CnnLayerConfig layer)
    {
        int outH = (height + 2 * layer.Padding - layer.KernelHeight) / layer.Stride + 1;
        int outW = (width + 2 * layer.Padding - layer.KernelWidth) / layer.Stride + 1;
        int expectedRows = batchSize * outH * outW;

        if (result.Rows != expectedRows)
        {
            throw new InvalidOperationException($"result.Rows ({result.Rows}) != expectedRows ({expectedRows}).");
        }

        if (result.UsedColumns != filters)
        {
            throw new InvalidOperationException($"result.UsedColumns ({result.UsedColumns}) != filters ({filters})");
        }

        var preAct = RentCnn(batchSize, filters, outH, outW);

        float* pSrc = result.Pointer;
        float* pDst = preAct.Pointer;

        int srcStride = result.ColumnsStride;
        int spatialSize = outH * outW;
        int batchStrideDst = filters * spatialSize;

        for (int b = 0; b < batchSize; b++)
        {
            int batchOffsetSrc = b * spatialSize;
            float* pDstBatch = pDst + (b * batchStrideDst);

            for (int f = 0; f < filters; f++)
            {
                float* pDstChannel = pDstBatch + (f * spatialSize);
                for (int spatialIdx = 0; spatialIdx < spatialSize; spatialIdx++)
                {
                    pDstChannel[spatialIdx] = pSrc[(batchOffsetSrc + spatialIdx) * srcStride + f];
                }
            }
        }

        return preAct;
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

    private CnnMatrix MaxPoolForward(CnnMatrix input, int poolSize, out NeuralMatrix indices)
    {
        int batch = input.Batch;
        int channels = input.Channels;
        int inH = input.Height;
        int inW = input.Width;

        int outH = inH / poolSize;
        int outW = inW / poolSize;

        var pooled = RentCnn(batch, channels, outH, outW);
        var idxMat = RentNeural(batch * channels * outH * outW, 1);

        float* pIn = input.Pointer;
        float* pOut = pooled.Pointer;
        float* pIdx = idxMat.Pointer;

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
                    sliceIdx[outIdx] = (float)maxIdx;
                    outIdx++;
                }
            }
        }

        indices = idxMat;
        return pooled;
    }

    private CnnMatrix MaxPoolBackward(CnnMatrix gradOutput, CnnMatrix input, NeuralMatrix indices, int poolSize)
    {
        int batch = input.Batch;
        int channels = input.Channels;
        int inH = input.Height;
        int inW = input.Width;
        int outH = gradOutput.Height;
        int outW = gradOutput.Width;

        var gradInput = RentCnn(batch, channels, inH, inW);

        int totalInputElements = batch * channels * inH * inW;
        new Span<float>(gradInput.Pointer, totalInputElements).Clear();

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

        for (int i = 0; i < _denseWeights.Count; i++)
        {
            var weights = _denseWeights[i];
            var biases = _denseBiases[i];

            int batchSize = current.Rows;
            int inFeatures = current.UsedColumns;
            int outFeatures = weights.Rows;

            var result = RentNeural(batchSize, outFeatures);

            GpuMatrixOps.ComputeDenseForwardGpu(
                current.Pointer,
                weights.Pointer,
                biases.Pointer,
                result.Pointer,
                batchSize, inFeatures, outFeatures,
                current.ColumnsStride, weights.ColumnsStride, result.ColumnsStride);

            if (storeIntermediates)
            {
                _densePreAct.Add(result.Copy());
            }

            _denseActivations[i](result);

            if (storeIntermediates)
            {
                _densePostAct.Add(result.Copy());
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
        for (int i = _denseWeights.Count - 1; i >= 0; i--)
        {
            var preAct = _densePreAct[i];
            var inputToLayer = (i == 0) ? _flattenedInput : _densePostAct[i - 1];

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
            bool skipDeriv = skipLastDerivative && (i == _denseWeights.Count - 1);

            for (int r = 0; r < batch; r++)
            {
                float* rowGO = pGradOut + r * strideGradOut;
                float* rowGP = pGradPre + r * strideGradPre;
                float* rowPA = pPreAct + r * stridePreAct;

                if (skipDeriv)
                {
                    Buffer.MemoryCopy(rowGO, rowGP, outDim * sizeof(float), outDim * sizeof(float));
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

            GpuMatrixOps.ComputeDenseWeightGradientGpu(
                inputToLayer.Pointer,
                gradPre.Pointer,
                dW.Pointer,
                batch, inDim, outDim,
                inputToLayer.ColumnsStride, gradPre.ColumnsStride, dW.ColumnsStride);

            var dB = RentNeural(1, outDim);
            dB.Clear();
            float* pDB = dB.Pointer;

            for (int r = 0; r < batch; r++)
            {
                float* rowGP = pGradPre + r * strideGradPre;
                int cOut = 0;
                if (Avx512F.IsSupported)
                {
                    int limit = outDim - (outDim % 16);
                    for (; cOut < limit; cOut += 16)
                    {
                        (Vector512.Load(pDB + cOut) + Vector512.Load(rowGP + cOut)).Store(pDB + cOut);
                    }
                }
                else if (Avx2.IsSupported)
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

            ClipGradients(dW, GradientClipNorm);
            ClipGradients(dB, GradientClipNorm);

            _denseOptimizers[i].UpdateDenseWeights(_denseWeights[i], _denseBiases[i], dW, dB);

            var weights = _denseWeights[i];
            int weightOutDim = weights.Rows;
            int weightInDim = weights.UsedColumns;
            var gradInput = RentNeural(batch, weightInDim);

            GpuMatrixOps.ComputeDenseInputGradientGpu(
                gradPre.Pointer,
                weights.Pointer,
                gradInput.Pointer,
                batch, weightOutDim, weightInDim,
                gradPre.ColumnsStride, weights.ColumnsStride, gradInput.ColumnsStride);

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
            if (Avx512F.IsSupported)
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
            else if (Avx2.IsSupported)
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
            if (Avx512F.IsSupported)
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
            else if (Avx2.IsSupported)
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

    private void ClearIntermediates()
    {
        DisposeList(_densePostAct);
        DisposeList(_densePreAct);
        DisposeList(_convInputs, skipFirst: true);
        DisposeList(_convPreAct);
        DisposeList(_convPostAct);
        DisposeList(_colInputs);
        DisposeList(_weightMatrices);
        DisposeList(_poolIndices);

        if (_flattenedInput?.Pointer != null)
        {
            _flattenedInput.Dispose();
            _flattenedInput = default;
        }

        _lastPooledOutput = null;
    }

    private static void DisposeList<T>(List<T> list, bool skipFirst = false) where T : IDisposable
    {
        int startIndex = skipFirst ? 1 : 0;
        for (int i = startIndex; i < list.Count; i++)
        {
            list[i].Dispose();
        }

        list.Clear();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static NeuralMatrix RentNeural(int rows, int cols) => NeuralMatrix.GetOrCreate(rows, cols);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static CnnMatrix RentCnn(int batch, int channels, int h, int w) => CnnMatrix.GetOrCreate(batch, channels, h, w);
}
