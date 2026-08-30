using NeutralNET.Activation;
using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;
using static NeutralNET.Activation.ActivationSelector;

namespace NeutralNET.Framework.Neural.CNN;

/// <summary>
/// Zero‑GC CNN framework with full object and buffer pooling, and pluggable optimizers.
/// </summary>
public unsafe class CnnNeuralFramework<TArch> : IDisposable
    where TArch : IArchitecture<TArch>
{
    private readonly NeuralNetworkConfig _baseConfig;
    private readonly CnnArchitectureConfig _cnnConfig;
    private readonly ActivationSelector _activationSelector = new();
    private readonly int _inputHeight;
    private readonly int _inputWidth;
    private readonly int _inputChannels;

    private readonly List<CnnMatrix> _convWeights = new();
    private readonly List<CnnMatrix> _convBiases = new();
    private readonly List<ActivationType> _convActivationTypes = new();

    private readonly List<NeuralMatrix> _denseWeights = new();
    private readonly List<NeuralMatrix> _denseBiases = new();
    private readonly List<ActivationFunction> _denseActivations = new();
    private readonly List<DerivativeFunction> _denseDerivatives = new();

    // Optimizers per layer
    private readonly List<ICnnOptimizer> _convOptimizers = new();
    private readonly List<ICnnOptimizer> _denseOptimizers = new();

    private readonly List<CnnMatrix> _convInputs = new();
    private readonly List<CnnMatrix> _convPreAct = new();
    private readonly List<CnnMatrix> _convPostAct = new();
    private readonly List<NeuralMatrix> _colInputs = new();
    private readonly List<NeuralMatrix> _weightMatrices = new();
    private readonly List<NeuralMatrix> _poolIndices = new();
    private readonly List<NeuralMatrix> _densePreAct = new();
    private readonly List<NeuralMatrix> _densePostAct = new();

    private NeuralMatrix? _flattenedInput;
    private CnnMatrix _lastPooledOutput;

    private readonly Random _rng;
    private static NeuralMatrix RentNeural(int rows, int cols) => NeuralMatrix.GetOrCreate(rows, cols);
    private static CnnMatrix RentCnn(int batch, int channels, int h, int w) => CnnMatrix.GetOrCreate(batch, channels, h, w);

    // ---------- Constructor ----------
    public CnnNeuralFramework(NeuralNetworkConfig baseConfig, CnnArchitectureConfig cnnConfig,
        int inputHeight = 32, int inputWidth = 32, int inputChannels = 3)
    {
        _baseConfig = baseConfig;
        _cnnConfig = cnnConfig;
        _rng = new Random();
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _inputChannels = inputChannels;

        int inChannels = inputChannels;

        // Conv layers
        foreach (var layer in cnnConfig.ConvLayers)
        {
            var fanIn = inChannels * layer.KernelHeight * layer.KernelWidth;
            var stddev = MathF.Sqrt(2.0f / fanIn);

            var weights = RentCnn(layer.Filters, inChannels, layer.KernelHeight, layer.KernelWidth);
            for (int f = 0; f < layer.Filters; f++)
                for (int c = 0; c < inChannels; c++)
                    for (int y = 0; y < layer.KernelHeight; y++)
                        for (int x = 0; x < layer.KernelWidth; x++)
                            weights[f, c, y, x] = NextGaussianFloat(0, stddev);

            var biases = RentCnn(1, layer.Filters, 1, 1);
            for (int f = 0; f < layer.Filters; f++)
                biases[0, f, 0, 0] = NextGaussianFloat(0, 0.1f);

            _convWeights.Add(weights);
            _convBiases.Add(biases);
            _convActivationTypes.Add(layer.Activation);

            // Create optimizer for this conv layer
            var opt = CnnOptimizerFactory.Create(cnnConfig.OptimizerConfig);
            _convOptimizers.Add(opt);

            inChannels = layer.Filters;
        }

        var flattenedSize = ComputeFlattenedSize(cnnConfig);
        var denseArch = new int[] { flattenedSize }.Concat(cnnConfig.DenseArchitecture).ToArray();

        // Dense layers
        for (int i = 0; i < denseArch.Length - 1; i++)
        {
            var inputSize = denseArch[i];
            var outputSize = denseArch[i + 1];
            float stddev = MathF.Sqrt(2.0f / inputSize);

            var weights = RentNeural(outputSize, inputSize);
            for (int outIdx = 0; outIdx < outputSize; outIdx++)
                for (int inIdx = 0; inIdx < inputSize; inIdx++)
                    weights.At(outIdx, inIdx) = NextGaussianFloat(0, stddev);

            var biases = RentNeural(1, outputSize);
            for (int j = 0; j < outputSize; j++)
                biases.At(0, j) = NextGaussianFloat(0, 0.1f);

            _denseWeights.Add(weights);
            _denseBiases.Add(biases);

            ActivationType actType = (i == denseArch.Length - 2)
                ? cnnConfig.OutputActivation
                : cnnConfig.DenseHiddenActivation;

            var act = _activationSelector.GetActivation(actType);
            var der = _activationSelector.GetDerivative(actType);

            _denseActivations.Add(act);
            _denseDerivatives.Add(der);

            // Create optimizer for this dense layer
            var opt = CnnOptimizerFactory.Create(cnnConfig.OptimizerConfig);
            _denseOptimizers.Add(opt);
        }
    }

    private float NextGaussianFloat(float mean, float stddev)
    {
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = 1.0 - _rng.NextDouble();
        return (float)(mean + stddev * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
    }

    private int ComputeFlattenedSize(CnnArchitectureConfig config)
    {
        int h = _inputHeight, w = _inputWidth, channels = _inputChannels;
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

    // ---------- Forward (Inference) ----------
    public NeuralMatrix Forward(CnnMatrix input)
    {
        CnnMatrix current = input;
        for (int layerIdx = 0; layerIdx < _cnnConfig.ConvLayers.Count; layerIdx++)
        {
            var layer = _cnnConfig.ConvLayers[layerIdx];
            var (convOut, colInput, weightMat) = ConvForward(current, layerIdx, false);
            ApplyActivation(convOut, layer.Activation);

            if (layer.UseMaxPool)
            {
                var pooled = MaxPoolForward(convOut, layer.PoolSize, out _);
                convOut.Dispose();
                current = pooled;
            }
            else
            {
                current = convOut;
            }

            colInput.Dispose();
            weightMat.Dispose();
        }

        var flat = Flatten(current);
        current.Dispose();

        var denseOut = DenseForward(flat, false);
        flat.Dispose();

        return denseOut;
    }

    // ---------- Training ----------
    public float Train(CnnMatrix input, NeuralMatrix target, float learningRate)
    {
        ClearIntermediates();
        CnnMatrix current = input;

        for (int layerIdx = 0; layerIdx < _cnnConfig.ConvLayers.Count; layerIdx++)
        {
            var layer = _cnnConfig.ConvLayers[layerIdx];
            _convInputs.Add(current);

            var (convPreAct, colInput, weightMat) = ConvForward(current, layerIdx, true);
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

        _lastPooledOutput = current;

        var flat = Flatten(current);
        _flattenedInput = flat;

        var denseOutput = DenseForward(flat, true);
        var probabilities = _densePostAct.Last();
        var loss = ComputeCrossEntropyLoss(probabilities, target);

        var grad = RentNeural(probabilities.Rows, probabilities.UsedColumns);
        float invBatch = 1.0f / probabilities.Rows;
        for (int r = 0; r < grad.Rows; r++)
            for (int c = 0; c < grad.UsedColumns; c++)
                grad.At(r, c) = (probabilities.At(r, c) - target.At(r, c)) * invBatch;

        var denseGrad = DenseBackward(grad, learningRate, true);

        if (_lastPooledOutput == null)
            throw new InvalidOperationException("_lastPooledOutput is null.");

        var lastPooled = _lastPooledOutput;
        var pooledGrad = RentCnn(lastPooled.Batch, lastPooled.Channels, lastPooled.Height, lastPooled.Width);
        int flatDim = lastPooled.Channels * lastPooled.Height * lastPooled.Width;

        for (int b = 0; b < pooledGrad.Batch; b++)
            for (int c = 0; c < pooledGrad.Channels; c++)
                for (int y = 0; y < lastPooled.Height; y++)
                    for (int x = 0; x < lastPooled.Width; x++)
                    {
                        int srcIdx = c * (lastPooled.Height * lastPooled.Width) + y * lastPooled.Width + x;
                        pooledGrad[b, c, y, x] = denseGrad.At(b, srcIdx);
                    }

        denseGrad.Dispose();

        CnnMatrix currentGrad = pooledGrad;

        for (int layerIdx = _cnnConfig.ConvLayers.Count - 1; layerIdx >= 0; layerIdx--)
        {
            var layer = _cnnConfig.ConvLayers[layerIdx];
            var preAct = _convPreAct[layerIdx];
            var postAct = _convPostAct[layerIdx];
            var colInput = _colInputs[layerIdx];
            var weightMat = _weightMatrices[layerIdx];
            var inputTensor = _convInputs[layerIdx];
            var indices = _poolIndices[layerIdx];

            CnnMatrix convGrad;
            if (layer.UseMaxPool)
            {
                convGrad = MaxPoolBackward(currentGrad, postAct, indices, layer.PoolSize);
                currentGrad.Dispose();
            }
            else
            {
                convGrad = currentGrad;
            }

            var preGrad = RentCnn(preAct.Batch, preAct.Channels, preAct.Height, preAct.Width);
            preGrad.CopyFrom(convGrad);
            ApplyDerivativeToGradient(preGrad, postAct, layer.Activation);
            convGrad.Dispose();

            int outH = preGrad.Height;
            int outW = preGrad.Width;
            int patches = preGrad.Batch * outH * outW;
            int filters = preGrad.Channels;
            var preGradMatrix = RentNeural(patches, filters);

            for (int b = 0; b < preGrad.Batch; b++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++)
                        for (int f = 0; f < filters; f++)
                        {
                            int patchIdx = (b * outH + oh) * outW + ow;
                            preGradMatrix.At(patchIdx, f) = preGrad[b, f, oh, ow];
                        }

            var dW = RentNeural(colInput.UsedColumns, filters);
            for (int patch = 0; patch < patches; patch++)
                for (int inner = 0; inner < colInput.UsedColumns; inner++)
                    for (int f = 0; f < filters; f++)
                        dW.At(inner, f) += colInput.At(patch, inner) * preGradMatrix.At(patch, f);

            var dB = RentNeural(1, filters);
            for (int patch = 0; patch < patches; patch++)
                for (int f = 0; f < filters; f++)
                    dB.At(0, f) += preGradMatrix.At(patch, f);

            // Use the optimizer to update conv weights
            _convOptimizers[layerIdx].UpdateConvWeights(_convWeights[layerIdx], _convBiases[layerIdx], dW, dB);

            var gradPatchMat = RentNeural(patches, colInput.UsedColumns);
            for (int patch = 0; patch < patches; patch++)
                for (int inner = 0; inner < colInput.UsedColumns; inner++)
                {
                    float sum = 0;
                    for (int f = 0; f < filters; f++)
                        sum += preGradMatrix.At(patch, f) * weightMat.At(f, inner);
                    gradPatchMat.At(patch, inner) = sum;
                }

            var inputGrad = RentCnn(inputTensor.Batch, inputTensor.Channels, inputTensor.Height, inputTensor.Width);
            inputGrad.Col2Im(gradPatchMat, layer.KernelHeight, layer.KernelWidth, layer.Stride, layer.Padding, 1.0f);

            preGrad.Dispose();
            preGradMatrix.Dispose();
            dW.Dispose();
            dB.Dispose();
            gradPatchMat.Dispose();

            currentGrad = inputGrad;
        }

        currentGrad?.Dispose();

        CleanupIntermediates();
        return loss;
    }

    // ---------- Conv Forward Helpers ----------
    private (CnnMatrix preAct, NeuralMatrix colInput, NeuralMatrix weightMat) ConvForward(CnnMatrix current, int layerIdx, bool storeIntermediates)
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

        if (!storeIntermediates)
        {
            colInput.Dispose();
            weightMat.Dispose();
        }

        return (preAct, colInput, weightMat);
    }

    private NeuralMatrix CreateWeightMatrix(CnnMatrix weights)
    {
        int innerDim = weights.Channels * weights.Height * weights.Width;
        int filters = weights.Batch;
        var weightMat = RentNeural(filters, innerDim);

        for (int f = 0; f < filters; f++)
            for (int inner = 0; inner < innerDim; inner++)
            {
                int c = inner / (weights.Height * weights.Width);
                int rem = inner % (weights.Height * weights.Width);
                int ky = rem / weights.Width;
                int kx = rem % weights.Width;
                weightMat.At(f, inner) = weights[f, c, ky, kx];
            }

        return weightMat;
    }

    private NeuralMatrix ComputeConvolution(NeuralMatrix colInput, NeuralMatrix weightMat)
    {
        int patches = colInput.Rows;
        int filters = weightMat.Rows;
        int innerDim = colInput.UsedColumns;

        var result = RentNeural(patches, filters);

        for (int patch = 0; patch < patches; patch++)
            for (int f = 0; f < filters; f++)
            {
                float sum = 0;
                for (int inner = 0; inner < innerDim; inner++)
                    sum += colInput.At(patch, inner) * weightMat.At(f, inner);
                result.At(patch, f) = sum;
            }

        return result;
    }

    private void AddBias(NeuralMatrix result, CnnMatrix biases)
    {
        int filters = biases.Channels;
        for (int patch = 0; patch < result.Rows; patch++)
            for (int f = 0; f < filters; f++)
                result.At(patch, f) += biases[0, f, 0, 0];
    }

    private CnnMatrix ReshapeToCnnMatrix(NeuralMatrix result, int batchSize, int filters, int height, int width, CnnLayerConfig layer)
    {
        int outH = (height + 2 * layer.Padding - layer.KernelHeight) / layer.Stride + 1;
        int outW = (width + 2 * layer.Padding - layer.KernelWidth) / layer.Stride + 1;
        int expectedRows = batchSize * outH * outW;

        if (result.Rows != expectedRows)
            throw new InvalidOperationException($"result.Rows ({result.Rows}) != expectedRows ({expectedRows}).");
        if (result.UsedColumns != filters)
            throw new InvalidOperationException($"result.UsedColumns ({result.UsedColumns}) != filters ({filters})");

        var preAct = RentCnn(batchSize, filters, outH, outW);

        for (int b = 0; b < batchSize; b++)
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++)
                {
                    int patchIdx = (b * outH + oh) * outW + ow;
                    for (int f = 0; f < filters; f++)
                        preAct[b, f, oh, ow] = result.At(patchIdx, f);
                }

        return preAct;
    }

    // ---------- MaxPool ----------
    private CnnMatrix MaxPoolForward(CnnMatrix input, int poolSize, out NeuralMatrix indices)
    {
        int outH = input.Height / poolSize;
        int outW = input.Width / poolSize;
        var pooled = RentCnn(input.Batch, input.Channels, outH, outW);
        var idxMat = RentNeural(input.Batch * input.Channels * outH * outW, 1);
        int idx = 0;

        for (int b = 0; b < input.Batch; b++)
            for (int c = 0; c < input.Channels; c++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++)
                    {
                        float maxVal = float.NegativeInfinity;
                        int maxIdx = 0;
                        for (int dy = 0; dy < poolSize; dy++)
                            for (int dx = 0; dx < poolSize; dx++)
                            {
                                int y = oh * poolSize + dy;
                                int x = ow * poolSize + dx;
                                float val = input[b, c, y, x];
                                if (val > maxVal)
                                {
                                    maxVal = val;
                                    maxIdx = y * input.Width + x;
                                }
                            }
                        pooled[b, c, oh, ow] = maxVal;
                        idxMat.At(idx++, 0) = maxIdx;
                    }

        indices = idxMat;
        return pooled;
    }

    private CnnMatrix MaxPoolBackward(CnnMatrix gradOutput, CnnMatrix input, NeuralMatrix indices, int poolSize)
    {
        var gradInput = RentCnn(input.Batch, input.Channels, input.Height, input.Width);
        gradInput.Clear();

        int outH = gradOutput.Height;
        int outW = gradOutput.Width;
        int idx = 0;
        int totalIndices = indices.Rows;

        for (int b = 0; b < gradOutput.Batch; b++)
            for (int c = 0; c < gradOutput.Channels; c++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++)
                    {
                        if (idx >= totalIndices)
                            throw new IndexOutOfRangeException($"MaxPoolBackward: idx {idx} >= {totalIndices}");
                        int maxIdx = (int)indices.At(idx++, 0);
                        int y = maxIdx / input.Width;
                        int x = maxIdx % input.Width;
                        gradInput[b, c, y, x] += gradOutput[b, c, oh, ow];
                    }

        return gradInput;
    }

    // ---------- Dense Forward ----------
    private NeuralMatrix DenseForward(NeuralMatrix input, bool storeIntermediates)
    {
        var current = input;
        for (int i = 0; i < _denseWeights.Count; i++)
        {
            var weights = _denseWeights[i];
            var biases = _denseBiases[i];
            var result = RentNeural(current.Rows, weights.Rows);

            for (int r = 0; r < current.Rows; r++)
                for (int outNeuron = 0; outNeuron < weights.Rows; outNeuron++)
                {
                    float sum = 0;
                    for (int inFeature = 0; inFeature < current.UsedColumns; inFeature++)
                        sum += current.At(r, inFeature) * weights.At(outNeuron, inFeature);
                    result.At(r, outNeuron) = sum + biases.At(0, outNeuron);
                }

            if (storeIntermediates)
            {
                _densePreAct.Add(result.Copy());
            }

            _denseActivations[i](result);

            if (storeIntermediates)
            {
                _densePostAct.Add(result);
            }

            current = result;
        }
        return current;
    }

    // ---------- Dense Backward ----------
    private NeuralMatrix DenseBackward(NeuralMatrix gradOutput, float learningRate, bool skipLastDerivative = false)
    {
        for (int i = _denseWeights.Count - 1; i >= 0; i--)
        {
            var preAct = _densePreAct[i];
            var inputToLayer = (i == 0) ? _flattenedInput : _densePostAct[i - 1];

            if (inputToLayer == null)
                throw new InvalidOperationException($"inputToLayer is null for layer {i}. _densePostAct[{i - 1}] may not be set.");

            var gradPre = RentNeural(preAct.Rows, preAct.UsedColumns);

            for (int r = 0; r < gradPre.Rows; r++)
                for (int c = 0; c < gradPre.UsedColumns; c++)
                    gradPre.At(r, c) = gradOutput.At(r, c);

            if (!skipLastDerivative || i < _denseWeights.Count - 1)
            {
                var derivativeFn = _denseDerivatives[i];
                for (int r = 0; r < gradPre.Rows; r++)
                    for (int c = 0; c < gradPre.UsedColumns; c++)
                        gradPre.At(r, c) *= derivativeFn(preAct.At(r, c));
            }

            var weights = _denseWeights[i];
            var dW = RentNeural(inputToLayer.UsedColumns, gradPre.UsedColumns);
            for (int r = 0; r < inputToLayer.Rows; r++)
                for (int cIn = 0; cIn < inputToLayer.UsedColumns; cIn++)
                    for (int cOut = 0; cOut < gradPre.UsedColumns; cOut++)
                        dW.At(cIn, cOut) += inputToLayer.At(r, cIn) * gradPre.At(r, cOut);

            var dB = RentNeural(1, gradPre.UsedColumns);
            for (int r = 0; r < gradPre.Rows; r++)
                for (int c = 0; c < gradPre.UsedColumns; c++)
                    dB.At(0, c) += gradPre.At(r, c);

            // Use optimizer for dense weights
            _denseOptimizers[i].UpdateDenseWeights(_denseWeights[i], _denseBiases[i], dW, dB);

            var gradInput = RentNeural(gradPre.Rows, weights.UsedColumns);
            for (int r = 0; r < gradPre.Rows; r++)
                for (int cIn = 0; cIn < weights.UsedColumns; cIn++)
                {
                    float sum = 0;
                    for (int cOut = 0; cOut < weights.Rows; cOut++)
                        sum += gradPre.At(r, cOut) * weights.At(cOut, cIn);
                    gradInput.At(r, cIn) = sum;
                }

            gradOutput.Dispose();
            dW.Dispose();
            dB.Dispose();
            gradPre.Dispose();

            gradOutput = gradInput;
        }

        return gradOutput;
    }

    // ---------- Utilities ----------
    private void ApplyActivation(CnnMatrix matrix, ActivationType type)
    {
        for (int b = 0; b < matrix.Batch; b++)
            for (int c = 0; c < matrix.Channels; c++)
                for (int y = 0; y < matrix.Height; y++)
                    for (int x = 0; x < matrix.Width; x++)
                    {
                        float val = matrix[b, c, y, x];
                        matrix[b, c, y, x] = type switch
                        {
                            ActivationType.ReLU => val < 0 ? 0 : val,
                            ActivationType.LeakyReLU => val < 0 ? 0.01f * val : val,
                            ActivationType.Sigmoid => 1.0f / (1.0f + MathF.Exp(-val)),
                            ActivationType.Tanh => MathF.Tanh(val),
                            ActivationType.Identity => val,
                            _ => throw new NotImplementedException($"Activation {type} not implemented")
                        };
                    }
    }

    private void ApplyDerivativeToGradient(CnnMatrix gradient, CnnMatrix postAct, ActivationType type)
    {
        for (int b = 0; b < gradient.Batch; b++)
            for (int c = 0; c < gradient.Channels; c++)
                for (int y = 0; y < gradient.Height; y++)
                    for (int x = 0; x < gradient.Width; x++)
                    {
                        float p = postAct[b, c, y, x];
                        float grad = gradient[b, c, y, x];
                        gradient[b, c, y, x] = type switch
                        {
                            ActivationType.ReLU => p <= 0 ? 0 : grad,
                            ActivationType.LeakyReLU => p <= 0 ? 0.01f * grad : grad,
                            ActivationType.Sigmoid => grad * p * (1 - p),
                            ActivationType.Tanh => grad * (1 - p * p),
                            ActivationType.Identity => grad,
                            _ => throw new NotImplementedException($"Derivative for {type} not implemented")
                        };
                    }
    }

    private NeuralMatrix Flatten(CnnMatrix input)
    {
        int flatDim = input.Channels * input.Height * input.Width;
        var flat = RentNeural(input.Batch, flatDim);
        for (int b = 0; b < input.Batch; b++)
            for (int c = 0; c < input.Channels; c++)
                for (int y = 0; y < input.Height; y++)
                    for (int x = 0; x < input.Width; x++)
                    {
                        int idx = c * (input.Height * input.Width) + y * input.Width + x;
                        flat.At(b, idx) = input[b, c, y, x];
                    }
        return flat;
    }

    private float ComputeCrossEntropyLoss(NeuralMatrix predictions, NeuralMatrix targets)
    {
        float loss = 0;
        float eps = 1e-8f;
        for (int r = 0; r < predictions.Rows; r++)
            for (int c = 0; c < predictions.UsedColumns; c++)
                loss -= targets.At(r, c) * MathF.Log(Math.Max(predictions.At(r, c), eps));
        return loss / predictions.Rows;
    }

    // ---------- Cleanup ----------
    private void ClearIntermediates()
    {
        foreach (var m in _densePostAct) m.Dispose();
        _densePostAct.Clear();
        _densePreAct.Clear();

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
        _lastPooledOutput = null!;
    }

    private void DisposeList<T>(List<T> list, bool skipFirst = false) where T : IDisposable
    {
        int startIndex = skipFirst ? 1 : 0;
        for (int i = startIndex; i < list.Count; i++)
            list[i].Dispose();
        list.Clear();
    }

    private void CleanupIntermediates() => ClearIntermediates();

    public void Dispose()
    {
        ClearIntermediates();

        // Dispose weights and biases
        foreach (var w in _convWeights) w.Dispose();
        foreach (var b in _convBiases) b.Dispose();
        foreach (var w in _denseWeights) w.Dispose();
        foreach (var b in _denseBiases) b.Dispose();

        // Dispose optimizers
        foreach (var opt in _convOptimizers) opt.Dispose();
        foreach (var opt in _denseOptimizers) opt.Dispose();
    }
}
