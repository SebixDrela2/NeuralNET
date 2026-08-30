using NeutralNET.Activation;
using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;
using static NeutralNET.Activation.ActivationSelector;

namespace NeutralNET.Framework.Neural.CNN;

/// <summary>
/// Trainable CNN using Im2Col + dense head.
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

        foreach (var layer in cnnConfig.ConvLayers)
        {
            int fanIn = inChannels * layer.KernelHeight * layer.KernelWidth;
            float stddev = MathF.Sqrt(2.0f / fanIn);

            var weights = new CnnMatrix(layer.Filters, inChannels, layer.KernelHeight, layer.KernelWidth);
            var weightData = weights.ToArray();
            for (int i = 0; i < weights.AllocatedLength; i++)
                weightData[i] = NextGaussianFloat(0, stddev);

            var biases = new CnnMatrix(1, layer.Filters, 1, 1);
            biases.Clear();

            _convWeights.Add(weights);
            _convBiases.Add(biases);
            _convActivationTypes.Add(layer.Activation);

            inChannels = layer.Filters;
        }

        int flattenedSize = ComputeFlattenedSize(cnnConfig);
        int[] denseArch = new int[] { flattenedSize }.Concat(cnnConfig.DenseArchitecture).ToArray();

        for (int i = 0; i < denseArch.Length - 1; i++)
        {
            int inputSize = denseArch[i];
            int outputSize = denseArch[i + 1];
            float stddev = MathF.Sqrt(2.0f / inputSize);

            var weights = new NeuralMatrix(outputSize, inputSize);
            var weightData = weights.ToArray();
            for (int j = 0; j < weights.AllocatedLength; j++)
                weightData[j] = NextGaussianFloat(0, stddev);

            var biases = new NeuralMatrix(1, outputSize);
            biases.Clear();

            _denseWeights.Add(weights);
            _denseBiases.Add(biases);

            ActivationType actType = (i == denseArch.Length - 2)
                ? cnnConfig.OutputActivation
                : cnnConfig.DenseHiddenActivation;
            var act = _activationSelector.GetActivation(actType);
            var der = _activationSelector.GetDerivative(actType);
            _denseActivations.Add(act);
            _denseDerivatives.Add(der);
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

    public float Train(CnnMatrix input, NeuralMatrix target, float learningRate)
    {
        Console.WriteLine("=== TRAIN START ===");
        Console.WriteLine($"Input: {input.Batch}x{input.Channels}x{input.Height}x{input.Width}");
        Console.WriteLine($"Target: {target.Rows}x{target.UsedColumns}");

        ClearIntermediates();

        CnnMatrix current = input;

        for (int layerIdx = 0; layerIdx < _cnnConfig.ConvLayers.Count; layerIdx++)
        {
            var layer = _cnnConfig.ConvLayers[layerIdx];
            Console.WriteLine($"Layer {layerIdx}: Conv {layer.KernelHeight}x{layer.KernelWidth} -> {layer.Filters} filters, Pool: {layer.UseMaxPool}");

            _convInputs.Add(current);

            var (convPreAct, colInput, weightMat) = ConvForward(current, layerIdx, true);
            Console.WriteLine($"  ConvForward: preAct {convPreAct.Batch}x{convPreAct.Channels}x{convPreAct.Height}x{convPreAct.Width}, colInput {colInput.Rows}x{colInput.UsedColumns}");

            _convPreAct.Add(convPreAct);
            _colInputs.Add(colInput);
            _weightMatrices.Add(weightMat);

            var convPostAct = new CnnMatrix(convPreAct.Batch, convPreAct.Channels, convPreAct.Height, convPreAct.Width);
            convPostAct.CopyFrom(convPreAct);
            ApplyActivation(convPostAct, layer.Activation);
            _convPostAct.Add(convPostAct);

            CnnMatrix pooled;
            if (layer.UseMaxPool)
            {
                pooled = MaxPoolForward(convPostAct, layer.PoolSize, out var indices);
                Console.WriteLine($"  MaxPoolForward: pooled {pooled.Batch}x{pooled.Channels}x{pooled.Height}x{pooled.Width}, indices {indices.Rows}x{indices.UsedColumns}");
                _poolIndices.Add(indices);
            }
            else
            {
                pooled = convPostAct;
                _poolIndices.Add(new NeuralMatrix(1, 1));
            }

            current = pooled;
        }

        _lastPooledOutput = current;
        Console.WriteLine($"Last pooled output: {current.Batch}x{current.Channels}x{current.Height}x{current.Width}");

        var flat = Flatten(current);
        _flattenedInput = flat;
        Console.WriteLine($"Flattened: {flat.Rows}x{flat.UsedColumns}");

        var denseOutput = DenseForward(flat, true);
        Console.WriteLine($"Dense output: {denseOutput.Rows}x{denseOutput.UsedColumns}");

        var probabilities = _densePostAct.Last();
        Console.WriteLine($"Probabilities: {probabilities.Rows}x{probabilities.UsedColumns}");

        float loss = ComputeCrossEntropyLoss(probabilities, target);
        Console.WriteLine($"Loss: {loss}");
        Console.WriteLine("Starting backward pass...");

        // Gradient
        var grad = new NeuralMatrix(probabilities.Rows, probabilities.UsedColumns);
        float invBatch = 1.0f / probabilities.Rows;
        for (int r = 0; r < grad.Rows; r++)
        {
            for (int c = 0; c < grad.UsedColumns; c++)
            {
                grad.At(r, c) = (probabilities.At(r, c) - target.At(r, c)) * invBatch;
            }
        }
        Console.WriteLine("  Gradient computed");

        Console.WriteLine("  Calling DenseBackward...");
        var denseGrad = DenseBackward(grad, learningRate, true);
        Console.WriteLine("  DenseBackward complete");

        if (_lastPooledOutput == null)
        {
            throw new InvalidOperationException("_lastPooledOutput is null. Forward pass likely failed.");
        }

        var lastPooled = _lastPooledOutput;
        var pooledGrad = new CnnMatrix(lastPooled.Batch, lastPooled.Channels, lastPooled.Height, lastPooled.Width);
        int flatDim = lastPooled.Channels * lastPooled.Height * lastPooled.Width;
        for (int b = 0; b < pooledGrad.Batch; b++)
        {
            for (int c = 0; c < pooledGrad.Channels; c++)
            {
                for (int y = 0; y < lastPooled.Height; y++)
                {
                    for (int x = 0; x < lastPooled.Width; x++)
                    {
                        int srcIdx = b * flatDim + c * (lastPooled.Height * lastPooled.Width) + y * lastPooled.Width + x;
                        pooledGrad[b, c, y, x] = denseGrad.At(b, srcIdx);
                    }
                }
            }
        }
        denseGrad.Dispose();
        Console.WriteLine("  Unflatten complete");

        CnnMatrix currentGrad = pooledGrad;

        for (int layerIdx = _cnnConfig.ConvLayers.Count - 1; layerIdx >= 0; layerIdx--)
        {
            Console.WriteLine($"  Backward layer {layerIdx}...");
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
                Console.WriteLine($"    MaxPoolBackward...");
                convGrad = MaxPoolBackward(currentGrad, postAct, indices, layer.PoolSize);
                currentGrad.Dispose();
            }
            else
            {
                convGrad = currentGrad;
            }
            Console.WriteLine($"    convGrad: {convGrad.Batch}x{convGrad.Channels}x{convGrad.Height}x{convGrad.Width}");

            var preGrad = new CnnMatrix(preAct.Batch, preAct.Channels, preAct.Height, preAct.Width);
            preGrad.CopyFrom(convGrad);
            Console.WriteLine($"    Applying derivative...");
            ApplyDerivativeToGradient(preGrad, postAct, layer.Activation);
            convGrad.Dispose();
            Console.WriteLine($"    preGrad: {preGrad.Batch}x{preGrad.Channels}x{preGrad.Height}x{preGrad.Width}");

            int outH = preGrad.Height;
            int outW = preGrad.Width;
            int patches = preGrad.Batch * outH * outW;
            int filters = preGrad.Channels;

            Console.WriteLine($"    Creating preGradMatrix {patches}x{filters}...");
            var preGradMatrix = new NeuralMatrix(patches, filters);
            for (int b = 0; b < preGrad.Batch; b++)
            {
                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        for (int f = 0; f < filters; f++)
                        {
                            int patchIdx = (b * outH + oh) * outW + ow;
                            preGradMatrix.At(patchIdx, f) = preGrad[b, f, oh, ow];
                        }
                    }
                }
            }
            Console.WriteLine($"    preGradMatrix complete");

            Console.WriteLine($"    Computing dW...");
            var dW = new NeuralMatrix(colInput.UsedColumns, filters);
            for (int patch = 0; patch < patches; patch++)
            {
                for (int inner = 0; inner < colInput.UsedColumns; inner++)
                {
                    for (int f = 0; f < filters; f++)
                    {
                        dW.At(inner, f) += colInput.At(patch, inner) * preGradMatrix.At(patch, f);
                    }
                }
            }
            Console.WriteLine($"    dW complete");

            Console.WriteLine($"    Computing dB...");
            var dB = new NeuralMatrix(1, filters);
            for (int patch = 0; patch < patches; patch++)
            {
                for (int f = 0; f < filters; f++)
                {
                    dB.At(0, f) += preGradMatrix.At(patch, f);
                }
            }
            Console.WriteLine($"    dB complete");

            Console.WriteLine($"    Updating conv weights...");
            UpdateConvWeights(layerIdx, dW, dB, learningRate);
            Console.WriteLine($"    Update complete");

            Console.WriteLine($"    Computing gradPatchMat...");
            var gradPatchMat = new NeuralMatrix(patches, colInput.UsedColumns);
            for (int patch = 0; patch < patches; patch++)
            {
                for (int inner = 0; inner < colInput.UsedColumns; inner++)
                {
                    float sum = 0;
                    for (int f = 0; f < filters; f++)
                    {
                        sum += preGradMatrix.At(patch, f) * weightMat.At(f, inner);
                    }
                    gradPatchMat.At(patch, inner) = sum;
                }
            }
            Console.WriteLine($"    gradPatchMat complete");

            Console.WriteLine($"    Col2Im...");
            var inputGrad = new CnnMatrix(inputTensor.Batch, inputTensor.Channels, inputTensor.Height, inputTensor.Width);
            inputGrad.Col2Im(gradPatchMat, layer.KernelHeight, layer.KernelWidth, layer.Stride, layer.Padding, 1.0f);
            Console.WriteLine($"    Col2Im complete");

            preGrad.Dispose();
            preGradMatrix.Dispose();
            dW.Dispose();
            dB.Dispose();
            gradPatchMat.Dispose();

            currentGrad = inputGrad;
            Console.WriteLine($"  Layer {layerIdx} backward complete");
        }

        currentGrad?.Dispose();
        Console.WriteLine("Backward pass complete!");

        Console.WriteLine("  Cleaning up intermediates...");
        CleanupIntermediates();
        Console.WriteLine("  Cleanup complete");

        return loss;
    }

    private (CnnMatrix preAct, NeuralMatrix colInput, NeuralMatrix weightMat) ConvForward(CnnMatrix current, int layerIdx, bool storeIntermediates)
    {
        var layer = _cnnConfig.ConvLayers[layerIdx];
        var weights = _convWeights[layerIdx];
        var biases = _convBiases[layerIdx];

        var colInput = current.Im2Col(layer.KernelHeight, layer.KernelWidth, layer.Stride, layer.Padding);

        int innerDim = weights.Channels * weights.Height * weights.Width;
        int filters = weights.Batch;

        var weightMat = new NeuralMatrix(filters, innerDim);
        for (int f = 0; f < filters; f++)
        {
            for (int inner = 0; inner < innerDim; inner++)
            {
                int c = inner / (weights.Height * weights.Width);
                int rem = inner % (weights.Height * weights.Width);
                int ky = rem / weights.Width;
                int kx = rem % weights.Width;
                weightMat.At(f, inner) = weights[f, c, ky, kx];
            }
        }

        var result = new NeuralMatrix(colInput.Rows, filters);
        for (int patch = 0; patch < colInput.Rows; patch++)
        {
            for (int f = 0; f < filters; f++)
            {
                float sum = 0;
                for (int inner = 0; inner < innerDim; inner++)
                {
                    sum += colInput.At(patch, inner) * weightMat.At(f, inner);
                }
                result.At(patch, f) = sum;
            }
        }

        for (int patch = 0; patch < result.Rows; patch++)
        {
            for (int f = 0; f < filters; f++)
            {
                result.At(patch, f) += biases[0, f, 0, 0];
            }
        }

        int outH = (current.Height + 2 * layer.Padding - layer.KernelHeight) / layer.Stride + 1;
        int outW = (current.Width + 2 * layer.Padding - layer.KernelWidth) / layer.Stride + 1;
        var preAct = new CnnMatrix(current.Batch, filters, outH, outW);
        for (int b = 0; b < current.Batch; b++)
        {
            for (int oh = 0; oh < outH; oh++)
            {
                for (int ow = 0; ow < outW; ow++)
                {
                    for (int f = 0; f < filters; f++)
                    {
                        int patchIdx = (b * outH + oh) * outW + ow;
                        preAct[b, f, oh, ow] = result.At(patchIdx, f);
                    }
                }
            }
        }

        result.Dispose();

        if (!storeIntermediates)
        {
            colInput.Dispose();
            weightMat.Dispose();
        }

        return (preAct, colInput, weightMat);
    }

    private CnnMatrix MaxPoolForward(CnnMatrix input, int poolSize, out NeuralMatrix indices)
    {
        int outH = input.Height / poolSize;
        int outW = input.Width / poolSize;
        var pooled = new CnnMatrix(input.Batch, input.Channels, outH, outW);
        var idxMat = new NeuralMatrix(input.Batch * input.Channels * outH * outW, 1);
        int idx = 0;

        for (int b = 0; b < input.Batch; b++)
        {
            for (int c = 0; c < input.Channels; c++)
            {
                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        float maxVal = float.NegativeInfinity;
                        int maxIdx = 0;
                        for (int dy = 0; dy < poolSize; dy++)
                        {
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
                        }
                        pooled[b, c, oh, ow] = maxVal;
                        idxMat.At(idx++, 0) = maxIdx;
                    }
                }
            }
        }

        indices = idxMat;
        return pooled;
    }

    private CnnMatrix MaxPoolBackward(CnnMatrix gradOutput, CnnMatrix input, NeuralMatrix indices, int poolSize)
    {
        var gradInput = new CnnMatrix(input.Batch, input.Channels, input.Height, input.Width);
        gradInput.Clear();

        int outH = gradOutput.Height;
        int outW = gradOutput.Width;
        int idx = 0;
        int totalIndices = indices.Rows;

        for (int b = 0; b < gradOutput.Batch; b++)
        {
            for (int c = 0; c < gradOutput.Channels; c++)
            {
                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        if (idx >= totalIndices)
                        {
                            throw new IndexOutOfRangeException($"MaxPoolBackward: idx {idx} >= {totalIndices}");
                        }
                        int maxIdx = (int)indices.At(idx++, 0);
                        int y = maxIdx / input.Width;
                        int x = maxIdx % input.Width;
                        gradInput[b, c, y, x] += gradOutput[b, c, oh, ow];
                    }
                }
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

            var result = new NeuralMatrix(current.Rows, weights.Rows);
            for (int r = 0; r < current.Rows; r++)
            {
                for (int outNeuron = 0; outNeuron < weights.Rows; outNeuron++)
                {
                    float sum = 0;
                    for (int inFeature = 0; inFeature < current.UsedColumns; inFeature++)
                    {
                        sum += current.At(r, inFeature) * weights.At(outNeuron, inFeature);
                    }
                    result.At(r, outNeuron) = sum + biases.At(0, outNeuron);
                }
            }

            if (storeIntermediates)
            {
                _densePreAct.Add(result.Copy());
            }

            _denseActivations[i](result);

            if (storeIntermediates)
            {
                _densePostAct.Add(result.Copy());
            }

            if (current.Pointer != input.Pointer)
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

            var gradPre = new NeuralMatrix(preAct.Rows, preAct.UsedColumns);
            for (int r = 0; r < gradPre.Rows; r++)
            {
                for (int c = 0; c < gradPre.UsedColumns; c++)
                {
                    gradPre.At(r, c) = gradOutput.At(r, c);
                }
            }

            if (!skipLastDerivative || i < _denseWeights.Count - 1)
            {
                var derivativeFn = _denseDerivatives[i];
                for (int r = 0; r < gradPre.Rows; r++)
                {
                    for (int c = 0; c < gradPre.UsedColumns; c++)
                    {
                        gradPre.At(r, c) *= derivativeFn(preAct.At(r, c));
                    }
                }
            }

            var weights = _denseWeights[i];

            var dW = new NeuralMatrix(inputToLayer.UsedColumns, gradPre.UsedColumns);
            for (int r = 0; r < inputToLayer.Rows; r++)
            {
                for (int cIn = 0; cIn < inputToLayer.UsedColumns; cIn++)
                {
                    for (int cOut = 0; cOut < gradPre.UsedColumns; cOut++)
                    {
                        dW.At(cIn, cOut) += inputToLayer.At(r, cIn) * gradPre.At(r, cOut);
                    }
                }
            }

            var dB = new NeuralMatrix(1, gradPre.UsedColumns);
            for (int r = 0; r < gradPre.Rows; r++)
            {
                for (int c = 0; c < gradPre.UsedColumns; c++)
                {
                    dB.At(0, c) += gradPre.At(r, c);
                }
            }

            UpdateDenseWeights(i, dW, dB, learningRate);

            var gradInput = new NeuralMatrix(gradPre.Rows, weights.UsedColumns);
            for (int r = 0; r < gradPre.Rows; r++)
            {
                for (int cIn = 0; cIn < weights.UsedColumns; cIn++)
                {
                    float sum = 0;
                    for (int cOut = 0; cOut < weights.Rows; cOut++)
                    {
                        sum += gradPre.At(r, cOut) * weights.At(cOut, cIn);
                    }
                    gradInput.At(r, cIn) = sum;
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
        for (int b = 0; b < matrix.Batch; b++)
        {
            for (int c = 0; c < matrix.Channels; c++)
            {
                for (int y = 0; y < matrix.Height; y++)
                {
                    for (int x = 0; x < matrix.Width; x++)
                    {
                        float val = matrix[b, c, y, x];
                        matrix[b, c, y, x] = type switch
                        {
                            ActivationType.ReLU => val < 0 ? 0 : val,
                            ActivationType.Sigmoid => 1.0f / (1.0f + MathF.Exp(-val)),
                            ActivationType.Tanh => MathF.Tanh(val),
                            ActivationType.Identity => val,
                            _ => throw new NotImplementedException($"Activation {type} not implemented")
                        };
                    }
                }
            }
        }
    }

    private void ApplyDerivativeToGradient(CnnMatrix gradient, CnnMatrix postAct, ActivationType type)
    {
        for (int b = 0; b < gradient.Batch; b++)
        {
            for (int c = 0; c < gradient.Channels; c++)
            {
                for (int y = 0; y < gradient.Height; y++)
                {
                    for (int x = 0; x < gradient.Width; x++)
                    {
                        float p = postAct[b, c, y, x];
                        float grad = gradient[b, c, y, x];
                        gradient[b, c, y, x] = type switch
                        {
                            ActivationType.ReLU => p <= 0 ? 0 : grad,
                            ActivationType.Sigmoid => grad * p * (1 - p),
                            ActivationType.Tanh => grad * (1 - p * p),
                            ActivationType.Identity => grad,
                            _ => throw new NotImplementedException($"Derivative for {type} not implemented")
                        };
                    }
                }
            }
        }
    }

    private void UpdateConvWeights(int layerIdx, NeuralMatrix dW, NeuralMatrix dB, float learningRate)
    {
        var weights = _convWeights[layerIdx];
        var biases = _convBiases[layerIdx];

        int innerDim = dW.Rows;
        int filters = dW.UsedColumns;

        for (int f = 0; f < filters; f++)
        {
            for (int inner = 0; inner < innerDim; inner++)
            {
                int c = inner / (weights.Height * weights.Width);
                int rem = inner % (weights.Height * weights.Width);
                int ky = rem / weights.Width;
                int kx = rem % weights.Width;
                weights[f, c, ky, kx] -= learningRate * dW.At(inner, f);
            }
        }

        for (int f = 0; f < filters; f++)
        {
            biases[0, f, 0, 0] -= learningRate * dB.At(0, f);
        }
    }

    private void UpdateDenseWeights(int layerIdx, NeuralMatrix dW, NeuralMatrix dB, float learningRate)
    {
        var weights = _denseWeights[layerIdx];
        var biases = _denseBiases[layerIdx];

        int inputSize = dW.Rows;
        int outputSize = dW.UsedColumns;

        for (int outIdx = 0; outIdx < outputSize; outIdx++)
        {
            for (int inIdx = 0; inIdx < inputSize; inIdx++)
            {
                weights.At(outIdx, inIdx) -= learningRate * dW.At(inIdx, outIdx);
            }
        }

        for (int i = 0; i < biases.UsedColumns; i++)
        {
            biases.At(0, i) -= learningRate * dB.At(0, i);
        }
    }

    private NeuralMatrix Flatten(CnnMatrix input)
    {
        int flatDim = input.Channels * input.Height * input.Width;
        var flat = new NeuralMatrix(input.Batch, flatDim);
        for (int b = 0; b < input.Batch; b++)
        {
            for (int c = 0; c < input.Channels; c++)
            {
                int channelOffset = c * (input.Height * input.Width);
                for (int y = 0; y < input.Height; y++)
                {
                    int rowOffset = y * input.Width;
                    for (int x = 0; x < input.Width; x++)
                    {
                        int flatIdx = channelOffset + rowOffset + x;
                        flat.At(b, flatIdx) = input[b, c, y, x];
                    }
                }
            }
        }
        return flat;
    }

    private float ComputeCrossEntropyLoss(NeuralMatrix predictions, NeuralMatrix targets)
    {
        float loss = 0;
        float eps = 1e-8f;
        for (int r = 0; r < predictions.Rows; r++)
        {
            for (int c = 0; c < predictions.UsedColumns; c++)
            {
                loss -= targets.At(r, c) * MathF.Log(Math.Max(predictions.At(r, c), eps));
            }
        }
        return loss / predictions.Rows;
    }

    private void ClearIntermediates()
    {
        Console.WriteLine("    Clearing intermediates...");

        // Dispose _densePostAct (which holds the same objects as _densePreAct)
        Console.WriteLine("      Disposing _densePostAct...");
        try
        {
            foreach (var m in _densePostAct)
            {
                if (m.Pointer != null)
                {
                    try { m.Dispose(); }
                    catch (Exception ex) { Console.WriteLine($"        Error: {ex.Message}"); }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"        Error in _densePostAct: {ex.Message}");
        }
        _densePostAct.Clear();
        Console.WriteLine("      _densePostAct disposed and cleared.");

        // Just clear _densePreAct (objects are already freed by _densePostAct disposal)
        Console.WriteLine("      Clearing _densePreAct references...");
        _densePreAct.Clear();
        Console.WriteLine("      _densePreAct cleared.");

        // Dispose other lists - skip first element of _convInputs (it's the input owned by caller)
        DisposeList(_convInputs, "_convInputs", skipFirst: true);
        DisposeList(_convPreAct, "_convPreAct");
        DisposeList(_convPostAct, "_convPostAct");
        DisposeList(_colInputs, "_colInputs");
        DisposeList(_weightMatrices, "_weightMatrices");
        DisposeList(_poolIndices, "_poolIndices");

        Console.WriteLine("      Disposing _flattenedInput...");
        if (_flattenedInput?.Pointer != null)
        {
            try { _flattenedInput.Dispose(); }
            catch (Exception ex) { Console.WriteLine($"        Error: {ex.Message}"); }
            _flattenedInput = default;
        }
        Console.WriteLine("      _flattenedInput disposed.");

        _lastPooledOutput = null;
        Console.WriteLine("      Lists cleared.");
        Console.WriteLine("    Intermediates cleared.");
    }

    private void DisposeList<T>(List<T> list, string name, bool skipFirst = false) where T : IDisposable
    {
        Console.WriteLine($"      Disposing {name}...");
        try
        {
            int startIndex = skipFirst ? 1 : 0;
            for (int i = startIndex; i < list.Count; i++)
            {
                try { list[i].Dispose(); }
                catch (Exception ex) { Console.WriteLine($"        Error: {ex.Message}"); }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"        Error in {name}: {ex.Message}");
        }
        list.Clear();
        Console.WriteLine($"      {name} disposed and cleared.");
    }

    private void CleanupIntermediates()
    {
        ClearIntermediates();
    }

    public void Dispose()
    {
        Console.WriteLine("Disposing CnnNeuralFramework...");
        ClearIntermediates();
        foreach (var w in _convWeights) w.Dispose();
        foreach (var b in _convBiases) b.Dispose();
        foreach (var w in _denseWeights) w.Dispose();
        foreach (var b in _denseBiases) b.Dispose();
        Console.WriteLine("Dispose complete.");
    }
}
