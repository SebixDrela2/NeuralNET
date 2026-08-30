using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public class CnnAdamOptimizer : ICnnOptimizer
{
    private readonly float _learningRate;
    private readonly float _weightDecay;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;

    private int _t;

    // Conv state
    private NeuralMatrix? _convMWeights;
    private NeuralMatrix? _convVWeights;
    private NeuralMatrix? _convMBiases;
    private NeuralMatrix? _convVBiases;

    // Dense state
    private NeuralMatrix? _denseMWeights;
    private NeuralMatrix? _denseVWeights;
    private NeuralMatrix? _denseMBiases;
    private NeuralMatrix? _denseVBiases;

    public CnnAdamOptimizer(CnnOptimizerConfig config)
    {
        _learningRate = config.LearningRate;
        _weightDecay = config.WeightDecay;
        _beta1 = config.Beta1;
        _beta2 = config.Beta2;
        _epsilon = config.Epsilon;
        _t = 0;
    }

    public void UpdateConvWeights(CnnMatrix weights, CnnMatrix biases, NeuralMatrix dW, NeuralMatrix dB)
    {
        int innerDim = dW.Rows;
        int filters = dW.UsedColumns;

        if (_convMWeights == null || _convMWeights.Rows != innerDim || _convMWeights.UsedColumns != filters)
        {
            _convMWeights?.Dispose();
            _convVWeights?.Dispose();
            _convMWeights = NeuralMatrix.GetOrCreate(innerDim, filters);
            _convVWeights = NeuralMatrix.GetOrCreate(innerDim, filters);
            _convMWeights.Clear();
            _convVWeights.Clear();
        }
        if (_convMBiases == null || _convMBiases.Rows != 1 || _convMBiases.UsedColumns != filters)
        {
            _convMBiases?.Dispose();
            _convVBiases?.Dispose();
            _convMBiases = NeuralMatrix.GetOrCreate(1, filters);
            _convVBiases = NeuralMatrix.GetOrCreate(1, filters);
            _convMBiases.Clear();
            _convVBiases.Clear();
        }

        _t++;
        float lr = _learningRate;
        float wd = _weightDecay;
        float b1 = _beta1;
        float b2 = _beta2;
        float eps = _epsilon;
        float t = _t;

        for (int f = 0; f < filters; f++)
            for (int inner = 0; inner < innerDim; inner++)
            {
                int c = inner / (weights.Height * weights.Width);
                int rem = inner % (weights.Height * weights.Width);
                int ky = rem / weights.Width;
                int kx = rem % weights.Width;

                float grad = dW.At(inner, f) + wd * weights[f, c, ky, kx];

                float m = b1 * _convMWeights.At(inner, f) + (1 - b1) * grad;
                _convMWeights.At(inner, f) = m;
                float v = b2 * _convVWeights.At(inner, f) + (1 - b2) * grad * grad;
                _convVWeights.At(inner, f) = v;

                float mHat = m / (1 - MathF.Pow(b1, t));
                float vHat = v / (1 - MathF.Pow(b2, t));

                weights[f, c, ky, kx] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
            }

        for (int f = 0; f < filters; f++)
        {
            float grad = dB.At(0, f);
            float m = b1 * _convMBiases.At(0, f) + (1 - b1) * grad;
            _convMBiases.At(0, f) = m;
            float v = b2 * _convVBiases.At(0, f) + (1 - b2) * grad * grad;
            _convVBiases.At(0, f) = v;
            float mHat = m / (1 - MathF.Pow(b1, t));
            float vHat = v / (1 - MathF.Pow(b2, t));
            biases[0, f, 0, 0] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
        }
    }

    public void UpdateDenseWeights(NeuralMatrix weights, NeuralMatrix biases, NeuralMatrix dW, NeuralMatrix dB)
    {
        int inputSize = dW.Rows;
        int outputSize = dW.UsedColumns;

        if (_denseMWeights == null || _denseMWeights.Rows != inputSize || _denseMWeights.UsedColumns != outputSize)
        {
            _denseMWeights?.Dispose();
            _denseVWeights?.Dispose();
            _denseMWeights = NeuralMatrix.GetOrCreate(inputSize, outputSize);
            _denseVWeights = NeuralMatrix.GetOrCreate(inputSize, outputSize);
            _denseMWeights.Clear();
            _denseVWeights.Clear();
        }
        if (_denseMBiases == null || _denseMBiases.Rows != 1 || _denseMBiases.UsedColumns != outputSize)
        {
            _denseMBiases?.Dispose();
            _denseVBiases?.Dispose();
            _denseMBiases = NeuralMatrix.GetOrCreate(1, outputSize);
            _denseVBiases = NeuralMatrix.GetOrCreate(1, outputSize);
            _denseMBiases.Clear();
            _denseVBiases.Clear();
        }

        _t++;
        float lr = _learningRate;
        float wd = _weightDecay;
        float b1 = _beta1;
        float b2 = _beta2;
        float eps = _epsilon;
        float t = _t;

        for (int outIdx = 0; outIdx < outputSize; outIdx++)
            for (int inIdx = 0; inIdx < inputSize; inIdx++)
            {
                float grad = dW.At(inIdx, outIdx) + wd * weights.At(outIdx, inIdx);
                float m = b1 * _denseMWeights.At(inIdx, outIdx) + (1 - b1) * grad;
                _denseMWeights.At(inIdx, outIdx) = m;
                float v = b2 * _denseVWeights.At(inIdx, outIdx) + (1 - b2) * grad * grad;
                _denseVWeights.At(inIdx, outIdx) = v;
                float mHat = m / (1 - MathF.Pow(b1, t));
                float vHat = v / (1 - MathF.Pow(b2, t));
                weights.At(outIdx, inIdx) -= lr * mHat / (MathF.Sqrt(vHat) + eps);
            }

        for (int i = 0; i < outputSize; i++)
        {
            float grad = dB.At(0, i);
            float m = b1 * _denseMBiases.At(0, i) + (1 - b1) * grad;
            _denseMBiases.At(0, i) = m;
            float v = b2 * _denseVBiases.At(0, i) + (1 - b2) * grad * grad;
            _denseVBiases.At(0, i) = v;
            float mHat = m / (1 - MathF.Pow(b1, t));
            float vHat = v / (1 - MathF.Pow(b2, t));
            biases.At(0, i) -= lr * mHat / (MathF.Sqrt(vHat) + eps);
        }
    }

    public void Dispose()
    {
        _convMWeights?.Dispose();
        _convVWeights?.Dispose();
        _convMBiases?.Dispose();
        _convVBiases?.Dispose();
        _denseMWeights?.Dispose();
        _denseVWeights?.Dispose();
        _denseMBiases?.Dispose();
        _denseVBiases?.Dispose();
    }
}
