using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public class CnnSGDOptimizer : ICnnOptimizer
{
    private readonly float _learningRate;
    private readonly float _weightDecay;
    private readonly float _momentum;

    // State per layer – these will be created per optimizer instance
    private NeuralMatrix? _convVelocityWeights;
    private NeuralMatrix? _convVelocityBiases;
    private NeuralMatrix? _denseVelocityWeights;
    private NeuralMatrix? _denseVelocityBiases;

    public CnnSGDOptimizer(CnnOptimizerConfig config)
    {
        _learningRate = config.LearningRate;
        _weightDecay = config.WeightDecay;
        _momentum = config.Momentum;
    }

    public void UpdateConvWeights(CnnMatrix weights, CnnMatrix biases, NeuralMatrix dW, NeuralMatrix dB)
    {
        int innerDim = dW.Rows;
        int filters = dW.UsedColumns;

        // Initialize velocities if needed
        if (_convVelocityWeights == null || _convVelocityWeights.Rows != innerDim || _convVelocityWeights.UsedColumns != filters)
        {
            _convVelocityWeights?.Dispose();
            _convVelocityWeights = NeuralMatrix.GetOrCreate(innerDim, filters);
            _convVelocityWeights.Clear();
        }
        if (_convVelocityBiases == null || _convVelocityBiases.Rows != 1 || _convVelocityBiases.UsedColumns != filters)
        {
            _convVelocityBiases?.Dispose();
            _convVelocityBiases = NeuralMatrix.GetOrCreate(1, filters);
            _convVelocityBiases.Clear();
        }

        float lr = _learningRate;
        float wd = _weightDecay;
        float mu = _momentum;

        for (int f = 0; f < filters; f++)
            for (int inner = 0; inner < innerDim; inner++)
            {
                int c = inner / (weights.Height * weights.Width);
                int rem = inner % (weights.Height * weights.Width);
                int ky = rem / weights.Width;
                int kx = rem % weights.Width;

                float grad = dW.At(inner, f) + wd * weights[f, c, ky, kx];
                float vel = mu * _convVelocityWeights.At(inner, f) - lr * grad;
                _convVelocityWeights.At(inner, f) = vel;
                weights[f, c, ky, kx] += vel;
            }

        for (int f = 0; f < filters; f++)
        {
            float grad = dB.At(0, f);
            float vel = mu * _convVelocityBiases.At(0, f) - lr * grad;
            _convVelocityBiases.At(0, f) = vel;
            biases[0, f, 0, 0] += vel;
        }
    }

    public void UpdateDenseWeights(NeuralMatrix weights, NeuralMatrix biases, NeuralMatrix dW, NeuralMatrix dB)
    {
        int inputSize = dW.Rows;
        int outputSize = dW.UsedColumns;

        if (_denseVelocityWeights == null || _denseVelocityWeights.Rows != inputSize || _denseVelocityWeights.UsedColumns != outputSize)
        {
            _denseVelocityWeights?.Dispose();
            _denseVelocityWeights = NeuralMatrix.GetOrCreate(inputSize, outputSize);
            _denseVelocityWeights.Clear();
        }
        if (_denseVelocityBiases == null || _denseVelocityBiases.Rows != 1 || _denseVelocityBiases.UsedColumns != outputSize)
        {
            _denseVelocityBiases?.Dispose();
            _denseVelocityBiases = NeuralMatrix.GetOrCreate(1, outputSize);
            _denseVelocityBiases.Clear();
        }

        float lr = _learningRate;
        float wd = _weightDecay;
        float mu = _momentum;

        for (int outIdx = 0; outIdx < outputSize; outIdx++)
            for (int inIdx = 0; inIdx < inputSize; inIdx++)
            {
                float grad = dW.At(inIdx, outIdx) + wd * weights.At(outIdx, inIdx);
                float vel = mu * _denseVelocityWeights.At(inIdx, outIdx) - lr * grad;
                _denseVelocityWeights.At(inIdx, outIdx) = vel;
                weights.At(outIdx, inIdx) += vel;
            }

        for (int i = 0; i < outputSize; i++)
        {
            float grad = dB.At(0, i);
            float vel = mu * _denseVelocityBiases.At(0, i) - lr * grad;
            _denseVelocityBiases.At(0, i) = vel;
            biases.At(0, i) += vel;
        }
    }

    public void Dispose()
    {
        _convVelocityWeights?.Dispose();
        _convVelocityBiases?.Dispose();
        _denseVelocityWeights?.Dispose();
        _denseVelocityBiases?.Dispose();
    }
}
