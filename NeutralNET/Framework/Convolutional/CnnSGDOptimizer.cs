using System;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public class CnnSGDOptimizer : ICnnOptimizer, IDisposable
{
    private readonly float _learningRate;
    private readonly float _weightDecay;
    private readonly float _momentum;

    private readonly OptimizerHyperLayerParameterSet _convHyperParameters;
    private readonly OptimizerHyperLayerParameterSet _denseHyperParameters;

    private bool _disposed;

    public CnnSGDOptimizer(
        CnnOptimizerConfig config,
        OptimizerHyperLayerParameterSet convHyperParameters,
        OptimizerHyperLayerParameterSet denseHyperParameters)
    {
        _learningRate = config.LearningRate;
        _weightDecay = config.WeightDecay;
        _momentum = config.Momentum;
        _convHyperParameters = convHyperParameters;
        _denseHyperParameters = denseHyperParameters;
    }

    public unsafe void Update(CnnMatrix weights, CnnMatrix biases, NeuralMatrix dW, NeuralMatrix dB)
    {
        int filterCount = dW.Rows;
        int innerDim = dW.UsedColumns;

        var mWeights = _convHyperParameters.MWeights;
        var mBiases = _convHyperParameters.MBiases;

        float lr = _learningRate;
        float wd = _weightDecay;
        float mu = _momentum;

        float* pW = weights.Pointer;
        float* pBiases = biases.Pointer;
        float* pdW = dW.Pointer;
        float* pdB = dB.Pointer;
        float* pVW = mWeights.Pointer;
        float* pVB = mBiases.Pointer;

        int dWStride = dW.ColumnsStride;
        int vStride = mWeights.ColumnsStride;

        for (int f = 0; f < filterCount; f++)
        {
            float* rowW = pW + f * innerDim;
            float* rowDW = pdW + f * dWStride;
            float* rowV = pVW + f * vStride;

            for (int i = 0; i < innerDim; i++)
            {
                float grad = rowDW[i] + wd * rowW[i];
                float vel = mu * rowV[i] - lr * grad;
                rowV[i] = vel;
                rowW[i] += vel;
            }
        }

        for (int fb = 0; fb < filterCount; fb++)
        {
            float grad = pdB[fb];
            float vel = mu * pVB[fb] - lr * grad;
            pVB[fb] = vel;
            pBiases[fb] += vel;
        }
    }

    public unsafe void Update(NeuralMatrix weights, NeuralMatrix biases, NeuralMatrix dW, NeuralMatrix dB)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        int inputSize = dW.Rows;
        int outputSize = dW.UsedColumns;

        var mWeights = _denseHyperParameters.MWeights;
        var mBiases = _denseHyperParameters.MBiases;

        float lr = _learningRate;
        float wd = _weightDecay;
        float mu = _momentum;

        float* pW = weights.Pointer;
        float* pBiases = biases.Pointer;
        float* pdW = dW.Pointer;
        float* pdB = dB.Pointer;
        float* pVW = mWeights.Pointer;
        float* pVB = mBiases.Pointer;

        int wStride = weights.ColumnsStride;
        int dWStride = dW.ColumnsStride;
        int vStride = mWeights.ColumnsStride;

        for (int inIdx = 0; inIdx < inputSize; inIdx++)
        {
            float* rowDW = pdW + inIdx * dWStride;
            float* rowV = pVW + inIdx * vStride;
            float* pWBase = pW + inIdx;

            for (int o = 0; o < outputSize; o++)
            {
                float w = pWBase[o * wStride];
                float grad = rowDW[o] + wd * w;
                float vel = mu * rowV[o] - lr * grad;
                rowV[o] = vel;
                pWBase[o * wStride] = w + vel;
            }
        }

        for (int bi = 0; bi < outputSize; bi++)
        {
            float grad = pdB[bi];
            float vel = mu * pVB[bi] - lr * grad;
            pVB[bi] = vel;
            pBiases[bi] += vel;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _denseHyperParameters?.Dispose();
        _convHyperParameters?.Dispose();
    }
}
