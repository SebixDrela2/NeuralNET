using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public class CnnAdamWOptimizer : ICnnOptimizer, IDisposable
{
    private readonly float _learningRate;
    private readonly float _weightDecay;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;

    private int _t;
    private float _b1_pow = 1.0f;
    private float _b2_pow = 1.0f;

    private readonly OptimizerHyperLayerParameterSet _convHyperParameters;
    private readonly OptimizerHyperLayerParameterSet _denseHyperParameters;

    public CnnAdamWOptimizer(
        CnnOptimizerConfig config,
        OptimizerHyperLayerParameterSet convHyperParameters,
        OptimizerHyperLayerParameterSet denseHyperParameters)
    {
        _learningRate = config.LearningRate;
        _weightDecay = config.WeightDecay;
        _beta1 = config.Beta1;
        _beta2 = config.Beta2;
        _epsilon = config.Epsilon;
        _t = 0;
        _convHyperParameters = convHyperParameters;
        _denseHyperParameters = denseHyperParameters;
    }

    public unsafe void Update(CnnMatrix weights, CnnMatrix biases, NeuralMatrix dW, NeuralMatrix dB)
    {
        int filterCount = dW.Rows;
        int innerDim = dW.UsedColumns;

        var mWeights = _convHyperParameters.MWeights;
        var vWeights = _convHyperParameters.VWeights;
        var mBiases = _convHyperParameters.MBiases;
        var vBiases = _convHyperParameters.VBiases;

        _t++;
        _b1_pow *= _beta1;
        _b2_pow *= _beta2;
        float c_m = 1.0f / (1.0f - _b1_pow);
        float c_v = 1.0f / (1.0f - _b2_pow);
        float one_minus_b1 = 1.0f - _beta1;
        float one_minus_b2 = 1.0f - _beta2;

        float* pW = weights.Pointer;
        float* pBiases = biases.Pointer;
        float* pdW = dW.Pointer;
        float* pdB = dB.Pointer;
        float* pM = mWeights.Pointer;
        float* pV = vWeights.Pointer;
        float* pMBiases = mBiases.Pointer;
        float* pVBiases = vBiases.Pointer;

        int dWStride = dW.ColumnsStride;
        int mStride = mWeights.ColumnsStride;
        int vStride = vWeights.ColumnsStride;

        for (int f = 0; f < filterCount; f++)
        {
            float* rowW = pW + f * innerDim;
            float* rowDW = pdW + f * dWStride;
            float* rowM = pM + f * mStride;
            float* rowV = pV + f * vStride;

            int i = 0;

            if (Avx512F.IsSupported)
            {
                var vB1 = Vector512.Create(_beta1);
                var vOneMinusB1 = Vector512.Create(one_minus_b1);
                var vB2 = Vector512.Create(_beta2);
                var vOneMinusB2 = Vector512.Create(one_minus_b2);
                var vCm = Vector512.Create(c_m);
                var vCv = Vector512.Create(c_v);
                var vLr = Vector512.Create(_learningRate);
                var vWd = Vector512.Create(_weightDecay);
                var vEps = Vector512.Create(_epsilon);

                int vecLimit = innerDim - (innerDim % 16);
                for (; i < vecLimit; i += 16)
                {
                    var vW = Vector512.Load(rowW + i);
                    var vGrad = Vector512.Load(rowDW + i);
                    var vM = Vector512.Load(rowM + i);
                    var vV = Vector512.Load(rowV + i);
                    var vMNew = (vB1 * vM) + (vOneMinusB1 * vGrad);
                    vMNew.Store(rowM + i);

                    var vVNew = (vB2 * vV) + (vOneMinusB2 * (vGrad * vGrad));
                    vVNew.Store(rowV + i);

                    var vMHat = vMNew * vCm;
                    var vVHat = vVNew * vCv;

                    var vDenom = Vector512.Sqrt(vVHat) + vEps;
                    var vStep = (vLr * vMHat) / vDenom;
                    var vWNew = vW - vStep - (vLr * vWd * vW);
                    vWNew.Store(rowW + i);
                }
            }
            else if (Avx2.IsSupported)
            {
                var vB1 = Vector256.Create(_beta1);
                var vOneMinusB1 = Vector256.Create(one_minus_b1);
                var vB2 = Vector256.Create(_beta2);
                var vOneMinusB2 = Vector256.Create(one_minus_b2);
                var vCm = Vector256.Create(c_m);
                var vCv = Vector256.Create(c_v);
                var vLr = Vector256.Create(_learningRate);
                var vWd = Vector256.Create(_weightDecay);
                var vEps = Vector256.Create(_epsilon);

                int vecLimit = innerDim - (innerDim % 8);
                for (; i < vecLimit; i += 8)
                {
                    var vW = Vector256.Load(rowW + i);
                    var vGrad = Vector256.Load(rowDW + i);
                    var vM = Vector256.Load(rowM + i);
                    var vV = Vector256.Load(rowV + i);

                    var vMNew = (vB1 * vM) + (vOneMinusB1 * vGrad);
                    vMNew.Store(rowM + i);

                    var vVNew = (vB2 * vV) + (vOneMinusB2 * (vGrad * vGrad));
                    vVNew.Store(rowV + i);

                    var vMHat = vMNew * vCm;
                    var vVHat = vVNew * vCv;

                    var vDenom = Vector256.Sqrt(vVHat) + vEps;
                    var vStep = (vLr * vMHat) / vDenom;

                    var vWNew = vW - vStep - (vLr * vWd * vW);
                    vWNew.Store(rowW + i);
                }
            }

            for (; i < innerDim; i++)
            {
                float w = rowW[i];
                float grad = rowDW[i];

                float m = _beta1 * rowM[i] + one_minus_b1 * grad;
                rowM[i] = m;

                float v = _beta2 * rowV[i] + one_minus_b2 * grad * grad;
                rowV[i] = v;

                float mHat = m * c_m;
                float vHat = v * c_v;

                float step = _learningRate * mHat / (MathF.Sqrt(vHat) + _epsilon);
                rowW[i] = w - step - _learningRate * _weightDecay * w;
            }
        }

        int fb = 0;
        if (Avx512F.IsSupported)
        {
            var vB1 = Vector512.Create(_beta1);
            var vOneMinusB1 = Vector512.Create(one_minus_b1);
            var vB2 = Vector512.Create(_beta2);
            var vOneMinusB2 = Vector512.Create(one_minus_b2);
            var vCm = Vector512.Create(c_m);
            var vCv = Vector512.Create(c_v);
            var vLr = Vector512.Create(_learningRate);
            var vEps = Vector512.Create(_epsilon);

            int vecLimit = filterCount - (filterCount % 16);
            for (; fb < vecLimit; fb += 16)
            {
                var vGrad = Vector512.Load(pdB + fb);
                var vM = Vector512.Load(pMBiases + fb);
                var vV = Vector512.Load(pVBiases + fb);
                var vB = Vector512.Load(pBiases + fb);

                var vMNew = (vB1 * vM) + (vOneMinusB1 * vGrad);
                vMNew.Store(pMBiases + fb);

                var vVNew = (vB2 * vV) + (vOneMinusB2 * (vGrad * vGrad));
                vVNew.Store(pVBiases + fb);

                var vMHat = vMNew * vCm;
                var vVHat = vVNew * vCv;

                var vDenom = Vector512.Sqrt(vVHat) + vEps;
                var vStep = (vLr * vMHat) / vDenom;

                (vB - vStep).Store(pBiases + fb);
            }
        }
        else if (Avx2.IsSupported)
        {
            var vB1 = Vector256.Create(_beta1);
            var vOneMinusB1 = Vector256.Create(one_minus_b1);
            var vB2 = Vector256.Create(_beta2);
            var vOneMinusB2 = Vector256.Create(one_minus_b2);
            var vCm = Vector256.Create(c_m);
            var vCv = Vector256.Create(c_v);
            var vLr = Vector256.Create(_learningRate);
            var vEps = Vector256.Create(_epsilon);

            int vecLimit = filterCount - (filterCount % 8);
            for (; fb < vecLimit; fb += 8)
            {
                var vGrad = Vector256.Load(pdB + fb);
                var vM = Vector256.Load(pMBiases + fb);
                var vV = Vector256.Load(pVBiases + fb);
                var vB = Vector256.Load(pBiases + fb);

                var vMNew = (vB1 * vM) + (vOneMinusB1 * vGrad);
                vMNew.Store(pMBiases + fb);

                var vVNew = (vB2 * vV) + (vOneMinusB2 * (vGrad * vGrad));
                vVNew.Store(pVBiases + fb);

                var vMHat = vMNew * vCm;
                var vVHat = vVNew * vCv;

                var vDenom = Vector256.Sqrt(vVHat) + vEps;
                var vStep = (vLr * vMHat) / vDenom;

                (vB - vStep).Store(pBiases + fb);
            }
        }

        for (; fb < filterCount; fb++)
        {
            float grad = pdB[fb];
            float m = _beta1 * pMBiases[fb] + one_minus_b1 * grad;
            pMBiases[fb] = m;

            float v = _beta2 * pVBiases[fb] + one_minus_b2 * grad * grad;
            pVBiases[fb] = v;

            float mHat = m * c_m;
            float vHat = v * c_v;

            pBiases[fb] -= _learningRate * mHat / (MathF.Sqrt(vHat) + _epsilon);
        }
    }

    public unsafe void Update(NeuralMatrix weights, NeuralMatrix biases, NeuralMatrix dW, NeuralMatrix dB)
    {
        int inputSize = dW.Rows;
        int outputSize = dW.UsedColumns;

        var mWeights = _denseHyperParameters.MWeights;
        var vWeights = _denseHyperParameters.VWeights;
        var mBiases = _denseHyperParameters.MBiases;
        var vBiases = _denseHyperParameters.VBiases;

        _t++;
        _b1_pow *= _beta1;
        _b2_pow *= _beta2;
        float c_m = 1.0f / (1.0f - _b1_pow);
        float c_v = 1.0f / (1.0f - _b2_pow);
        float one_minus_b1 = 1.0f - _beta1;
        float one_minus_b2 = 1.0f - _beta2;

        float* pW = weights.Pointer;
        float* pBiases = biases.Pointer;
        float* pdW = dW.Pointer;
        float* pdB = dB.Pointer;
        float* pM = mWeights.Pointer;
        float* pV = vWeights.Pointer;
        float* pMBiases = mBiases.Pointer;
        float* pVBiases = vBiases.Pointer;

        int wStride = weights.ColumnsStride;
        int dWStride = dW.ColumnsStride;
        int mStride = mWeights.ColumnsStride;
        int vStride = vWeights.ColumnsStride;

        for (int inIdx = 0; inIdx < inputSize; inIdx++)
        {
            float* rowDW = pdW + inIdx * dWStride;
            float* rowM = pM + inIdx * mStride;
            float* rowV = pV + inIdx * vStride;
            float* pWBase = pW + inIdx;   // stride wStride between outputs

            int o = 0;

            if (Avx512F.IsSupported)
            {
                var vB1 = Vector512.Create(_beta1);
                var vOneMinusB1 = Vector512.Create(one_minus_b1);
                var vB2 = Vector512.Create(_beta2);
                var vOneMinusB2 = Vector512.Create(one_minus_b2);
                var vCm = Vector512.Create(c_m);
                var vCv = Vector512.Create(c_v);
                var vLr = Vector512.Create(_learningRate);
                var vWd = Vector512.Create(_weightDecay);
                var vEps = Vector512.Create(_epsilon);

                int vecLimit = outputSize - (outputSize % 16);
                for (; o < vecLimit; o += 16)
                {
                    var vGrad = Vector512.Load(rowDW + o);
                    var vM = Vector512.Load(rowM + o);
                    var vV = Vector512.Load(rowV + o);
                    var vW = Vector512.Create(
                        pWBase[(o + 0) * wStride], pWBase[(o + 1) * wStride],
                        pWBase[(o + 2) * wStride], pWBase[(o + 3) * wStride],
                        pWBase[(o + 4) * wStride], pWBase[(o + 5) * wStride],
                        pWBase[(o + 6) * wStride], pWBase[(o + 7) * wStride],
                        pWBase[(o + 8) * wStride], pWBase[(o + 9) * wStride],
                        pWBase[(o + 10) * wStride], pWBase[(o + 11) * wStride],
                        pWBase[(o + 12) * wStride], pWBase[(o + 13) * wStride],
                        pWBase[(o + 14) * wStride], pWBase[(o + 15) * wStride]
                    );

                    var vMNew = (vB1 * vM) + (vOneMinusB1 * vGrad);
                    vMNew.Store(rowM + o);

                    var vVNew = (vB2 * vV) + (vOneMinusB2 * (vGrad * vGrad));
                    vVNew.Store(rowV + o);

                    var vMHat = vMNew * vCm;
                    var vVHat = vVNew * vCv;

                    var vDenom = Vector512.Sqrt(vVHat) + vEps;
                    var vStep = (vLr * vMHat) / vDenom;
                    var vWNew = vW - vStep - (vLr * vWd * vW);

                    pWBase[(o + 0) * wStride] = vWNew.GetElement(0);
                    pWBase[(o + 1) * wStride] = vWNew.GetElement(1);
                    pWBase[(o + 2) * wStride] = vWNew.GetElement(2);
                    pWBase[(o + 3) * wStride] = vWNew.GetElement(3);
                    pWBase[(o + 4) * wStride] = vWNew.GetElement(4);
                    pWBase[(o + 5) * wStride] = vWNew.GetElement(5);
                    pWBase[(o + 6) * wStride] = vWNew.GetElement(6);
                    pWBase[(o + 7) * wStride] = vWNew.GetElement(7);
                    pWBase[(o + 8) * wStride] = vWNew.GetElement(8);
                    pWBase[(o + 9) * wStride] = vWNew.GetElement(9);
                    pWBase[(o + 10) * wStride] = vWNew.GetElement(10);
                    pWBase[(o + 11) * wStride] = vWNew.GetElement(11);
                    pWBase[(o + 12) * wStride] = vWNew.GetElement(12);
                    pWBase[(o + 13) * wStride] = vWNew.GetElement(13);
                    pWBase[(o + 14) * wStride] = vWNew.GetElement(14);
                    pWBase[(o + 15) * wStride] = vWNew.GetElement(15);
                }
            }
            else if (Avx2.IsSupported)
            {
                var vB1 = Vector256.Create(_beta1);
                var vOneMinusB1 = Vector256.Create(one_minus_b1);
                var vB2 = Vector256.Create(_beta2);
                var vOneMinusB2 = Vector256.Create(one_minus_b2);
                var vCm = Vector256.Create(c_m);
                var vCv = Vector256.Create(c_v);
                var vLr = Vector256.Create(_learningRate);
                var vWd = Vector256.Create(_weightDecay);
                var vEps = Vector256.Create(_epsilon);

                int vecLimit = outputSize - (outputSize % 8);
                for (; o < vecLimit; o += 8)
                {
                    var vGrad = Vector256.Load(rowDW + o);
                    var vM = Vector256.Load(rowM + o);
                    var vV = Vector256.Load(rowV + o);

                    var vW = Vector256.Create(
                        pWBase[(o + 0) * wStride], pWBase[(o + 1) * wStride],
                        pWBase[(o + 2) * wStride], pWBase[(o + 3) * wStride],
                        pWBase[(o + 4) * wStride], pWBase[(o + 5) * wStride],
                        pWBase[(o + 6) * wStride], pWBase[(o + 7) * wStride]
                    );

                    var vMNew = (vB1 * vM) + (vOneMinusB1 * vGrad);
                    vMNew.Store(rowM + o);

                    var vVNew = (vB2 * vV) + (vOneMinusB2 * (vGrad * vGrad));
                    vVNew.Store(rowV + o);

                    var vMHat = vMNew * vCm;
                    var vVHat = vVNew * vCv;

                    var vDenom = Vector256.Sqrt(vVHat) + vEps;
                    var vStep = (vLr * vMHat) / vDenom;

                    var vWNew = vW - vStep - (vLr * vWd * vW);

                    pWBase[(o + 0) * wStride] = vWNew.GetElement(0);
                    pWBase[(o + 1) * wStride] = vWNew.GetElement(1);
                    pWBase[(o + 2) * wStride] = vWNew.GetElement(2);
                    pWBase[(o + 3) * wStride] = vWNew.GetElement(3);
                    pWBase[(o + 4) * wStride] = vWNew.GetElement(4);
                    pWBase[(o + 5) * wStride] = vWNew.GetElement(5);
                    pWBase[(o + 6) * wStride] = vWNew.GetElement(6);
                    pWBase[(o + 7) * wStride] = vWNew.GetElement(7);
                }
            }

            for (; o < outputSize; o++)
            {
                float w = pWBase[o * wStride];
                float grad = rowDW[o];

                float m = _beta1 * rowM[o] + one_minus_b1 * grad;
                rowM[o] = m;

                float v = _beta2 * rowV[o] + one_minus_b2 * grad * grad;
                rowV[o] = v;

                float mHat = m * c_m;
                float vHat = v * c_v;

                float step = _learningRate * mHat / (MathF.Sqrt(vHat) + _epsilon);
                pWBase[o * wStride] = w - step - _learningRate * _weightDecay * w;
            }
        }

        int bi = 0;
        if (Avx512F.IsSupported)
        {
            var vB1 = Vector512.Create(_beta1);
            var vOneMinusB1 = Vector512.Create(one_minus_b1);
            var vB2 = Vector512.Create(_beta2);
            var vOneMinusB2 = Vector512.Create(one_minus_b2);
            var vCm = Vector512.Create(c_m);
            var vCv = Vector512.Create(c_v);
            var vLr = Vector512.Create(_learningRate);
            var vEps = Vector512.Create(_epsilon);

            int vecLimit = outputSize - (outputSize % 16);
            for (; bi < vecLimit; bi += 16)
            {
                var vGrad = Vector512.Load(pdB + bi);
                var vM = Vector512.Load(pMBiases + bi);
                var vV = Vector512.Load(pVBiases + bi);
                var vB = Vector512.Load(pBiases + bi);

                var vMNew = (vB1 * vM) + (vOneMinusB1 * vGrad);
                vMNew.Store(pMBiases + bi);

                var vVNew = (vB2 * vV) + (vOneMinusB2 * (vGrad * vGrad));
                vVNew.Store(pVBiases + bi);

                var vMHat = vMNew * vCm;
                var vVHat = vVNew * vCv;

                var vDenom = Vector512.Sqrt(vVHat) + vEps;
                var vStep = (vLr * vMHat) / vDenom;

                (vB - vStep).Store(pBiases + bi);
            }
        }
        else if (Avx2.IsSupported)
        {
            var vB1 = Vector256.Create(_beta1);
            var vOneMinusB1 = Vector256.Create(one_minus_b1);
            var vB2 = Vector256.Create(_beta2);
            var vOneMinusB2 = Vector256.Create(one_minus_b2);
            var vCm = Vector256.Create(c_m);
            var vCv = Vector256.Create(c_v);
            var vLr = Vector256.Create(_learningRate);
            var vEps = Vector256.Create(_epsilon);

            int vecLimit = outputSize - (outputSize % 8);
            for (; bi < vecLimit; bi += 8)
            {
                var vGrad = Vector256.Load(pdB + bi);
                var vM = Vector256.Load(pMBiases + bi);
                var vV = Vector256.Load(pVBiases + bi);
                var vB = Vector256.Load(pBiases + bi);

                var vMNew = (vB1 * vM) + (vOneMinusB1 * vGrad);
                vMNew.Store(pMBiases + bi);

                var vVNew = (vB2 * vV) + (vOneMinusB2 * (vGrad * vGrad));
                vVNew.Store(pVBiases + bi);

                var vMHat = vMNew * vCm;
                var vVHat = vVNew * vCv;

                var vDenom = Vector256.Sqrt(vVHat) + vEps;
                var vStep = (vLr * vMHat) / vDenom;

                (vB - vStep).Store(pBiases + bi);
            }
        }

        for (; bi < outputSize; bi++)
        {
            float grad = pdB[bi];
            float m = _beta1 * pMBiases[bi] + one_minus_b1 * grad;
            pMBiases[bi] = m;

            float v = _beta2 * pVBiases[bi] + one_minus_b2 * grad * grad;
            pVBiases[bi] = v;

            float mHat = m * c_m;
            float vHat = v * c_v;

            pBiases[bi] -= _learningRate * mHat / (MathF.Sqrt(vHat) + _epsilon);
        }
    }

    public void Dispose()
    {
        _denseHyperParameters?.Dispose();
        _convHyperParameters?.Dispose();
    }
}
