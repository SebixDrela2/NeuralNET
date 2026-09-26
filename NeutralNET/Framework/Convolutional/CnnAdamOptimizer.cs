using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public class CnnAdamOptimizer : ICnnOptimizer, IDisposable
{
    private readonly float _learningRate;
    private readonly float _weightDecay;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;

    private int _t;

    // Conv state
    private readonly AdamHyperLayerParameters _convHyperParameters;
    private readonly AdamHyperLayerParameters _denseHyperParameters;

    public CnnAdamOptimizer(
        CnnOptimizerConfig config,
        AdamHyperLayerParameters convHyperParameters,
        AdamHyperLayerParameters denseHyperParameters)
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
        int innerDim = dW.Rows;
        int filters = dW.UsedColumns;

        var mWeights = _convHyperParameters.MWeights;
        var vWeights = _convHyperParameters.VWeights;
        var mBiases = _convHyperParameters.MBiases;
        var vBiases = _convHyperParameters.VBiases;

        _t++;
        float lr = _learningRate;
        float wd = _weightDecay;
        float b1 = _beta1;
        float b2 = _beta2;
        float eps = _epsilon;
        float t = _t;

        float c_m = 1.0f / (1.0f - MathF.Pow(b1, t));
        float c_v = 1.0f / (1.0f - MathF.Pow(b2, t));
        float one_minus_b1 = 1.0f - b1;
        float one_minus_b2 = 1.0f - b2;

        bool hasAvx512 = Avx512F.IsSupported;
        bool hasAvx2 = Avx2.IsSupported;

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

        // =========================================================================
        // 1. CONVOLUTION WEIGHTS UPDATE
        // =========================================================================
        for (int inner = 0; inner < innerDim; inner++)
        {
            float* rowDW = pdW + inner * dWStride;
            float* rowM = pM + inner * mStride;
            float* rowV = pV + inner * vStride;
            float* pWBase = pW + inner;

            int f = 0;

            if (hasAvx512)
            {
                var vB1 = Vector512.Create(b1);
                var vOneMinusB1 = Vector512.Create(one_minus_b1);
                var vB2 = Vector512.Create(b2);
                var vOneMinusB2 = Vector512.Create(one_minus_b2);
                var vCm = Vector512.Create(c_m);
                var vCv = Vector512.Create(c_v);
                var vLr = Vector512.Create(lr);
                var vWd = Vector512.Create(wd);
                var vEps = Vector512.Create(eps);

                int vecLimit = filters - (filters % 16);
                for (; f < vecLimit; f += 16)
                {
                    var vW = Vector512.Create(
                        pWBase[(f + 0) * innerDim], pWBase[(f + 1) * innerDim],
                        pWBase[(f + 2) * innerDim], pWBase[(f + 3) * innerDim],
                        pWBase[(f + 4) * innerDim], pWBase[(f + 5) * innerDim],
                        pWBase[(f + 6) * innerDim], pWBase[(f + 7) * innerDim],
                        pWBase[(f + 8) * innerDim], pWBase[(f + 9) * innerDim],
                        pWBase[(f + 10) * innerDim], pWBase[(f + 11) * innerDim],
                        pWBase[(f + 12) * innerDim], pWBase[(f + 13) * innerDim],
                        pWBase[(f + 14) * innerDim], pWBase[(f + 15) * innerDim]
                    );

                    var vGrad = Vector512.Load(rowDW + f);
                    var vM = Vector512.Load(rowM + f);
                    var vV = Vector512.Load(rowV + f);

                    vGrad = vGrad + (vWd * vW);

                    var vMNew = (vB1 * vM) + (vOneMinusB1 * vGrad);
                    vMNew.Store(rowM + f);

                    var vVNew = (vB2 * vV) + (vOneMinusB2 * (vGrad * vGrad));
                    vVNew.Store(rowV + f);

                    var vMHat = vMNew * vCm;
                    var vVHat = vVNew * vCv;

                    var vDenom = Vector512.Sqrt(vVHat) + vEps;
                    var vStep = (vLr * vMHat) / vDenom;

                    var vWNew = vW - vStep;

                    pWBase[(f + 0) * innerDim] = vWNew.GetElement(0);
                    pWBase[(f + 1) * innerDim] = vWNew.GetElement(1);
                    pWBase[(f + 2) * innerDim] = vWNew.GetElement(2);
                    pWBase[(f + 3) * innerDim] = vWNew.GetElement(3);
                    pWBase[(f + 4) * innerDim] = vWNew.GetElement(4);
                    pWBase[(f + 5) * innerDim] = vWNew.GetElement(5);
                    pWBase[(f + 6) * innerDim] = vWNew.GetElement(6);
                    pWBase[(f + 7) * innerDim] = vWNew.GetElement(7);
                    pWBase[(f + 8) * innerDim] = vWNew.GetElement(8);
                    pWBase[(f + 9) * innerDim] = vWNew.GetElement(9);
                    pWBase[(f + 10) * innerDim] = vWNew.GetElement(10);
                    pWBase[(f + 11) * innerDim] = vWNew.GetElement(11);
                    pWBase[(f + 12) * innerDim] = vWNew.GetElement(12);
                    pWBase[(f + 13) * innerDim] = vWNew.GetElement(13);
                    pWBase[(f + 14) * innerDim] = vWNew.GetElement(14);
                    pWBase[(f + 15) * innerDim] = vWNew.GetElement(15);
                }
            }
            else if (hasAvx2)
            {
                var vB1 = Vector256.Create(b1);
                var vOneMinusB1 = Vector256.Create(one_minus_b1);
                var vB2 = Vector256.Create(b2);
                var vOneMinusB2 = Vector256.Create(one_minus_b2);
                var vCm = Vector256.Create(c_m);
                var vCv = Vector256.Create(c_v);
                var vLr = Vector256.Create(lr);
                var vWd = Vector256.Create(wd);
                var vEps = Vector256.Create(eps);

                int vecLimit = filters - (filters % 8);
                for (; f < vecLimit; f += 8)
                {
                    var vW = Vector256.Create(
                        pWBase[(f + 0) * innerDim], pWBase[(f + 1) * innerDim],
                        pWBase[(f + 2) * innerDim], pWBase[(f + 3) * innerDim],
                        pWBase[(f + 4) * innerDim], pWBase[(f + 5) * innerDim],
                        pWBase[(f + 6) * innerDim], pWBase[(f + 7) * innerDim]
                    );

                    var vGrad = Vector256.Load(rowDW + f);
                    var vM = Vector256.Load(rowM + f);
                    var vV = Vector256.Load(rowV + f);

                    vGrad = vGrad + (vWd * vW);

                    var vMNew = (vB1 * vM) + (vOneMinusB1 * vGrad);
                    vMNew.Store(rowM + f);

                    var vVNew = (vB2 * vV) + (vOneMinusB2 * (vGrad * vGrad));
                    vVNew.Store(rowV + f);

                    var vMHat = vMNew * vCm;
                    var vVHat = vVNew * vCv;

                    var vDenom = Vector256.Sqrt(vVHat) + vEps;
                    var vStep = (vLr * vMHat) / vDenom;

                    var vWNew = vW - vStep;

                    pWBase[(f + 0) * innerDim] = vWNew.GetElement(0);
                    pWBase[(f + 1) * innerDim] = vWNew.GetElement(1);
                    pWBase[(f + 2) * innerDim] = vWNew.GetElement(2);
                    pWBase[(f + 3) * innerDim] = vWNew.GetElement(3);
                    pWBase[(f + 4) * innerDim] = vWNew.GetElement(4);
                    pWBase[(f + 5) * innerDim] = vWNew.GetElement(5);
                    pWBase[(f + 6) * innerDim] = vWNew.GetElement(6);
                    pWBase[(f + 7) * innerDim] = vWNew.GetElement(7);
                }
            }

            for (; f < filters; f++)
            {
                float w = pWBase[f * innerDim];
                float grad = rowDW[f] + wd * w;

                float m = b1 * rowM[f] + one_minus_b1 * grad;
                rowM[f] = m;

                float v = b2 * rowV[f] + one_minus_b2 * grad * grad;
                rowV[f] = v;

                float mHat = m * c_m;
                float vHat = v * c_v;

                pWBase[f * innerDim] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
            }
        }

        // =========================================================================
        // 2. CONVOLUTION BIASES UPDATE
        // =========================================================================
        int fb = 0;
        if (hasAvx512)
        {
            var vB1 = Vector512.Create(b1);
            var vOneMinusB1 = Vector512.Create(one_minus_b1);
            var vB2 = Vector512.Create(b2);
            var vOneMinusB2 = Vector512.Create(one_minus_b2);
            var vCm = Vector512.Create(c_m);
            var vCv = Vector512.Create(c_v);
            var vLr = Vector512.Create(lr);
            var vEps = Vector512.Create(eps);

            int vecLimit = filters - (filters % 16);
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

                var vBNew = vB - vStep;
                vBNew.Store(pBiases + fb);
            }
        }
        else if (hasAvx2)
        {
            var vB1 = Vector256.Create(b1);
            var vOneMinusB1 = Vector256.Create(one_minus_b1);
            var vB2 = Vector256.Create(b2);
            var vOneMinusB2 = Vector256.Create(one_minus_b2);
            var vCm = Vector256.Create(c_m);
            var vCv = Vector256.Create(c_v);
            var vLr = Vector256.Create(lr);
            var vEps = Vector256.Create(eps);

            int vecLimit = filters - (filters % 8);
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

                var vBNew = vB - vStep;
                vBNew.Store(pBiases + fb);
            }
        }

        for (; fb < filters; fb++)
        {
            float grad = pdB[fb];
            float m = b1 * pMBiases[fb] + one_minus_b1 * grad;
            pMBiases[fb] = m;

            float v = b2 * pVBiases[fb] + one_minus_b2 * grad * grad;
            pVBiases[fb] = v;

            float mHat = m * c_m;
            float vHat = v * c_v;

            pBiases[fb] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
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
        float lr = _learningRate;
        float wd = _weightDecay;
        float b1 = _beta1;
        float b2 = _beta2;
        float eps = _epsilon;
        float t = _t;

        float c_m = 1.0f / (1.0f - MathF.Pow(b1, t));
        float c_v = 1.0f / (1.0f - MathF.Pow(b2, t));
        float one_minus_b1 = 1.0f - b1;
        float one_minus_b2 = 1.0f - b2;

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

        // =========================================================================
        // 1. DENSE WEIGHTS UPDATE
        // =========================================================================
        for (int inIdx = 0; inIdx < inputSize; inIdx++)
        {
            float* rowDW = pdW + inIdx * dWStride;
            float* rowM = pM + inIdx * mStride;
            float* rowV = pV + inIdx * vStride;
            float* pWBase = pW + inIdx;

            int outIdx = 0;

            if (Avx512F.IsSupported)
            {
                var vB1 = Vector512.Create(b1);
                var vOneMinusB1 = Vector512.Create(one_minus_b1);
                var vB2 = Vector512.Create(b2);
                var vOneMinusB2 = Vector512.Create(one_minus_b2);
                var vCm = Vector512.Create(c_m);
                var vCv = Vector512.Create(c_v);
                var vLr = Vector512.Create(lr);
                var vWd = Vector512.Create(wd);
                var vEps = Vector512.Create(eps);

                int vecLimit = outputSize - (outputSize % 16);
                for (; outIdx < vecLimit; outIdx += 16)
                {
                    var vW = Vector512.Create(
                        pWBase[(outIdx + 0) * wStride], pWBase[(outIdx + 1) * wStride],
                        pWBase[(outIdx + 2) * wStride], pWBase[(outIdx + 3) * wStride],
                        pWBase[(outIdx + 4) * wStride], pWBase[(outIdx + 5) * wStride],
                        pWBase[(outIdx + 6) * wStride], pWBase[(outIdx + 7) * wStride],
                        pWBase[(outIdx + 8) * wStride], pWBase[(outIdx + 9) * wStride],
                        pWBase[(outIdx + 10) * wStride], pWBase[(outIdx + 11) * wStride],
                        pWBase[(outIdx + 12) * wStride], pWBase[(outIdx + 13) * wStride],
                        pWBase[(outIdx + 14) * wStride], pWBase[(outIdx + 15) * wStride]
                    );

                    var vGrad = Vector512.Load(rowDW + outIdx);
                    var vM = Vector512.Load(rowM + outIdx);
                    var vV = Vector512.Load(rowV + outIdx);

                    vGrad = vGrad + (vWd * vW);

                    var vMNew = (vB1 * vM) + (vOneMinusB1 * vGrad);
                    vMNew.Store(rowM + outIdx);

                    var vVNew = (vB2 * vV) + (vOneMinusB2 * (vGrad * vGrad));
                    vVNew.Store(rowV + outIdx);

                    var vMHat = vMNew * vCm;
                    var vVHat = vVNew * vCv;

                    var vDenom = Vector512.Sqrt(vVHat) + vEps;
                    var vStep = (vLr * vMHat) / vDenom;

                    var vWNew = vW - vStep;

                    pWBase[(outIdx + 0) * wStride] = vWNew.GetElement(0);
                    pWBase[(outIdx + 1) * wStride] = vWNew.GetElement(1);
                    pWBase[(outIdx + 2) * wStride] = vWNew.GetElement(2);
                    pWBase[(outIdx + 3) * wStride] = vWNew.GetElement(3);
                    pWBase[(outIdx + 4) * wStride] = vWNew.GetElement(4);
                    pWBase[(outIdx + 5) * wStride] = vWNew.GetElement(5);
                    pWBase[(outIdx + 6) * wStride] = vWNew.GetElement(6);
                    pWBase[(outIdx + 7) * wStride] = vWNew.GetElement(7);
                    pWBase[(outIdx + 8) * wStride] = vWNew.GetElement(8);
                    pWBase[(outIdx + 9) * wStride] = vWNew.GetElement(9);
                    pWBase[(outIdx + 10) * wStride] = vWNew.GetElement(10);
                    pWBase[(outIdx + 11) * wStride] = vWNew.GetElement(11);
                    pWBase[(outIdx + 12) * wStride] = vWNew.GetElement(12);
                    pWBase[(outIdx + 13) * wStride] = vWNew.GetElement(13);
                    pWBase[(outIdx + 14) * wStride] = vWNew.GetElement(14);
                    pWBase[(outIdx + 15) * wStride] = vWNew.GetElement(15);
                }
            }
            else if (Avx2.IsSupported)
            {
                var vB1 = Vector256.Create(b1);
                var vOneMinusB1 = Vector256.Create(one_minus_b1);
                var vB2 = Vector256.Create(b2);
                var vOneMinusB2 = Vector256.Create(one_minus_b2);
                var vCm = Vector256.Create(c_m);
                var vCv = Vector256.Create(c_v);
                var vLr = Vector256.Create(lr);
                var vWd = Vector256.Create(wd);
                var vEps = Vector256.Create(eps);

                int vecLimit = outputSize - (outputSize % 8);
                for (; outIdx < vecLimit; outIdx += 8)
                {
                    var vW = Vector256.Create(
                        pWBase[(outIdx + 0) * wStride], pWBase[(outIdx + 1) * wStride],
                        pWBase[(outIdx + 2) * wStride], pWBase[(outIdx + 3) * wStride],
                        pWBase[(outIdx + 4) * wStride], pWBase[(outIdx + 5) * wStride],
                        pWBase[(outIdx + 6) * wStride], pWBase[(outIdx + 7) * wStride]
                    );

                    var vGrad = Vector256.Load(rowDW + outIdx);
                    var vM = Vector256.Load(rowM + outIdx);
                    var vV = Vector256.Load(rowV + outIdx);

                    vGrad = vGrad + (vWd * vW);

                    var vMNew = (vB1 * vM) + (vOneMinusB1 * vGrad);
                    vMNew.Store(rowM + outIdx);

                    var vVNew = (vB2 * vV) + (vOneMinusB2 * (vGrad * vGrad));
                    vVNew.Store(rowV + outIdx);

                    var vMHat = vMNew * vCm;
                    var vVHat = vVNew * vCv;

                    var vDenom = Vector256.Sqrt(vVHat) + vEps;
                    var vStep = (vLr * vMHat) / vDenom;

                    var vWNew = vW - vStep;

                    pWBase[(outIdx + 0) * wStride] = vWNew.GetElement(0);
                    pWBase[(outIdx + 1) * wStride] = vWNew.GetElement(1);
                    pWBase[(outIdx + 2) * wStride] = vWNew.GetElement(2);
                    pWBase[(outIdx + 3) * wStride] = vWNew.GetElement(3);
                    pWBase[(outIdx + 4) * wStride] = vWNew.GetElement(4);
                    pWBase[(outIdx + 5) * wStride] = vWNew.GetElement(5);
                    pWBase[(outIdx + 6) * wStride] = vWNew.GetElement(6);
                    pWBase[(outIdx + 7) * wStride] = vWNew.GetElement(7);
                }
            }

            for (; outIdx < outputSize; outIdx++)
            {
                float w = pWBase[outIdx * wStride];
                float grad = rowDW[outIdx] + wd * w;

                float m = b1 * rowM[outIdx] + one_minus_b1 * grad;
                rowM[outIdx] = m;

                float v = b2 * rowV[outIdx] + one_minus_b2 * grad * grad;
                rowV[outIdx] = v;

                float mHat = m * c_m;
                float vHat = v * c_v;

                pWBase[outIdx * wStride] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
            }
        }

        // =========================================================================
        // 2. DENSE BIASES UPDATE
        // =========================================================================
        int i = 0;
        if (Avx512F.IsSupported)
        {
            var vB1 = Vector512.Create(b1);
            var vOneMinusB1 = Vector512.Create(one_minus_b1);
            var vB2 = Vector512.Create(b2);
            var vOneMinusB2 = Vector512.Create(one_minus_b2);
            var vCm = Vector512.Create(c_m);
            var vCv = Vector512.Create(c_v);
            var vLr = Vector512.Create(lr);
            var vEps = Vector512.Create(eps);

            int vecLimit = outputSize - (outputSize % 16);
            for (; i < vecLimit; i += 16)
            {
                var vGrad = Vector512.Load(pdB + i);
                var vM = Vector512.Load(pMBiases + i);
                var vV = Vector512.Load(pVBiases + i);
                var vB = Vector512.Load(pBiases + i);

                var vMNew = (vB1 * vM) + (vOneMinusB1 * vGrad);
                vMNew.Store(pMBiases + i);

                var vVNew = (vB2 * vV) + (vOneMinusB2 * (vGrad * vGrad));
                vVNew.Store(pVBiases + i);

                var vMHat = vMNew * vCm;
                var vVHat = vVNew * vCv;

                var vDenom = Vector512.Sqrt(vVHat) + vEps;
                var vStep = (vLr * vMHat) / vDenom;

                var vBNew = vB - vStep;
                vBNew.Store(pBiases + i);
            }
        }
        else if (Avx2.IsSupported)
        {
            var vB1 = Vector256.Create(b1);
            var vOneMinusB1 = Vector256.Create(one_minus_b1);
            var vB2 = Vector256.Create(b2);
            var vOneMinusB2 = Vector256.Create(one_minus_b2);
            var vCm = Vector256.Create(c_m);
            var vCv = Vector256.Create(c_v);
            var vLr = Vector256.Create(lr);
            var vEps = Vector256.Create(eps);

            int vecLimit = outputSize - (outputSize % 8);
            for (; i < vecLimit; i += 8)
            {
                var vGrad = Vector256.Load(pdB + i);
                var vM = Vector256.Load(pMBiases + i);
                var vV = Vector256.Load(pVBiases + i);
                var vB = Vector256.Load(pBiases + i);

                var vMNew = (vB1 * vM) + (vOneMinusB1 * vGrad);
                vMNew.Store(pMBiases + i);

                var vVNew = (vB2 * vV) + (vOneMinusB2 * (vGrad * vGrad));
                vVNew.Store(pVBiases + i);

                var vMHat = vMNew * vCm;
                var vVHat = vVNew * vCv;

                var vDenom = Vector256.Sqrt(vVHat) + vEps;
                var vStep = (vLr * vMHat) / vDenom;

                var vBNew = vB - vStep;
                vBNew.Store(pBiases + i);
            }
        }

        for (; i < outputSize; i++)
        {
            float grad = pdB[i];
            float m = b1 * pMBiases[i] + one_minus_b1 * grad;
            pMBiases[i] = m;

            float v = b2 * pVBiases[i] + one_minus_b2 * grad * grad;
            pVBiases[i] = v;

            float mHat = m * c_m;
            float vHat = v * c_v;

            pBiases[i] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
        }
    }

    public void Dispose()
    {
        _denseHyperParameters?.Dispose();
        _convHyperParameters?.Dispose();
    }
}
