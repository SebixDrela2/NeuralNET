using NeutralNET.Framework.Convolutional;

namespace NeutralNET.Framework.Neural.CNN;


using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

public unsafe class VectorizedBatchNorm
{
    public static void BatchNormForward(CnnMatrix input, BatchNormParams bn)
    {
        int batch = input.Batch;
        int channels = input.Channels;
        int spatial = input.Height * input.Width;
        int batchStride = channels * spatial;
        int N = batch * spatial;
        float invN = 1.0f / N;

        float* pIn = input.Pointer;
        float* pOut = bn.Output.Pointer;
        float* pNorm = bn.Normalized.Pointer;
        float* pMean = bn.Mean.Pointer;
        float* pInvStd = bn.InvStd.Pointer;
        float* pGamma = bn.Gamma.Pointer;
        float* pBeta = bn.Beta.Pointer;
        float* pRunMean = bn.RunningMean.Pointer;
        float* pRunVar = bn.RunningVar.Pointer;

        float momentum = bn.Momentum;
        float eps = bn.Epsilon;

        for (int c = 0; c < channels; c++)
        {
            // ----------------------------------------------------
            // 1. Mean Computation
            // ----------------------------------------------------
            double sum = 0;
            for (int b = 0; b < batch; b++)
            {
                float* pBatch = pIn + b * batchStride + c * spatial;
                sum += SumFloatArray(pBatch, spatial);
            }
            float mean = (float)(sum * invN);

            // ----------------------------------------------------
            // 2. Variance Computation
            // ----------------------------------------------------
            double varSum = 0;
            for (int b = 0; b < batch; b++)
            {
                float* pBatch = pIn + b * batchStride + c * spatial;
                varSum += VarianceSumFloatArray(pBatch, spatial, mean);
            }
            float var = (float)(varSum * invN);
            float invStd = 1.0f / MathF.Sqrt(var + eps);

            pMean[c] = mean;
            pInvStd[c] = invStd;

            float gamma = pGamma[c];
            float beta = pBeta[c];

            // ----------------------------------------------------
            // 3. Normalization and Output Generation
            // ----------------------------------------------------
            for (int b = 0; b < batch; b++)
            {
                float* pInBatch = pIn + b * batchStride + c * spatial;
                float* pOutBatch = pOut + b * batchStride + c * spatial;
                float* pNormBatch = pNorm + b * batchStride + c * spatial;

                int i = 0;

                if (Avx512F.IsSupported)
                {
                    Vector512<float> vMean = Vector512.Create(mean);
                    Vector512<float> vInvStd = Vector512.Create(invStd);
                    Vector512<float> vGamma = Vector512.Create(gamma);
                    Vector512<float> vBeta = Vector512.Create(beta);

                    for (; i <= spatial - 16; i += 16)
                    {
                        Vector512<float> vx = Avx512F.LoadVector512(pInBatch + i);
                        Vector512<float> vxhat = Avx512F.Multiply(Avx512F.Subtract(vx, vMean), vInvStd);
                        Avx512F.Store(pNormBatch + i, vxhat);

                        // vOut = vxhat * vGamma + vBeta
                        Vector512<float> vout = Avx512F.FusedMultiplyAdd(vxhat, vGamma, vBeta);
                        Avx512F.Store(pOutBatch + i, vout);
                    }
                }
                else if (Avx2.IsSupported)
                {
                    Vector256<float> vMean = Vector256.Create(mean);
                    Vector256<float> vInvStd = Vector256.Create(invStd);
                    Vector256<float> vGamma = Vector256.Create(gamma);
                    Vector256<float> vBeta = Vector256.Create(beta);

                    for (; i <= spatial - 8; i += 8)
                    {
                        Vector256<float> vx = Avx2.LoadVector256(pInBatch + i);
                        Vector256<float> vxhat = Avx2.Multiply(Avx2.Subtract(vx, vMean), vInvStd);
                        Avx2.Store(pNormBatch + i, vxhat);

                        // vOut = vxhat * vGamma + vBeta
                        Vector256<float> vout = Fma.IsSupported
                            ? Fma.MultiplyAdd(vxhat, vGamma, vBeta)
                            : Avx2.Add(Avx2.Multiply(vxhat, vGamma), vBeta);

                        Avx2.Store(pOutBatch + i, vout);
                    }
                }

                // Tail Scalar
                for (; i < spatial; i++)
                {
                    float xhat = (pInBatch[i] - mean) * invStd;
                    pNormBatch[i] = xhat;
                    pOutBatch[i] = gamma * xhat + beta;
                }
            }

            // ----------------------------------------------------
            // 4. Update Running Mean and Variance
            // ----------------------------------------------------
            float unbias = N > 1 ? N / (N - 1.0f) : 1.0f;
            pRunMean[c] = (1 - momentum) * pRunMean[c] + momentum * mean;
            pRunVar[c] = (1 - momentum) * pRunVar[c] + momentum * var * unbias;
        }
    }

    public static void BatchNormInference(CnnMatrix input, BatchNormParams bn)
    {
        int batch = input.Batch;
        int channels = input.Channels;
        int spatial = input.Height * input.Width;
        int batchStride = channels * spatial;

        float* pIn = input.Pointer;
        float* pOut = bn.Output.Pointer;
        float* pRunMean = bn.RunningMean.Pointer;
        float* pRunVar = bn.RunningVar.Pointer;
        float* pGamma = bn.Gamma.Pointer;
        float* pBeta = bn.Beta.Pointer;
        float eps = bn.Epsilon;

        for (int c = 0; c < channels; c++)
        {
            // Simplify equation: out = gamma * (x - mean) * invStd + beta
            // out = x * (gamma * invStd) + (beta - mean * gamma * invStd)
            // out = x * scale + offset
            float invStd = 1.0f / MathF.Sqrt(pRunVar[c] + eps);
            float scale = pGamma[c] * invStd;
            float offset = pBeta[c] - pRunMean[c] * scale;

            for (int b = 0; b < batch; b++)
            {
                float* pInBatch = pIn + b * batchStride + c * spatial;
                float* pOutBatch = pOut + b * batchStride + c * spatial;

                int i = 0;

                if (Avx512F.IsSupported)
                {
                    Vector512<float> vScale = Vector512.Create(scale);
                    Vector512<float> vOffset = Vector512.Create(offset);

                    for (; i <= spatial - 16; i += 16)
                    {
                        Vector512<float> vx = Avx512F.LoadVector512(pInBatch + i);
                        Vector512<float> vout = Avx512F.FusedMultiplyAdd(vx, vScale, vOffset);
                        Avx512F.Store(pOutBatch + i, vout);
                    }
                }
                else if (Avx2.IsSupported)
                {
                    Vector256<float> vScale = Vector256.Create(scale);
                    Vector256<float> vOffset = Vector256.Create(offset);

                    for (; i <= spatial - 8; i += 8)
                    {
                        Vector256<float> vx = Avx2.LoadVector256(pInBatch + i);
                        Vector256<float> vout = Fma.IsSupported
                            ? Fma.MultiplyAdd(vx, vScale, vOffset)
                            : Avx2.Add(Avx2.Multiply(vx, vScale), vOffset);

                        Avx2.Store(pOutBatch + i, vout);
                    }
                }

                // Tail Scalar
                for (; i < spatial; i++)
                {
                    pOutBatch[i] = pInBatch[i] * scale + offset;
                }
            }
        }
    }

    public static void BatchNormBackward(CnnMatrix gradOutput, BatchNormParams bn)
    {
        int batch = gradOutput.Batch;
        int channels = gradOutput.Channels;
        int spatial = gradOutput.Height * gradOutput.Width;
        int batchStride = channels * spatial;
        int N = batch * spatial;
        float invN = 1.0f / N;

        float* pGradOut = gradOutput.Pointer;
        float* pNorm = bn.Normalized.Pointer;
        float* pInvStd = bn.InvStd.Pointer;
        float* pGamma = bn.Gamma.Pointer;
        float* pGradIn = bn.GradInput.Pointer;
        float* pGradGamma = bn.GradGamma.Pointer;
        float* pGradBeta = bn.GradBeta.Pointer;

        for (int c = 0; c < channels; c++)
        {
            float gamma = pGamma[c];
            float invStd = pInvStd[c];

            double sumDG = 0;
            double sumDGNorm = 0;

            // 1. Accumulate Sum(dOut) and Sum(dOut * xhat)
            for (int b = 0; b < batch; b++)
            {
                float* pOutBatch = pGradOut + b * batchStride + c * spatial;
                float* pNormBatch = pNorm + b * batchStride + c * spatial;

                AccumulateBackwardSums(pOutBatch, pNormBatch, spatial, ref sumDG, ref sumDGNorm);
            }

            pGradGamma[c] = (float)sumDGNorm;
            pGradBeta[c] = (float)sumDG;

            float meanDG = (float)(sumDG * invN);
            float meanDGNorm = (float)(sumDGNorm * invN);
            float scaleFactor = gamma * invStd;

            // 2. Compute Input Gradient
            for (int b = 0; b < batch; b++)
            {
                float* pOutBatch = pGradOut + b * batchStride + c * spatial;
                float* pNormBatch = pNorm + b * batchStride + c * spatial;
                float* pGradInBatch = pGradIn + b * batchStride + c * spatial;

                int i = 0;

                if (Avx512F.IsSupported)
                {
                    Vector512<float> vScaleFactor = Vector512.Create(scaleFactor);
                    Vector512<float> vMeanDG = Vector512.Create(meanDG);
                    Vector512<float> vMeanDGNorm = Vector512.Create(meanDGNorm);

                    for (; i <= spatial - 16; i += 16)
                    {
                        Vector512<float> vdout = Avx512F.LoadVector512(pOutBatch + i);
                        Vector512<float> vxhat = Avx512F.LoadVector512(pNormBatch + i);

                        // term = (dout - meanDG - xhat * meanDGNorm)
                        Vector512<float> vTerm = Avx512F.Subtract(
                            Avx512F.Subtract(vdout, vMeanDG),
                            Avx512F.Multiply(vxhat, vMeanDGNorm)
                        );

                        Vector512<float> vGradIn = Avx512F.Multiply(vTerm, vScaleFactor);
                        Avx512F.Store(pGradInBatch + i, vGradIn);
                    }
                }
                else if (Avx2.IsSupported)
                {
                    Vector256<float> vScaleFactor = Vector256.Create(scaleFactor);
                    Vector256<float> vMeanDG = Vector256.Create(meanDG);
                    Vector256<float> vMeanDGNorm = Vector256.Create(meanDGNorm);

                    for (; i <= spatial - 8; i += 8)
                    {
                        Vector256<float> vdout = Avx2.LoadVector256(pOutBatch + i);
                        Vector256<float> vxhat = Avx2.LoadVector256(pNormBatch + i);

                        Vector256<float> vTerm = Avx2.Subtract(
                            Avx2.Subtract(vdout, vMeanDG),
                            Avx2.Multiply(vxhat, vMeanDGNorm)
                        );

                        Vector256<float> vGradIn = Avx2.Multiply(vTerm, vScaleFactor);
                        Avx2.Store(pGradInBatch + i, vGradIn);
                    }
                }

                // Tail Scalar
                for (; i < spatial; i++)
                {
                    float dout = pOutBatch[i];
                    float xhat = pNormBatch[i];
                    pGradInBatch[i] = scaleFactor * (dout - meanDG - xhat * meanDGNorm);
                }
            }
        }
    }

    #region Helper Vector Reductions

    private static double SumFloatArray(float* ptr, int count)
    {
        int i = 0;
        double sum = 0;

        if (Avx512F.IsSupported)
        {
            Vector512<float> vAcc = Vector512<float>.Zero;
            for (; i <= count - 16; i += 16)
            {
                vAcc = Avx512F.Add(vAcc, Avx512F.LoadVector512(ptr + i));
            }
            sum += HorizontalSum(vAcc);
        }
        else if (Avx2.IsSupported)
        {
            Vector256<float> vAcc = Vector256<float>.Zero;
            for (; i <= count - 8; i += 8)
            {
                vAcc = Avx2.Add(vAcc, Avx2.LoadVector256(ptr + i));
            }
            sum += HorizontalSum(vAcc);
        }

        for (; i < count; i++)
        {
            sum += ptr[i];
        }

        return sum;
    }

    private static double VarianceSumFloatArray(float* ptr, int count, float mean)
    {
        int i = 0;
        double varSum = 0;

        if (Avx512F.IsSupported)
        {
            Vector512<float> vMean = Vector512.Create(mean);
            Vector512<float> vAcc = Vector512<float>.Zero;

            for (; i <= count - 16; i += 16)
            {
                Vector512<float> vD = Avx512F.Subtract(Avx512F.LoadVector512(ptr + i), vMean);
                vAcc = Avx512F.FusedMultiplyAdd(vD, vD, vAcc);
            }
            varSum += HorizontalSum(vAcc);
        }
        else if (Avx2.IsSupported)
        {
            Vector256<float> vMean = Vector256.Create(mean);
            Vector256<float> vAcc = Vector256<float>.Zero;

            for (; i <= count - 8; i += 8)
            {
                Vector256<float> vD = Avx2.Subtract(Avx2.LoadVector256(ptr + i), vMean);
                vAcc = Fma.IsSupported
                    ? Fma.MultiplyAdd(vD, vD, vAcc)
                    : Avx2.Add(vAcc, Avx2.Multiply(vD, vD));
            }
            varSum += HorizontalSum(vAcc);
        }

        for (; i < count; i++)
        {
            float d = ptr[i] - mean;
            varSum += d * d;
        }

        return varSum;
    }

    private static void AccumulateBackwardSums(float* pOutBatch, float* pNormBatch, int count, ref double sumDG, ref double sumDGNorm)
    {
        int i = 0;

        if (Avx512F.IsSupported)
        {
            Vector512<float> vSumDG = Vector512<float>.Zero;
            Vector512<float> vSumDGNorm = Vector512<float>.Zero;

            for (; i <= count - 16; i += 16)
            {
                Vector512<float> vdout = Avx512F.LoadVector512(pOutBatch + i);
                Vector512<float> vxhat = Avx512F.LoadVector512(pNormBatch + i);

                vSumDG = Avx512F.Add(vSumDG, vdout);
                vSumDGNorm = Avx512F.FusedMultiplyAdd(vdout, vxhat, vSumDGNorm);
            }

            sumDG += HorizontalSum(vSumDG);
            sumDGNorm += HorizontalSum(vSumDGNorm);
        }
        else if (Avx2.IsSupported)
        {
            Vector256<float> vSumDG = Vector256<float>.Zero;
            Vector256<float> vSumDGNorm = Vector256<float>.Zero;

            for (; i <= count - 8; i += 8)
            {
                Vector256<float> vdout = Avx2.LoadVector256(pOutBatch + i);
                Vector256<float> vxhat = Avx2.LoadVector256(pNormBatch + i);

                vSumDG = Avx2.Add(vSumDG, vdout);
                vSumDGNorm = Fma.IsSupported
                    ? Fma.MultiplyAdd(vdout, vxhat, vSumDGNorm)
                    : Avx2.Add(vSumDGNorm, Avx2.Multiply(vdout, vxhat));
            }

            sumDG += HorizontalSum(vSumDG);
            sumDGNorm += HorizontalSum(vSumDGNorm);
        }

        for (; i < count; i++)
        {
            float dout = pOutBatch[i];
            float xhat = pNormBatch[i];
            sumDG += dout;
            sumDGNorm += dout * xhat;
        }
    }

    private static float HorizontalSum(Vector512<float> v)
    {
        Vector256<float> low = v.GetLower();
        Vector256<float> high = v.GetUpper();
        return HorizontalSum(Avx2.Add(low, high));
    }

    private static float HorizontalSum(Vector256<float> v)
    {
        // 256-bit to 128-bit reduction
        Vector128<float> low = v.GetLower();
        Vector128<float> high = v.GetUpper();
        Vector128<float> v128 = Sse.Add(low, high);

        // Horizontal add inside 128-bit vector
        Vector128<float> shuf = Sse.Shuffle(v128, v128, 0b_01_00_11_10);
        Vector128<float> sums = Sse.Add(v128, shuf);
        shuf = Sse.Shuffle(sums, sums, 0b_10_11_00_01);
        sums = Sse.Add(sums, shuf);

        return sums.ToScalar();
    }

    #endregion
}
