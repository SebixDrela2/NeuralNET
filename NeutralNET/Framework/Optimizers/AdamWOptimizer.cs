using NeutralNET.Framework.Neural;
using NeutralNET.Matrices;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NeutralNET.Framework.Optimizers;

internal class AdamWOptimizer<TArch>(
    NeuralNetworkConfig config,
    TArch architecture,
    TArch gradientArchitecture) : IOptimizer where TArch : IArchitecture<TArch>
{
    private int _timeStep = 1;

    public void Learn()
    {
        float beta1 = config.Beta1;
        float beta2 = config.Beta2;
        float epsilon = config.Epsilon;
        float lr = config.LearningRate;
        float wd = config.WeightDecay;

        for (var i = 0; i < architecture.Count; i++)
        {
            UpdateAdamMomentsVectorized(
                architecture.MatrixMWeights[i], architecture.MatrixVWeights[i],
                gradientArchitecture.MatrixWeights[i], beta1, beta2);

            UpdateAdamMomentsVectorized(
                architecture.MatrixMBiases[i], architecture.MatrixVBiases[i],
                gradientArchitecture.MatrixBiases[i], beta1, beta2);

            ApplyAdamWUpdateVectorized(
                architecture.MatrixWeights[i],
                architecture.MatrixMWeights[i],
                architecture.MatrixVWeights[i],
                lr, wd, beta1, beta2, epsilon, _timeStep);

            ApplyAdamWUpdateVectorized(
                architecture.MatrixBiases[i],
                architecture.MatrixMBiases[i],
                architecture.MatrixVBiases[i],
                lr, wd, beta1, beta2, epsilon, _timeStep);
        }

        _timeStep++;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void UpdateAdamMomentsVectorized(
        NeuralMatrix mMatrix, NeuralMatrix vMatrix, NeuralMatrix gradient,
        float beta1, float beta2)
    {
        float* mPtr = mMatrix.Pointer;
        float* vPtr = vMatrix.Pointer;
        float* gPtr = gradient.Pointer;
        float* end = mPtr + mMatrix.AllocatedLength;

        var beta1Vec = Vector512.Create(beta1);
        var beta2Vec = Vector512.Create(beta2);
        var oneMinusBeta1 = Vector512.Create(1 - beta1);
        var oneMinusBeta2 = Vector512.Create(1 - beta2);

        if (false)
        {
            for (; mPtr != end;)
            {
                var m = Avx512F.LoadVector512(mPtr);
                var v = Avx512F.LoadVector512(vPtr);
                var g = Avx512F.LoadVector512(gPtr);

                var newM = Avx512F.Add(
                    Avx512F.Multiply(beta1Vec, m),
                    Avx512F.Multiply(oneMinusBeta1, g));

                var gSq = Avx512F.Multiply(g, g);
                var newV = Avx512F.Add(
                    Avx512F.Multiply(beta2Vec, v),
                    Avx512F.Multiply(oneMinusBeta2, gSq));

                newM.Store(mPtr);
                newV.Store(vPtr);

                mPtr += NeuralMatrix.Alignment;
                vPtr += NeuralMatrix.Alignment;
                gPtr += NeuralMatrix.Alignment;
            }
        }
        else if (Avx2.IsSupported)
        {
            // Fallback to AVX2 if AVX512 is not available
            for (; mPtr != end;)
            {
                var m = Vector256.Create(mPtr[0], mPtr[1], mPtr[2], mPtr[3],
                                         mPtr[4], mPtr[5], mPtr[6], mPtr[7]);
                var v = Vector256.Create(vPtr[0], vPtr[1], vPtr[2], vPtr[3],
                                         vPtr[4], vPtr[5], vPtr[6], vPtr[7]);
                var g = Vector256.Create(gPtr[0], gPtr[1], gPtr[2], gPtr[3],
                                         gPtr[4], gPtr[5], gPtr[6], gPtr[7]);

                var beta1Vec256 = Vector256.Create(beta1);
                var beta2Vec256 = Vector256.Create(beta2);
                var oneMinusBeta1Vec256 = Vector256.Create(1 - beta1);
                var oneMinusBeta2Vec256 = Vector256.Create(1 - beta2);

                var newM = Avx2.Add(
                    Avx2.Multiply(beta1Vec256, m),
                    Avx2.Multiply(oneMinusBeta1Vec256, g));

                var gSq = Avx2.Multiply(g, g);
                var newV = Avx2.Add(
                    Avx2.Multiply(beta2Vec256, v),
                    Avx2.Multiply(oneMinusBeta2Vec256, gSq));

                // Store results (AVX2 version - 8 floats at a time)
                for (int j = 0; j < 8; j++)
                {
                    mPtr[j] = newM.GetElement(j);
                    vPtr[j] = newV.GetElement(j);
                }

                mPtr += 8;
                vPtr += 8;
                gPtr += 8;
            }
        }
        else
        {
            // Scalar fallback implementation
            for (float* endScalar = mPtr + mMatrix.AllocatedLength; mPtr != endScalar; mPtr++, vPtr++, gPtr++)
            {
                *mPtr = beta1 * (*mPtr) + (1 - beta1) * (*gPtr);
                *vPtr = beta2 * (*vPtr) + (1 - beta2) * (*gPtr) * (*gPtr);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void ApplyAdamWUpdateVectorized(
        NeuralMatrix param, NeuralMatrix m, NeuralMatrix v,
        float lr, float wd, float beta1, float beta2, float epsilon, int t)
    {
        float* p = param.Pointer;
        float* mPtr = m.Pointer;
        float* vPtr = v.Pointer;
        float* end = p + param.AllocatedLength;

        float beta1T = MathF.Pow(beta1, t);
        float beta2T = MathF.Pow(beta2, t);

        if (false)
        {
            var mCorrVec = Vector512.Create(1 / (1 - beta1T));
            var vCorrVec = Vector512.Create(1 / (1 - beta2T));
            var lrVec = Vector512.Create(lr);
            var epsVec = Vector512.Create(epsilon);
            var lrWdVec = Vector512.Create(lr * wd); // AdamW: decay = lr * wd * param

            for (; p != end;)
            {
                var paramVec = Avx512F.LoadVector512(p);
                var mVec = Avx512F.LoadVector512(mPtr);
                var vVec = Avx512F.LoadVector512(vPtr);

                // Bias correction
                var mHat = Avx512F.Multiply(mVec, mCorrVec);
                var vHat = Avx512F.Multiply(vVec, vCorrVec);

                // Adam update step
                var sqrtV = Avx512F.Sqrt(vHat);
                var denom = Avx512F.Add(sqrtV, epsVec);
                var step = Avx512F.Divide(mHat, denom);
                step = Avx512F.Multiply(lrVec, step);

                // AdamW: Apply Adam update first
                var newParam = Avx512F.Subtract(paramVec, step);

                // AdamW: Then apply decoupled weight decay
                var decay = Avx512F.Multiply(lrWdVec, paramVec);
                newParam = Avx512F.Subtract(newParam, decay);

                newParam.Store(p);

                p += NeuralMatrix.Alignment;
                mPtr += NeuralMatrix.Alignment;
                vPtr += NeuralMatrix.Alignment;
            }
        }
        else if (Avx2.IsSupported)
        {
            // Fallback to AVX2
            var mCorrVec = Vector256.Create(1 / (1 - beta1T));
            var vCorrVec = Vector256.Create(1 / (1 - beta2T));
            var lrVec = Vector256.Create(lr);
            var epsVec = Vector256.Create(epsilon);
            var lrWdVec = Vector256.Create(lr * wd);

            for (; p != end;)
            {
                // Load 8 floats at a time
                var paramVec = Vector256.Create(p[0], p[1], p[2], p[3],
                                                p[4], p[5], p[6], p[7]);
                var mVec = Vector256.Create(mPtr[0], mPtr[1], mPtr[2], mPtr[3],
                                            mPtr[4], mPtr[5], mPtr[6], mPtr[7]);
                var vVec = Vector256.Create(vPtr[0], vPtr[1], vPtr[2], vPtr[3],
                                            vPtr[4], vPtr[5], vPtr[6], vPtr[7]);

                // Bias correction
                var mHat = Avx2.Multiply(mVec, mCorrVec);
                var vHat = Avx2.Multiply(vVec, vCorrVec);

                // Adam update step
                var sqrtV = Avx2.Sqrt(vHat);
                var denom = Avx2.Add(sqrtV, epsVec);
                var step = Avx2.Divide(mHat, denom);
                step = Avx2.Multiply(lrVec, step);

                // AdamW: Apply Adam update first
                var newParam = Avx2.Subtract(paramVec, step);

                // AdamW: Then apply decoupled weight decay
                var decay = Avx2.Multiply(lrWdVec, paramVec);
                newParam = Avx2.Subtract(newParam, decay);

                // Store results
                for (int j = 0; j < 8; j++)
                {
                    p[j] = newParam.GetElement(j);
                }

                p += 8;
                mPtr += 8;
                vPtr += 8;
            }
        }
        else
        {
            // Scalar fallback implementation
            float mCorr = 1 / (1 - beta1T);
            float vCorr = 1 / (1 - beta2T);
            float lrWd = lr * wd;

            for (float* endScalar = p + param.AllocatedLength; p != endScalar; p++, mPtr++, vPtr++)
            {
                float mHat = (*mPtr) * mCorr;
                float vHat = (*vPtr) * vCorr;

                // Adam update
                float step = lr * mHat / (MathF.Sqrt(vHat) + epsilon);

                // AdamW: Apply Adam update then weight decay
                *p = *p - step - lrWd * (*p);
            }
        }
    }
}
