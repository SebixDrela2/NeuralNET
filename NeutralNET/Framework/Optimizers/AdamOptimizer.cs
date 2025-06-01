using NeutralNET.Framework.Neural;
using NeutralNET.Matrices;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NeutralNET.Framework.Optimizers;

internal class AdamOptimizer<TArch>(
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

            ApplyAdamUpdateVectorized(
                architecture.MatrixWeights[i],
                architecture.MatrixMWeights[i],
                architecture.MatrixVWeights[i],
                lr, wd, beta1, beta2, epsilon, _timeStep);

            ApplyAdamUpdateVectorized(
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

        var beta1Vec = Vector256.Create(beta1);
        var beta2Vec = Vector256.Create(beta2);
        var oneMinusBeta1 = Vector256.Create(1 - beta1);
        var oneMinusBeta2 = Vector256.Create(1 - beta2);

        if (Avx2.IsSupported)
        {
            for (; mPtr != end;)
            {
                var m = Avx.LoadVector256(mPtr);
                var v = Avx.LoadVector256(vPtr);
                var g = Avx.LoadVector256(gPtr);

                var newM = Avx.Add(Avx.Multiply(beta1Vec, m),
                              Avx.Multiply(oneMinusBeta1, g));

                var gSq = Avx.Multiply(g, g);
                var newV = Avx.Add(Avx.Multiply(beta2Vec, v),
                              Avx.Multiply(oneMinusBeta2, gSq));

                newM.Store(mPtr);
                newV.Store(vPtr);

                mPtr += 8; vPtr += 8; gPtr += 8;
            }
        }
        else
        {
            // Scalar implementation
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void ApplyAdamUpdateVectorized(
        NeuralMatrix param, NeuralMatrix m, NeuralMatrix v,
        float lr, float wd, float beta1, float beta2, float epsilon, int t)
    {
        float* p = param.Pointer;
        float* mPtr = m.Pointer;
        float* vPtr = v.Pointer;
        float* end = p + param.AllocatedLength;

        float beta1T = MathF.Pow(beta1, t);
        float beta2T = MathF.Pow(beta2, t);

        var mCorrVec = Vector256.Create(1 / (1 - beta1T));
        var vCorrVec = Vector256.Create(1 / (1 - beta2T));
        var lrVec = Vector256.Create(lr);
        var epsVec = Vector256.Create(epsilon);
        var wdVec = Vector256.Create(lr * wd);

        if (Avx2.IsSupported)
        {
            for (; p != end;)
            {
                var paramVec = Avx.LoadVector256(p);
                var mVec = Avx.LoadVector256(mPtr);
                var vVec = Avx.LoadVector256(vPtr);

                var mHat = Avx.Multiply(mVec, mCorrVec);
                var vHat = Avx.Multiply(vVec, vCorrVec);

                var sqrtV = Avx.Sqrt(vHat);
                var denom = Avx.Add(sqrtV, epsVec);
                var step = Avx.Divide(mHat, denom);
                step = Avx.Multiply(lrVec, step);

                var decay = Avx.Multiply(wdVec, paramVec);
                var newParam = Avx.Subtract(Avx.Subtract(paramVec, decay), step);

                newParam.Store(p);

                p += 8; mPtr += 8; vPtr += 8;
            }
        }
    }
}
