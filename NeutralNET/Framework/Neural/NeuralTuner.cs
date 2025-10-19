using NeutralNET.Matrices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NeutralNET.Framework.Neural;

internal unsafe class NeuralTuner(Random rng)
{
    public void ApplyBatchNormForward(NeuralMatrix input, BatchNormLayer bn)
    {
        int cols = input.UsedColumns;
        int stride = input.ColumnsStride;
        float* inPtr = input.Pointer;
        int rows = input.Rows;

        int vecWidth = NeuralMatrix.Alignment;

        int colBlockCount = cols / vecWidth;
        int colRemainder = cols % vecWidth;

        Vector512<float> momentumVec = Vector512.Create(bn.Momentum);
        Vector512<float> oneMinusMomentumVec = Vector512.Create(1f - bn.Momentum);
        Vector512<float> epsilonVec = Vector512.Create(bn.Epsilon);

        for (int block = 0; block < colBlockCount; block++)
        {
            int j = block * vecWidth;

            Vector512<float> sumVec = Vector512<float>.Zero;

            for (int i = 0; i < rows; i++)
            {
                float* ptr = inPtr + i * stride + j;
                Vector512<float> rowVec = Avx512F.LoadAlignedVector512(ptr);
                sumVec = Avx512F.Add(sumVec, rowVec);
            }

            Vector512<float> meanVec = Avx512F.Divide(sumVec, Vector512.Create((float)rows));
            Vector512<float> varSumVec = Vector512<float>.Zero;

            for (int i = 0; i < rows; i++)
            {
                float* ptr = inPtr + i * stride + j;
                Vector512<float> rowVec = Avx512F.LoadAlignedVector512(ptr);
                Vector512<float> diff = Avx512F.Subtract(rowVec, meanVec);
                Vector512<float> sq = Avx512F.Multiply(diff, diff);
                varSumVec = Avx512F.Add(varSumVec, sq);
            }

            Vector512<float> varVec = Avx512F.Divide(varSumVec, Vector512.Create((float)rows));
            Vector512<float> stdVec = Avx512F.Sqrt(Avx512F.Add(varVec, epsilonVec));

            float* runningMeanPtr = &bn.RunningMean.Pointer[j];
            float* runningVarPtr = &bn.RunningVar.Pointer[j];
            float* gammaPtr = &bn.Gamma.Pointer[j];
            float* betaPtr = &bn.Beta.Pointer[j];

            Vector512<float> runningMeanVec = Avx512F.LoadAlignedVector512(runningMeanPtr);
            Vector512<float> runningVarVec = Avx512F.LoadAlignedVector512(runningVarPtr);
            Vector512<float> gammaVec = Avx512F.LoadAlignedVector512(gammaPtr);
            Vector512<float> betaVec = Avx512F.LoadAlignedVector512(betaPtr);

            runningMeanVec = Avx512F.Add(Avx512F.Multiply(momentumVec, meanVec), Avx512F.Multiply(oneMinusMomentumVec, runningMeanVec));
            runningVarVec = Avx512F.Add(Avx512F.Multiply(momentumVec, varVec), Avx512F.Multiply(oneMinusMomentumVec, runningVarVec));

            Avx512F.Store(runningMeanPtr, runningMeanVec);
            Avx512F.Store(runningVarPtr, runningVarVec);

            for (int i = 0; i < rows; i++)
            {
                float* ptr = inPtr + i * stride + j;
                Vector512<float> valVec = Avx512F.LoadAlignedVector512(ptr);
                Vector512<float> normVec = Avx512F.Divide(Avx512F.Subtract(valVec, meanVec), stdVec);
                Vector512<float> scaledVec = Avx512F.Add(Avx512F.Multiply(normVec, gammaVec), betaVec);
                Avx512F.StoreAligned(ptr, scaledVec);
            }
        }


        if (colRemainder > 0)
        {
            int start = colBlockCount * vecWidth;
            for (int j = start; j < cols; j++)
            {
                float mean = 0f, var = 0f;

                for (int i = 0; i < rows; i++)
                    mean += inPtr[i * stride + j];
                mean /= rows;

                for (int i = 0; i < rows; i++)
                {
                    float val = inPtr[i * stride + j] - mean;
                    var += val * val;
                }
                var /= rows;
                float std = MathF.Sqrt(var + bn.Epsilon);

                ref float runningMean = ref bn.RunningMean.At(0, j);
                ref float runningVar = ref bn.RunningVar.At(0, j);
                ref float gamma = ref bn.Gamma.At(0, j);
                ref float beta = ref bn.Beta.At(0, j);

                runningMean = bn.Momentum * mean + (1f - bn.Momentum) * runningMean;
                runningVar = bn.Momentum * var + (1f - bn.Momentum) * runningVar;

                for (int i = 0; i < rows; i++)
                {
                    int idx = i * stride + j;
                    float norm = (inPtr[idx] - mean) / std;
                    inPtr[idx] = norm * gamma + beta;
                }
            }
        }
    }


    public void ApplyDropout(NeuralMatrix activations, float dropoutRate)
    {
        int total = activations.AllocatedLength;
        float keepProb = 1f - dropoutRate;

        float scale = 1f / keepProb;
        int i = 0;

        if (Avx2.IsSupported)
        {
            int vectorSize = NeuralMatrix.Alignment;
            var scaleVec = Vector512.Create(scale);
            var dropoutRateVec = Vector512.Create(dropoutRate);

            for (; i <= total - vectorSize; i += vectorSize)
            {
                var vec = Avx512F.LoadVector512(activations.Pointer + i);
                Vector512<float> randVec = GenerateRandomVector512();

                var mask = Avx512F.CompareGreaterThan(randVec, dropoutRateVec);
                var scaledVec = Avx512F.Multiply(vec, scaleVec);

                var result = Avx512F.BlendVariable(scaledVec, Vector512<float>.Zero, mask);
                vec.StoreAligned(activations.Pointer + i);
            }
        }

        for (; i < total; i++)
        {
            float rand = (float)rng.NextDouble();
            activations.Pointer[i] = rand < dropoutRate ? 0f : activations.Pointer[i] * scale;
        }
    }

    private unsafe Vector512<float> GenerateRandomVector512()
    {
        float* randArray = stackalloc float[NeuralMatrix.Alignment];

        for (int i = 0; i < NeuralMatrix.Alignment; i++)
        {
            randArray[i] = (float)rng.NextDouble();
        }

        return Avx512F.LoadVector512(randArray);
    }
}
