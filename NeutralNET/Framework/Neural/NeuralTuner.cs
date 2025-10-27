using NeutralNET.Matrices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NeutralNET.Framework.Neural;

public unsafe class NeuralTuner(Random rng)
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

        Vector256<float> momentumVec = Vector256.Create(BatchNormLayer.Momentum);
        Vector256<float> oneMinusMomentumVec = Vector256.Create(1f - BatchNormLayer.Momentum);
        Vector256<float> epsilonVec = Vector256.Create(BatchNormLayer.Epsilon);

        for (int block = 0; block < colBlockCount; block++)
        {
            int j = block * vecWidth;

            Vector256<float> sumVec = Vector256<float>.Zero;

            for (int i = 0; i < rows; i++)
            {
                float* ptr = inPtr + i * stride + j;
                Vector256<float> rowVec = Avx.LoadAlignedVector256(ptr);
                sumVec = Avx.Add(sumVec, rowVec);
            }

            Vector256<float> meanVec = Avx.Divide(sumVec, Vector256.Create((float)rows));
            Vector256<float> varSumVec = Vector256<float>.Zero;

            for (int i = 0; i < rows; i++)
            {
                float* ptr = inPtr + i * stride + j;
                Vector256<float> rowVec = Avx.LoadAlignedVector256(ptr);
                Vector256<float> diff = Avx.Subtract(rowVec, meanVec);
                Vector256<float> sq = Avx.Multiply(diff, diff);
                varSumVec = Avx.Add(varSumVec, sq);
            }

            Vector256<float> varVec = Avx.Divide(varSumVec, Vector256.Create((float)rows));
            Vector256<float> stdVec = Avx.Sqrt(Avx.Add(varVec, epsilonVec));

            float* runningMeanPtr = &bn.RunningMean.Pointer[j];
            float* runningVarPtr = &bn.RunningVar.Pointer[j];
            float* gammaPtr = &bn.Gamma.Pointer[j];
            float* betaPtr = &bn.Beta.Pointer[j];

            Vector256<float> runningMeanVec = Avx.LoadAlignedVector256(runningMeanPtr);
            Vector256<float> runningVarVec = Avx.LoadAlignedVector256(runningVarPtr);
            Vector256<float> gammaVec = Avx.LoadAlignedVector256(gammaPtr);
            Vector256<float> betaVec = Avx.LoadAlignedVector256(betaPtr);

            runningMeanVec = Avx.Add(Avx.Multiply(momentumVec, meanVec), Avx.Multiply(oneMinusMomentumVec, runningMeanVec));
            runningVarVec = Avx.Add(Avx.Multiply(momentumVec, varVec), Avx.Multiply(oneMinusMomentumVec, runningVarVec));

            Avx.Store(runningMeanPtr, runningMeanVec);
            Avx.Store(runningVarPtr, runningVarVec);

            for (int i = 0; i < rows; i++)
            {
                float* ptr = inPtr + i * stride + j;
                Vector256<float> valVec = Avx.LoadAlignedVector256(ptr);
                Vector256<float> normVec = Avx.Divide(Avx.Subtract(valVec, meanVec), stdVec);
                Vector256<float> scaledVec = Avx.Add(Avx.Multiply(normVec, gammaVec), betaVec);
                Avx.StoreAligned(ptr, scaledVec);
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
                float std = MathF.Sqrt(var + BatchNormLayer.Epsilon);

                ref float runningMean = ref bn.RunningMean.At(0, j);
                ref float runningVar = ref bn.RunningVar.At(0, j);
                ref float gamma = ref bn.Gamma.At(0, j);
                ref float beta = ref bn.Beta.At(0, j);

                runningMean = BatchNormLayer.Momentum * mean + (1f - BatchNormLayer.Momentum) * runningMean;
                runningVar = BatchNormLayer.Momentum * var + (1f - BatchNormLayer.Momentum) * runningVar;

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
            var scaleVec = Vector256.Create(scale);
            var dropoutRateVec = Vector256.Create(dropoutRate);
            int vectorSize = NeuralMatrix.Alignment;

            for (; i <= total - vectorSize; i += vectorSize)
            {
                var vec = Avx.LoadVector256(activations.Pointer + i);
                Vector256<float> randVec = GenerateRandomVector256();

                var mask = Avx.CompareGreaterThan(randVec, dropoutRateVec);
                var scaledVec = Avx.Multiply(vec, scaleVec);

                var result = Avx.BlendVariable(scaledVec, Vector256<float>.Zero, mask);
                vec.StoreAligned(activations.Pointer + i);
            }
        }

        for (; i < total; i++)
        {
            float rand = (float)rng.NextDouble();
            activations.Pointer[i] = rand < dropoutRate ? 0f : activations.Pointer[i] * scale;
        }
    }

    private unsafe Vector256<float> GenerateRandomVector256()
    {
        float* randArray = stackalloc float[NeuralMatrix.Alignment];

        for (int i = 0; i < NeuralMatrix.Alignment; i++)
        {
            randArray[i] = (float)rng.NextDouble();
        }

        return Avx.LoadVector256(randArray);
    }
}
