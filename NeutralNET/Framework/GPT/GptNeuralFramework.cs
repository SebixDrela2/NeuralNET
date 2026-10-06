using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.GPT;

public unsafe class GptNeuralFramework : IDisposable
{
    public GptConfig Config;   // mutable so LR can be changed during warmup
    public readonly int MaxBatchSize;
    public readonly int MaxSequenceLength;
    public readonly int VocabSize;
    public readonly int EmbedDim;
    public readonly int NumHeads;
    public readonly int HeadDim;
    public readonly int NumLayers;
    public readonly int MlpHiddenDim;

    public int CurrentBatchSize { get; private set; }
    public int CurrentSeqLen { get; private set; }

    public NeuralMatrix TokenEmbeddings;
    public NeuralMatrix PositionalEmbeddings;
    public NeuralMatrix OutputProjection;
    public TransformerLayerBuffers[] Layers;

    private NeuralMatrix _mTokEmb, _vTokEmb;
    private NeuralMatrix _mPosEmb, _vPosEmb;
    private NeuralMatrix _mOutProj, _vOutProj;
    private NeuralMatrix _gTokEmb, _gPosEmb;
    private int _stepCount = 0;

    public NeuralMatrix InputTokenIds;
    public NeuralMatrix ResidualStream;
    public NeuralMatrix LogitsOutput;
    public NeuralMatrix ResidualGrad;

    public NeuralMatrix DebugMTokEmb => _mTokEmb;
    public NeuralMatrix DebugVTokEmb => _vTokEmb;
    public NeuralMatrix DebugMPosEmb => _mPosEmb;
    public NeuralMatrix DebugVPosEmb => _vPosEmb;
    public NeuralMatrix DebugMOutProj => _mOutProj;
    public NeuralMatrix DebugVOutProj => _vOutProj;

    public int StepCount
    {
        get => _stepCount;
        set => _stepCount = value;
    }

    public static bool DiagnosticsEnabled = true;
    private int _forwardCounter = 0;

    private bool _disposed;

    public GptNeuralFramework(GptConfig config)
        : this(
            config.MaxBatchSize,
            config.ContextSize,
            config.VocabSize,
            config.EmbedDim,
            config.NumHeads,
            config.NumLayers,
            config.IntermediateDim / config.EmbedDim > 0 ? config.IntermediateDim / config.EmbedDim : 4,
            config.LearningRate)
    {
        Config = config;
    }

    public GptNeuralFramework(
        int maxBatchSize,
        int maxSequenceLength,
        int vocabSize,
        int embedDim,
        int numHeads,
        int numLayers,
        int mlpHiddenMultiplier,
        float learningRate)
    {
        if (embedDim % numHeads != 0)
            throw new ArgumentException("EmbedDim must be divisible by NumHeads.");

        Config = new GptConfig
        {
            MaxBatchSize = maxBatchSize,
            ContextSize = maxSequenceLength,
            VocabSize = vocabSize,
            EmbedDim = embedDim,
            NumHeads = numHeads,
            NumLayers = numLayers,
            IntermediateDim = embedDim * mlpHiddenMultiplier,
            LearningRate = learningRate
        };

        MaxBatchSize = maxBatchSize;
        MaxSequenceLength = maxSequenceLength;
        VocabSize = vocabSize;
        EmbedDim = embedDim;
        NumHeads = numHeads;
        HeadDim = embedDim / numHeads;
        NumLayers = numLayers;
        MlpHiddenDim = embedDim * mlpHiddenMultiplier;

        CurrentBatchSize = maxBatchSize;
        CurrentSeqLen = maxSequenceLength;

        TokenEmbeddings = NeuralMatrix.GetOrCreate(VocabSize, EmbedDim);
        PositionalEmbeddings = NeuralMatrix.GetOrCreate(MaxSequenceLength, EmbedDim);
        OutputProjection = NeuralMatrix.GetOrCreate(EmbedDim, VocabSize);

        _mTokEmb = NeuralMatrix.GetOrCreate(VocabSize, EmbedDim);
        _vTokEmb = NeuralMatrix.GetOrCreate(VocabSize, EmbedDim);
        _mPosEmb = NeuralMatrix.GetOrCreate(MaxSequenceLength, EmbedDim);
        _vPosEmb = NeuralMatrix.GetOrCreate(MaxSequenceLength, EmbedDim);
        _mOutProj = NeuralMatrix.GetOrCreate(EmbedDim, VocabSize);
        _vOutProj = NeuralMatrix.GetOrCreate(EmbedDim, VocabSize);
        _gTokEmb = NeuralMatrix.GetOrCreate(VocabSize, EmbedDim);
        _gPosEmb = NeuralMatrix.GetOrCreate(MaxSequenceLength, EmbedDim);

        float std = 0.02f;
        float projStd = std / MathF.Sqrt(2f * NumLayers);

        TokenEmbeddings.RandomizeGaussian(0f, std);
        PositionalEmbeddings.RandomizeGaussian(0f, std);
        OutputProjection.RandomizeGaussian(0f, projStd);

        Layers = new TransformerLayerBuffers[NumLayers];
        for (int i = 0; i < NumLayers; i++)
        {
            Layers[i] = new TransformerLayerBuffers(
                MaxBatchSize,
                MaxSequenceLength,
                EmbedDim,
                NumHeads,
                HeadDim,
                MlpHiddenDim
            );
        }

        InputTokenIds = NeuralMatrix.GetOrCreate(MaxBatchSize, MaxSequenceLength);
        ResidualStream = NeuralMatrix.GetOrCreate(MaxBatchSize * MaxSequenceLength, EmbedDim);
        ResidualGrad = NeuralMatrix.GetOrCreate(MaxBatchSize * MaxSequenceLength, EmbedDim);
        LogitsOutput = NeuralMatrix.GetOrCreate(MaxBatchSize * MaxSequenceLength, VocabSize);
    }

    public void SetLearningRate(float lr)
    {
        Config.LearningRate = lr;
    }

    public void SetBatchAndSequenceLimit(int batchSize, int seqLen)
    {
        if (batchSize > MaxBatchSize || seqLen > MaxSequenceLength)
            throw new ArgumentOutOfRangeException("Requested dimensions exceed pre-allocated maximum capacity.");

        CurrentBatchSize = batchSize;
        CurrentSeqLen = seqLen;

        InputTokenIds.SetRowSize(batchSize);
        ResidualStream.SetRowSize(batchSize * seqLen);
        ResidualGrad.SetRowSize(batchSize * seqLen);
        LogitsOutput.SetRowSize(batchSize * seqLen);

        for (int i = 0; i < NumLayers; i++)
        {
            Layers[i].SetBatchAndSequenceLimit(batchSize, seqLen);
        }
    }

    public void Forward(int* tokenIdsInput, int batchSize, int seqLen)
    {
        SetBatchAndSequenceLimit(batchSize, seqLen);
        NativeMemory.Copy(tokenIdsInput, InputTokenIds.Pointer, (nuint)(batchSize * seqLen * sizeof(int)));
        EmbeddingForwardInPlace(tokenIdsInput, ResidualStream, batchSize, seqLen);
        _forwardCounter++;
        for (int l = 0; l < NumLayers; l++)
            Layers[l].Forward(ResidualStream, batchSize, seqLen);
        LmHeadForwardInPlace(ResidualStream, LogitsOutput);
    }

    public void Backward(int* pTargets, int batchSize, int seqLen)
    {
        int totalTokens = batchSize * seqLen;
        float lr = Config.LearningRate;
        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float eps = 1e-8f;
        const float weightDecay = 0.01f;
        _stepCount++;

        float biasCorrection1 = 1.0f - MathF.Pow(beta1, _stepCount);
        float biasCorrection2 = 1.0f - MathF.Pow(beta2, _stepCount);

        float* pLogits = LogitsOutput.Pointer;
        float* pResidual = ResidualStream.Pointer;
        float* pResGrad = ResidualGrad.Pointer;
        float* pOutProj = OutputProjection.Pointer;

        int logitsStride = LogitsOutput.ColumnsStride;
        int resStride = ResidualStream.ColumnsStride;
        int projStride = OutputProjection.ColumnsStride;

        float* pGradients = (float*)NativeMemory.Alloc((nuint)(totalTokens * logitsStride), sizeof(float));

        try
        {
            // 1. CE gradient: (softmax(logits) - onehot) / T
            Parallel.For(0, totalTokens, idx =>
            {
                int target = pTargets[idx];
                float* logitRow = pLogits + idx * logitsStride;
                float* gradRow = pGradients + idx * logitsStride;

                float maxLogit = float.NegativeInfinity;
                for (int v = 0; v < VocabSize; v++)
                    if (logitRow[v] > maxLogit) maxLogit = logitRow[v];

                if (float.IsNaN(maxLogit) || float.IsInfinity(maxLogit)) maxLogit = 0f;

                float sumExp = 0f;
                for (int v = 0; v < VocabSize; v++)
                {
                    float e = MathF.Exp(logitRow[v] - maxLogit);
                    gradRow[v] = e;
                    sumExp += e;
                }

                float invSum = 1.0f / sumExp;

                for (int v = 0; v < VocabSize; v++)
                {
                    float prob = gradRow[v] * invSum;
                    float g = (prob - (v == target ? 1.0f : 0.0f)) / totalTokens;

                    // Clip per-element gradient to prevent runaway when logits grow large.
                    // Natural scale is ~1/99/T ~= 6e-7, so 1e-3 is ~1600x the natural magnitude.
                    const float clip = 1e-3f;
                    if (g > clip) g = clip;
                    else if (g < -clip) g = -clip;

                    gradRow[v] = g;
                }
            });

            // 2. dx residual: dX = dY * OutputProjection^T
            Parallel.For(0, totalTokens, t =>
            {
                float* dy = pGradients + t * logitsStride;
                float* dx = pResGrad + t * resStride;
                for (int d = 0; d < EmbedDim; d++)
                {
                    float sum = 0f;
                    float* wRow = pOutProj + d * projStride;
                    for (int v = 0; v < VocabSize; v++) sum += dy[v] * wRow[v];
                    dx[d] = sum;
                }
            });

            // Optional second-stage clip on dX to prevent runaway into the layers.
            Parallel.For(0, totalTokens, t =>
            {
                float* dx = pResGrad + t * resStride;
                for (int d = 0; d < EmbedDim; d++)
                {
                    if (dx[d] > 1e-2f) dx[d] = 1e-2f;
                    else if (dx[d] < -1e-2f) dx[d] = -1e-2f;
                }
            });

            // 3. OutputProjection AdamW
            float* mProj = _mOutProj.Pointer;
            float* vProj = _vOutProj.Pointer;

            Parallel.For(0, EmbedDim, d =>
            {
                float* weightRow = pOutProj + d * projStride;
                float* mRow = mProj + d * projStride;
                float* vRow = vProj + d * projStride;

                for (int v = 0; v < VocabSize; v++)
                {
                    float gradAcc = 0f;
                    for (int t = 0; t < totalTokens; t++)
                        gradAcc += pResidual[t * resStride + d] * pGradients[t * logitsStride + v];

                    mRow[v] = beta1 * mRow[v] + (1f - beta1) * gradAcc;
                    vRow[v] = beta2 * vRow[v] + (1f - beta2) * (gradAcc * gradAcc);

                    float mHat = mRow[v] / biasCorrection1;
                    float vHat = vRow[v] / biasCorrection2;

                    float w = weightRow[v];
                    w -= lr * weightDecay * w;
                    w -= lr * (mHat / (MathF.Sqrt(vHat) + eps));
                    weightRow[v] = w;
                }
            });

            // 4. Layers backward
            for (int l = NumLayers - 1; l >= 0; l--)
                Layers[l].Backward(ResidualGrad, batchSize, seqLen, lr, _stepCount);

            // 5. Embeddings backward
            BackwardEmbeddings(batchSize, seqLen, lr, biasCorrection1, biasCorrection2, beta1, beta2, eps, weightDecay);
        }
        finally
        {
            NativeMemory.Free(pGradients);
        }
    }

    private void BackwardEmbeddings(
        int batchSize, int seqLen, float lr,
        float biasCorrection1, float biasCorrection2,
        float beta1, float beta2, float eps, float weightDecay)
    {
        int* pInputs = (int*)InputTokenIds.Pointer;
        float* pResGrad = ResidualGrad.Pointer;
        int resStride = ResidualGrad.ColumnsStride;

        int tokStride = TokenEmbeddings.ColumnsStride;
        int posStride = PositionalEmbeddings.ColumnsStride;

        _gTokEmb.Clear();
        _gPosEmb.Clear();

        float* pGTok = _gTokEmb.Pointer;
        float* pGPos = _gPosEmb.Pointer;

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int tokenId = pInputs[b * seqLen + s];
                if (tokenId < 0 || tokenId >= VocabSize) continue;

                float* gradRow = pResGrad + (b * seqLen + s) * resStride;
                float* gTokRow = pGTok + tokenId * tokStride;
                float* gPosRow = pGPos + s * posStride;

                for (int d = 0; d < EmbedDim; d++)
                {
                    float g = gradRow[d];
                    gTokRow[d] += g;
                    gPosRow[d] += g;
                }
            }
        }

        float* pTok = TokenEmbeddings.Pointer;
        float* mTok = _mTokEmb.Pointer;
        float* vTok = _vTokEmb.Pointer;

        Parallel.For(0, VocabSize, v =>
        {
            float* g = pGTok + v * tokStride;
            float* w = pTok + v * tokStride;
            float* m = mTok + v * tokStride;
            float* vv = vTok + v * tokStride;

            for (int d = 0; d < EmbedDim; d++)
            {
                float gi = g[d];
                m[d] = beta1 * m[d] + (1f - beta1) * gi;
                vv[d] = beta2 * vv[d] + (1f - beta2) * (gi * gi);

                float mHat = m[d] / biasCorrection1;
                float vHat = vv[d] / biasCorrection2;

                float weight = w[d];
                weight -= lr * weightDecay * weight;
                weight -= lr * (mHat / (MathF.Sqrt(vHat) + eps));
                w[d] = weight;
            }
        });

        float* pPos = PositionalEmbeddings.Pointer;
        float* mPos = _mPosEmb.Pointer;
        float* vPos = _vPosEmb.Pointer;

        Parallel.For(0, seqLen, s =>
        {
            float* g = pGPos + s * posStride;
            float* w = pPos + s * posStride;
            float* m = mPos + s * posStride;
            float* vv = vPos + s * posStride;

            for (int d = 0; d < EmbedDim; d++)
            {
                float gi = g[d];
                m[d] = beta1 * m[d] + (1f - beta1) * gi;
                vv[d] = beta2 * vv[d] + (1f - beta2) * (gi * gi);

                float mHat = m[d] / biasCorrection1;
                float vHat = vv[d] / biasCorrection2;

                float weight = w[d];
                weight -= lr * weightDecay * weight;
                weight -= lr * (mHat / (MathF.Sqrt(vHat) + eps));
                w[d] = weight;
            }
        });
    }

    public void TrainStep(int[][] miniBatch)
    {
        int batchSize = miniBatch.Length;
        int seqLen = miniBatch[0].Length - 1;
        int totalTokens = batchSize * seqLen;

        int* pInputs = (int*)NativeMemory.Alloc((nuint)totalTokens, sizeof(int));
        int* pTargets = (int*)NativeMemory.Alloc((nuint)totalTokens, sizeof(int));

        try
        {
            for (int b = 0; b < batchSize; b++)
                for (int t = 0; t < seqLen; t++)
                {
                    int idx = b * seqLen + t;
                    pInputs[idx] = miniBatch[b][t];
                    pTargets[idx] = miniBatch[b][t + 1];
                }

            Forward(pInputs, batchSize, seqLen);
            Backward(pTargets, batchSize, seqLen);
        }
        finally
        {
            NativeMemory.Free(pInputs);
            NativeMemory.Free(pTargets);
        }
    }

    public int[] Generate(int[] promptTokens, int maxNewTokens, float temperature = 1.0f, float topP = 1.0f)
    {
        List<int> tokens = new(promptTokens);
        Random rnd = new();

        for (int step = 0; step < maxNewTokens; step++)
        {
            int contextStart = Math.Max(0, tokens.Count - MaxSequenceLength);
            int currentContextLen = tokens.Count - contextStart;
            int[] context = tokens.GetRange(contextStart, currentContextLen).ToArray();

            fixed (int* pTokens = context)
            {
                Forward(pTokens, 1, currentContextLen);
            }

            int lastTokenIdx = currentContextLen - 1;
            float* logits = LogitsOutput.Pointer + lastTokenIdx * LogitsOutput.ColumnsStride;

            float maxLogit = float.NegativeInfinity;
            for (int v = 0; v < VocabSize; v++)
            {
                logits[v] /= Math.Max(temperature, 1e-5f);
                if (logits[v] > maxLogit) maxLogit = logits[v];
            }

            float sumExp = 0f;
            Span<float> probs = stackalloc float[VocabSize];
            for (int v = 0; v < VocabSize; v++)
            {
                probs[v] = MathF.Exp(logits[v] - maxLogit);
                sumExp += probs[v];
            }
            for (int v = 0; v < VocabSize; v++) probs[v] /= sumExp;

            int nextToken = SampleTopP(probs, topP, rnd);
            tokens.Add(nextToken);
        }

        return tokens.ToArray();
    }

    private int SampleTopP(Span<float> probs, float topP, Random rnd)
    {
        if (topP >= 1.0f)
        {
            float r = (float)rnd.NextDouble();
            float cumulative = 0f;
            for (int i = 0; i < probs.Length; i++)
            {
                cumulative += probs[i];
                if (r <= cumulative) return i;
            }
            return probs.Length - 1;
        }

        List<(float Prob, int Index)> sorted = new(probs.Length);
        for (int i = 0; i < probs.Length; i++) sorted.Add((probs[i], i));
        sorted.Sort((a, b) => b.Prob.CompareTo(a.Prob));

        float cumSum = 0f;
        int cutoffIndex = sorted.Count - 1;
        for (int i = 0; i < sorted.Count; i++)
        {
            cumSum += sorted[i].Prob;
            if (cumSum >= topP) { cutoffIndex = i; break; }
        }

        float rSample = (float)rnd.NextDouble() * cumSum;
        float runningSum = 0f;
        for (int i = 0; i <= cutoffIndex; i++)
        {
            runningSum += sorted[i].Prob;
            if (rSample <= runningSum) return sorted[i].Index;
        }
        return sorted[0].Index;
    }

    private void EmbeddingForwardInPlace(int* tokens, NeuralMatrix residual, int batch, int seq)
    {
        int embedDim = EmbedDim;
        int tokStride = TokenEmbeddings.ColumnsStride;
        int posStride = PositionalEmbeddings.ColumnsStride;
        int resStride = residual.ColumnsStride;

        float* pTok = TokenEmbeddings.Pointer;
        float* pPos = PositionalEmbeddings.Pointer;
        float* pRes = residual.Pointer;

        Parallel.For(0, batch, b =>
        {
            for (int s = 0; s < seq; s++)
            {
                int tokenId = tokens[b * seq + s];
                int tokenRowIndex = (tokenId >= 0 && tokenId < VocabSize) ? tokenId : 0;

                float* tokRow = pTok + (tokenRowIndex * tokStride);
                float* posRow = pPos + (s * posStride);
                float* dstRow = pRes + ((b * seq + s) * resStride);

                int i = 0;
                if (Avx512F.IsSupported)
                {
                    int vecLimit = embedDim - (embedDim % 16);
                    for (; i < vecLimit; i += 16)
                    {
                        var vT = Vector512.Load(tokRow + i);
                        var vP = Vector512.Load(posRow + i);
                        (vT + vP).Store(dstRow + i);
                    }
                }
                else if (Avx2.IsSupported)
                {
                    int vecLimit = embedDim - (embedDim % 8);
                    for (; i < vecLimit; i += 8)
                    {
                        var vT = Vector256.Load(tokRow + i);
                        var vP = Vector256.Load(posRow + i);
                        (vT + vP).Store(dstRow + i);
                    }
                }

                for (; i < embedDim; i++) dstRow[i] = tokRow[i] + posRow[i];
            }
        });
    }

    private void LmHeadForwardInPlace(NeuralMatrix inputStream, NeuralMatrix logitsOut)
    {
        inputStream.Dot(OutputProjection, logitsOut);

        ClipLogits(logitsOut, -20f, 20f);
    }

    private unsafe void ClipLogits(NeuralMatrix logits, float min, float max)
    {
        int rows = logits.Rows;
        int cols = logits.UsedColumns;
        float* p = logits.Pointer;
        int stride = logits.ColumnsStride;
        Parallel.For(0, rows, r =>
        {
            float* row = p + r * stride;
            for (int v = 0; v < cols; v++)
            {
                if (row[v] > max) row[v] = max;
                else if (row[v] < min) row[v] = min;
            }
        });
    }

    public void Dispose()
    {
        if (_disposed) return;

        TokenEmbeddings.Dispose();
        PositionalEmbeddings.Dispose();
        OutputProjection.Dispose();
        _mTokEmb.Dispose(); _vTokEmb.Dispose();
        _mPosEmb.Dispose(); _vPosEmb.Dispose();
        _mOutProj.Dispose(); _vOutProj.Dispose();
        _gTokEmb.Dispose();
        _gPosEmb.Dispose();

        InputTokenIds.Dispose();
        ResidualStream.Dispose();
        ResidualGrad.Dispose();
        LogitsOutput.Dispose();

        if (Layers != null)
            for (int i = 0; i < Layers.Length; i++)
                Layers[i]?.Dispose();

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
