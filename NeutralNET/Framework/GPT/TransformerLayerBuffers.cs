using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.GPT;

public unsafe class TransformerLayerBuffers : IDisposable
{
    public readonly int EmbedDim;
    public readonly int NumHeads;
    public readonly int HeadDim;
    public readonly int MlpHiddenDim;

    public NeuralMatrix Wq, Wk, Wv, Wo;
    public NeuralMatrix WGate, WUp, WDown;
    public NeuralMatrix Norm1Scale, Norm2Scale;

    public NeuralMatrix QScratch;
    public NeuralMatrix KScratch;
    public NeuralMatrix VScratch;
    public NeuralMatrix AttnScoresScratch;
    public NeuralMatrix MlpHiddenScratch;
    public NeuralMatrix NormScratch;
    public NeuralMatrix ContextScratch;

    public NeuralMatrix KeyCache;
    public NeuralMatrix ValueCache;

    private bool _disposed;

    public TransformerLayerBuffers(
        int maxBatch,
        int maxSeq,
        int embedDim,
        int numHeads,
        int headDim,
        int mlpHiddenDim)
    {
        EmbedDim = embedDim;
        NumHeads = numHeads;
        HeadDim = headDim;
        MlpHiddenDim = mlpHiddenDim;

        // Weights initialization
        Wq = NeuralMatrix.GetOrCreate(embedDim, embedDim);
        Wk = NeuralMatrix.GetOrCreate(embedDim, embedDim);
        Wv = NeuralMatrix.GetOrCreate(embedDim, embedDim);
        Wo = NeuralMatrix.GetOrCreate(embedDim, embedDim);

        WGate = NeuralMatrix.GetOrCreate(embedDim, mlpHiddenDim);
        WUp = NeuralMatrix.GetOrCreate(embedDim, mlpHiddenDim);
        WDown = NeuralMatrix.GetOrCreate(mlpHiddenDim, embedDim);

        Norm1Scale = NeuralMatrix.GetOrCreate(1, embedDim);
        Norm2Scale = NeuralMatrix.GetOrCreate(1, embedDim);

        // Fill normalization weights with 1.0
        Norm1Scale.Fill(1.0f);
        Norm2Scale.Fill(1.0f);

        // Randomize weight matrices
        Wq.RandomizeGaussian(0f, 0.02f);
        Wk.RandomizeGaussian(0f, 0.02f);
        Wv.RandomizeGaussian(0f, 0.02f);
        Wo.RandomizeGaussian(0f, 0.02f);
        WGate.RandomizeGaussian(0f, 0.02f);
        WUp.RandomizeGaussian(0f, 0.02f);
        WDown.RandomizeGaussian(0f, 0.02f);

        // Scratch buffers for activations
        QScratch = NeuralMatrix.GetOrCreate(maxBatch * maxSeq, embedDim);
        KScratch = NeuralMatrix.GetOrCreate(maxBatch * maxSeq, embedDim);
        VScratch = NeuralMatrix.GetOrCreate(maxBatch * maxSeq, embedDim);
        AttnScoresScratch = NeuralMatrix.GetOrCreate(maxBatch * numHeads * maxSeq, maxSeq);
        ContextScratch = NeuralMatrix.GetOrCreate(maxBatch * maxSeq, embedDim);
        MlpHiddenScratch = NeuralMatrix.GetOrCreate(maxBatch * maxSeq, mlpHiddenDim);
        NormScratch = NeuralMatrix.GetOrCreate(maxBatch * maxSeq, embedDim);

        // KV Cache
        KeyCache = NeuralMatrix.GetOrCreate(maxBatch * numHeads * maxSeq, headDim);
        ValueCache = NeuralMatrix.GetOrCreate(maxBatch * numHeads * maxSeq, headDim);
    }

    public void SetBatchAndSequenceLimit(int batchSize, int seqLen)
    {
        int totalTokens = batchSize * seqLen;
        QScratch.SetRowSize(totalTokens);
        KScratch.SetRowSize(totalTokens);
        VScratch.SetRowSize(totalTokens);
        ContextScratch.SetRowSize(totalTokens);
        MlpHiddenScratch.SetRowSize(totalTokens);
        NormScratch.SetRowSize(totalTokens);
    }

    public void Forward(NeuralMatrix residual, int batch, int seq)
    {
        int totalTokens = batch * seq;

        // 1. RMSNorm + Self-Attention Block
        ApplyRmsNorm(residual, NormScratch, Norm1Scale, totalTokens);

        NormScratch.Dot(Wq, QScratch);
        NormScratch.Dot(Wk, KScratch);
        NormScratch.Dot(Wv, VScratch);

        UpdateKvCache(batch, seq, 0);
        ComputeCausalAttention(batch, seq, seq);

        ContextScratch.Dot(Wo, NormScratch);
        residual.SumVectorized(NormScratch);

        // 2. RMSNorm + SwiGLU MLP Block
        ApplyRmsNorm(residual, NormScratch, Norm2Scale, totalTokens);

        ComputeSwiGluMlp(totalTokens);

        MlpHiddenScratch.Dot(WDown, NormScratch);
        residual.SumVectorized(NormScratch);
    }

    public void ForwardStep(NeuralMatrix residual, int batch, int stepIndex)
    {
        int totalTokens = batch;

        // 1. RMSNorm + Self-Attention Block
        ApplyRmsNorm(residual, NormScratch, Norm1Scale, totalTokens);

        NormScratch.Dot(Wq, QScratch);
        NormScratch.Dot(Wk, KScratch);
        NormScratch.Dot(Wv, VScratch);

        UpdateKvCache(batch, 1, stepIndex);
        ComputeCausalAttention(batch, 1, stepIndex + 1);

        ContextScratch.Dot(Wo, NormScratch);
        residual.SumVectorized(NormScratch);

        // 2. RMSNorm + SwiGLU MLP Block
        ApplyRmsNorm(residual, NormScratch, Norm2Scale, totalTokens);

        ComputeSwiGluMlp(totalTokens);

        MlpHiddenScratch.Dot(WDown, NormScratch);
        residual.SumVectorized(NormScratch);
    }

    private void ApplyRmsNorm(NeuralMatrix input, NeuralMatrix output, NeuralMatrix scale, int numRows)
    {
        int cols = EmbedDim;
        int inStride = input.ColumnsStride;
        int outStride = output.ColumnsStride;
        float* pIn = input.Pointer;
        float* pOut = output.Pointer;
        float* pScale = scale.Pointer;

        for (int r = 0; r < numRows; r++)
        {
            float* inRow = pIn + r * inStride;
            float* outRow = pOut + r * outStride;

            float sumSq = 0f;
            int i = 0;

            if (Avx2.IsSupported)
            {
                var sumVec = Vector256<float>.Zero;
                int vecLimit = cols - (cols % 8);
                for (; i < vecLimit; i += 8)
                {
                    var v = Vector256.Load(inRow + i);
                    sumVec = Fma.IsSupported
                        ? Fma.MultiplyAdd(v, v, sumVec)
                        : Avx.Add(sumVec, Avx.Multiply(v, v));
                }
                var hi = Avx.ExtractVector128(sumVec, 1);
                var lo = sumVec.GetLower();
                var sum128 = Sse.Add(lo, hi);
                sum128 = Sse3.HorizontalAdd(sum128, sum128);
                sum128 = Sse3.HorizontalAdd(sum128, sum128);
                sumSq += sum128.ToScalar();
            }

            for (; i < cols; i++)
            {
                sumSq += inRow[i] * inRow[i];
            }

            float scaleFactor = 1.0f / MathF.Sqrt((sumSq / cols) + 1e-5f);

            i = 0;
            if (Avx2.IsSupported)
            {
                var vScaleFact = Vector256.Create(scaleFactor);
                int vecLimit = cols - (cols % 8);
                for (; i < vecLimit; i += 8)
                {
                    var vIn = Vector256.Load(inRow + i);
                    var vSc = Vector256.Load(pScale + i);
                    var vNorm = Avx.Multiply(vIn, vScaleFact);
                    Avx.Multiply(vNorm, vSc).Store(outRow + i);
                }
            }

            for (; i < cols; i++)
            {
                outRow[i] = inRow[i] * scaleFactor * pScale[i];
            }
        }
    }

    private void UpdateKvCache(int batch, int seqLen, int startStep)
    {
        int headDim = HeadDim;
        int numHeads = NumHeads;
        int kStride = KScratch.ColumnsStride;
        int vStride = VScratch.ColumnsStride;
        int cacheStride = KeyCache.ColumnsStride;

        float* pK = KScratch.Pointer;
        float* pV = VScratch.Pointer;
        float* pKCache = KeyCache.Pointer;
        float* pVCache = ValueCache.Pointer;

        int totalMaxSeq = KeyCache.Rows / (batch * numHeads);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int tokenIdx = b * seqLen + s;
                int currentSeqPos = startStep + s;

                float* srcKRow = pK + tokenIdx * kStride;
                float* srcVRow = pV + tokenIdx * vStride;

                for (int h = 0; h < numHeads; h++)
                {
                    int cacheRow = (b * numHeads + h) * totalMaxSeq + currentSeqPos;
                    float* dstK = pKCache + cacheRow * cacheStride;
                    float* dstV = pVCache + cacheRow * cacheStride;

                    float* srcKHead = srcKRow + h * headDim;
                    float* srcVHead = srcVRow + h * headDim;

                    int i = 0;
                    if (Avx2.IsSupported)
                    {
                        int vecLimit = headDim - (headDim % 8);
                        for (; i < vecLimit; i += 8)
                        {
                            Vector256.Load(srcKHead + i).Store(dstK + i);
                            Vector256.Load(srcVHead + i).Store(dstV + i);
                        }
                    }

                    for (; i < headDim; i++)
                    {
                        dstK[i] = srcKHead[i];
                        dstV[i] = srcVHead[i];
                    }
                }
            }
        }
    }

    private void ComputeCausalAttention(int batch, int querySeqLen, int keySeqLen)
    {
        float invSqrtDim = 1.0f / MathF.Sqrt(HeadDim);
        int numHeads = NumHeads;
        int headDim = HeadDim;

        int qStride = QScratch.ColumnsStride;
        int scoresStride = AttnScoresScratch.ColumnsStride;
        int cacheStride = KeyCache.ColumnsStride;
        int ctxStride = ContextScratch.ColumnsStride;

        float* pQ = QScratch.Pointer;
        float* pScores = AttnScoresScratch.Pointer;
        float* pKCache = KeyCache.Pointer;
        float* pVCache = ValueCache.Pointer;
        float* pCtx = ContextScratch.Pointer;

        int totalMaxSeq = KeyCache.Rows / (batch * numHeads);

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int headCacheOffset = (b * numHeads + h) * totalMaxSeq;

                for (int q = 0; q < querySeqLen; q++)
                {
                    int qTokenIdx = b * querySeqLen + q;
                    float* qPtr = pQ + qTokenIdx * qStride + h * headDim;
                    float* scoreRow = pScores + (b * numHeads + h) * scoresStride + q;

                    float maxVal = float.NegativeInfinity;

                    for (int k = 0; k < keySeqLen; k++)
                    {
                        if (querySeqLen > 1 && k > q)
                        {
                            scoreRow[k] = float.NegativeInfinity;
                            continue;
                        }

                        float* kPtr = pKCache + (headCacheOffset + k) * cacheStride;
                        float dot = 0f;
                        int i = 0;

                        if (Avx2.IsSupported)
                        {
                            var sumVec = Vector256<float>.Zero;
                            int vecLimit = headDim - (headDim % 8);
                            for (; i < vecLimit; i += 8)
                            {
                                var vQ = Vector256.Load(qPtr + i);
                                var vK = Vector256.Load(kPtr + i);
                                sumVec = Fma.IsSupported
                                    ? Fma.MultiplyAdd(vQ, vK, sumVec)
                                    : Avx.Add(sumVec, Avx.Multiply(vQ, vK));
                            }
                            var hi = Avx.ExtractVector128(sumVec, 1);
                            var lo = sumVec.GetLower();
                            var sum128 = Sse.Add(lo, hi);
                            sum128 = Sse3.HorizontalAdd(sum128, sum128);
                            sum128 = Sse3.HorizontalAdd(sum128, sum128);
                            dot += sum128.ToScalar();
                        }

                        for (; i < headDim; i++)
                        {
                            dot += qPtr[i] * kPtr[i];
                        }

                        float val = dot * invSqrtDim;
                        scoreRow[k] = val;
                        if (val > maxVal) maxVal = val;
                    }

                    float expSum = 0f;
                    for (int k = 0; k < keySeqLen; k++)
                    {
                        if (querySeqLen > 1 && k > q)
                        {
                            scoreRow[k] = 0f;
                        }
                        else
                        {
                            float e = MathF.Exp(scoreRow[k] - maxVal);
                            scoreRow[k] = e;
                            expSum += e;
                        }
                    }

                    float invExpSum = expSum > 0f ? 1.0f / expSum : 0f;
                    for (int k = 0; k < keySeqLen; k++)
                    {
                        scoreRow[k] *= invExpSum;
                    }

                    float* ctxPtr = pCtx + qTokenIdx * ctxStride + h * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        ctxPtr[d] = 0f;
                    }

                    for (int k = 0; k < keySeqLen; k++)
                    {
                        float weight = scoreRow[k];
                        if (weight == 0f) continue;

                        float* vPtr = pVCache + (headCacheOffset + k) * cacheStride;
                        int i = 0;

                        if (Avx2.IsSupported)
                        {
                            var vWeight = Vector256.Create(weight);
                            int vecLimit = headDim - (headDim % 8);
                            for (; i < vecLimit; i += 8)
                            {
                                var vCtx = Vector256.Load(ctxPtr + i);
                                var vVal = Vector256.Load(vPtr + i);
                                vCtx = Fma.IsSupported
                                    ? Fma.MultiplyAdd(vVal, vWeight, vCtx)
                                    : Avx.Add(vCtx, Avx.Multiply(vVal, vWeight));
                                vCtx.Store(ctxPtr + i);
                            }
                        }

                        for (; i < headDim; i++)
                        {
                            ctxPtr[i] += weight * vPtr[i];
                        }
                    }
                }
            }
        }
    }

    private void ComputeSwiGluMlp(int numRows)
    {
        NeuralMatrix gateBuf = QScratch;
        NeuralMatrix upBuf = MlpHiddenScratch;

        NormScratch.Dot(WGate, gateBuf);
        NormScratch.Dot(WUp, upBuf);

        int hiddenDim = MlpHiddenDim;
        int gateStride = gateBuf.ColumnsStride;
        int upStride = upBuf.ColumnsStride;

        float* pGate = gateBuf.Pointer;
        float* pUp = upBuf.Pointer;

        for (int r = 0; r < numRows; r++)
        {
            float* gRow = pGate + r * gateStride;
            float* uRow = pUp + r * upStride;

            for (int i = 0; i < hiddenDim; i++)
            {
                float x = gRow[i];
                float silu = x / (1.0f + MathF.Exp(-x));
                uRow[i] = silu * uRow[i];
            }
        }
    }

    public void Dispose()
    {
        if (_disposed) return;

        Wq.Dispose(); Wk.Dispose(); Wv.Dispose(); Wo.Dispose();
        WGate.Dispose(); WUp.Dispose(); WDown.Dispose();
        Norm1Scale.Dispose(); Norm2Scale.Dispose();
        QScratch.Dispose(); KScratch.Dispose(); VScratch.Dispose();
        AttnScoresScratch.Dispose(); MlpHiddenScratch.Dispose(); NormScratch.Dispose();
        ContextScratch.Dispose();
        KeyCache.Dispose(); ValueCache.Dispose();

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
