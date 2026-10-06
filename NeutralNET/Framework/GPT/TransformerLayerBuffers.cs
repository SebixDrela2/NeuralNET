using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.GPT;

public unsafe class TransformerLayerBuffers : IDisposable
{
    public readonly int EmbedDim;
    public readonly int NumHeads;
    public readonly int HeadDim;
    public readonly int MlpHiddenDim;

    private readonly int MaxTokens;

    public NeuralMatrix DebugAttnOut => AttnOut;
    public NeuralMatrix DebugWo => Wo;
    public NeuralMatrix DebugWq => Wq;
    public NeuralMatrix DebugQ => Q;
    public NeuralMatrix DebugK => K;
    public NeuralMatrix DebugV => V;
    public NeuralMatrix DebugDAttnOut => dAttnOut;
    public NeuralMatrix DebugDQ => dQ;
    public NeuralMatrix DebugDK => dK;
    public NeuralMatrix DebugDV => dV;
    public NeuralMatrix DebugMWq => mWq;
    public NeuralMatrix DebugVWq => vWq;
    public NeuralMatrix DebugMWk => mWk;
    public NeuralMatrix DebugVWk => vWk;
    public NeuralMatrix DebugMWv => mWv;
    public NeuralMatrix DebugVWv => vWv;
    public NeuralMatrix DebugMWo => mWo;
    public NeuralMatrix DebugVWo => vWo;
    public NeuralMatrix DebugMWGate => mWGate;
    public NeuralMatrix DebugVWGate => vWGate;
    public NeuralMatrix DebugMWUp => mWUp;
    public NeuralMatrix DebugVWUp => vWUp;
    public NeuralMatrix DebugMWDown => mWDown;
    public NeuralMatrix DebugVWDown => vWDown;
    public NeuralMatrix DebugMNorm1 => mNorm1;
    public NeuralMatrix DebugVNorm1 => vNorm1;
    public NeuralMatrix DebugMNorm2 => mNorm2;
    public NeuralMatrix DebugVNorm2 => vNorm2;

    public static bool DiagEnabled = false;
    public static int DiagStep = 0;
    public static int DiagLayerLimit = 99;

    public NeuralMatrix Wq, Wk, Wv, Wo;
    public NeuralMatrix WGate, WUp, WDown;
    public NeuralMatrix Norm1Scale, Norm2Scale;

    private NeuralMatrix mWq, vWq, mWk, vWk, mWv, vWv, mWo, vWo;
    private NeuralMatrix mWGate, vWGate, mWUp, vWUp, mWDown, vWDown;
    private NeuralMatrix mNorm1, vNorm1, mNorm2, vNorm2;
    private NeuralMatrix dNorm1Scale, dNorm2Scale;

    private NeuralMatrix InputResidual;
    private NeuralMatrix Norm1Out;
    private NeuralMatrix Q, K, V;
    private NeuralMatrix AttnOut;
    private NeuralMatrix AttnScores;
    private NeuralMatrix ResidualMid;
    private NeuralMatrix Norm2Out;
    private NeuralMatrix GatePre;
    private NeuralMatrix UpBranch;
    private NeuralMatrix MlpActivated;

    private NeuralMatrix dMlpPreDown;
    private NeuralMatrix dGatePre;
    private NeuralMatrix dUp;
    private NeuralMatrix dQ, dK, dV;
    private NeuralMatrix dAttnOut;
    private NeuralMatrix dNorm1, dNorm2;
    private NeuralMatrix ScratchE;

    private float* _rBuffer;
    private bool _disposed;

    public TransformerLayerBuffers(int maxBatch, int maxSeq, int embedDim, int numHeads, int headDim, int mlpHiddenDim)
    {
        int E = embedDim;
        int F = mlpHiddenDim;
        MaxTokens = maxBatch * maxSeq;
        EmbedDim = embedDim;
        NumHeads = numHeads;
        HeadDim = headDim;
        MlpHiddenDim = mlpHiddenDim;

        Wq = NeuralMatrix.GetOrCreate(E, E);
        Wk = NeuralMatrix.GetOrCreate(E, E);
        Wv = NeuralMatrix.GetOrCreate(E, E);
        Wo = NeuralMatrix.GetOrCreate(E, E);
        WGate = NeuralMatrix.GetOrCreate(E, F);
        WUp = NeuralMatrix.GetOrCreate(E, F);
        WDown = NeuralMatrix.GetOrCreate(F, E);

        Norm1Scale = NeuralMatrix.GetOrCreate(1, E);
        Norm2Scale = NeuralMatrix.GetOrCreate(1, E);
        Norm1Scale.Fill(1f);
        Norm2Scale.Fill(1f);

        const float std = 0.02f;
        Wq.RandomizeGaussian(0f, std);
        Wk.RandomizeGaussian(0f, std);
        Wv.RandomizeGaussian(0f, std);
        Wo.RandomizeGaussian(0f, std);
        WGate.RandomizeGaussian(0f, std);
        WUp.RandomizeGaussian(0f, std);
        WDown.RandomizeGaussian(0f, std);

        mWq = NeuralMatrix.GetOrCreate(E, E); vWq = NeuralMatrix.GetOrCreate(E, E);
        mWk = NeuralMatrix.GetOrCreate(E, E); vWk = NeuralMatrix.GetOrCreate(E, E);
        mWv = NeuralMatrix.GetOrCreate(E, E); vWv = NeuralMatrix.GetOrCreate(E, E);
        mWo = NeuralMatrix.GetOrCreate(E, E); vWo = NeuralMatrix.GetOrCreate(E, E);
        mWGate = NeuralMatrix.GetOrCreate(E, F); vWGate = NeuralMatrix.GetOrCreate(E, F);
        mWUp = NeuralMatrix.GetOrCreate(E, F); vWUp = NeuralMatrix.GetOrCreate(E, F);
        mWDown = NeuralMatrix.GetOrCreate(F, E); vWDown = NeuralMatrix.GetOrCreate(F, E);
        mNorm1 = NeuralMatrix.GetOrCreate(1, E); vNorm1 = NeuralMatrix.GetOrCreate(1, E);
        mNorm2 = NeuralMatrix.GetOrCreate(1, E); vNorm2 = NeuralMatrix.GetOrCreate(1, E);
        dNorm1Scale = NeuralMatrix.GetOrCreate(1, E);
        dNorm2Scale = NeuralMatrix.GetOrCreate(1, E);

        InputResidual = NeuralMatrix.GetOrCreate(MaxTokens, E);
        Norm1Out = NeuralMatrix.GetOrCreate(MaxTokens, E);
        Q = NeuralMatrix.GetOrCreate(MaxTokens, E);
        K = NeuralMatrix.GetOrCreate(MaxTokens, E);
        V = NeuralMatrix.GetOrCreate(MaxTokens, E);
        AttnOut = NeuralMatrix.GetOrCreate(MaxTokens, E);
        AttnScores = NeuralMatrix.GetOrCreate(maxBatch * numHeads * maxSeq, maxSeq);
        ResidualMid = NeuralMatrix.GetOrCreate(MaxTokens, E);
        Norm2Out = NeuralMatrix.GetOrCreate(MaxTokens, E);
        GatePre = NeuralMatrix.GetOrCreate(MaxTokens, F);
        UpBranch = NeuralMatrix.GetOrCreate(MaxTokens, F);
        MlpActivated = NeuralMatrix.GetOrCreate(MaxTokens, F);

        dMlpPreDown = NeuralMatrix.GetOrCreate(MaxTokens, F);
        dGatePre = NeuralMatrix.GetOrCreate(MaxTokens, F);
        dUp = NeuralMatrix.GetOrCreate(MaxTokens, F);
        dQ = NeuralMatrix.GetOrCreate(MaxTokens, E);
        dK = NeuralMatrix.GetOrCreate(MaxTokens, E);
        dV = NeuralMatrix.GetOrCreate(MaxTokens, E);
        dAttnOut = NeuralMatrix.GetOrCreate(MaxTokens, E);
        dNorm1 = NeuralMatrix.GetOrCreate(MaxTokens, E);
        dNorm2 = NeuralMatrix.GetOrCreate(MaxTokens, E);
        ScratchE = NeuralMatrix.GetOrCreate(MaxTokens, E);

        _rBuffer = (float*)NativeMemory.Alloc((nuint)MaxTokens, sizeof(float));
    }

    public void SetBatchAndSequenceLimit(int batchSize, int seqLen)
    {
        int T = batchSize * seqLen;
        int scoreRows = batchSize * NumHeads * seqLen;

        InputResidual.SetRowSize(T);
        Norm1Out.SetRowSize(T);
        Q.SetRowSize(T);
        K.SetRowSize(T);
        V.SetRowSize(T);
        AttnOut.SetRowSize(T);
        AttnScores.SetRowSize(scoreRows);
        ResidualMid.SetRowSize(T);
        Norm2Out.SetRowSize(T);
        GatePre.SetRowSize(T);
        UpBranch.SetRowSize(T);
        MlpActivated.SetRowSize(T);

        dMlpPreDown.SetRowSize(T);
        dGatePre.SetRowSize(T);
        dUp.SetRowSize(T);
        dQ.SetRowSize(T);
        dK.SetRowSize(T);
        dV.SetRowSize(T);
        dAttnOut.SetRowSize(T);
        dNorm1.SetRowSize(T);
        dNorm2.SetRowSize(T);
        ScratchE.SetRowSize(T);
    }

    public void Forward(NeuralMatrix residual, int batch, int seq)
    {
        int T = batch * seq;
        int E = EmbedDim;

        CopyRows(residual, InputResidual, T, E);
        ApplyRmsNorm(residual, Norm1Out, Norm1Scale, T);

        Norm1Out.Dot(Wq, Q);
        Norm1Out.Dot(Wk, K);
        Norm1Out.Dot(Wv, V);

        AttentionForward(batch, seq);

        AttnOut.Dot(Wo, ScratchE);
        residual.SumVectorized(ScratchE);

        CopyRows(residual, ResidualMid, T, E);
        ApplyRmsNorm(residual, Norm2Out, Norm2Scale, T);

        Norm2Out.Dot(WGate, GatePre);
        Norm2Out.Dot(WUp, UpBranch);

        ComputeSwiGlu(T);

        MlpActivated.Dot(WDown, ScratchE);
        residual.SumVectorized(ScratchE);
    }

    public void Backward(NeuralMatrix residualGrad, int batch, int seqLen, float lr, int stepCount)
    {
        int T = batch * seqLen;

        // ---- MLP backward ----
        residualGrad.DotTranspose(WDown, dMlpPreDown);
        UpdateWeightsAdamW(MlpActivated, residualGrad, WDown, mWDown, vWDown, lr, stepCount, "WDown");

        BackwardSwiGlu(T, dMlpPreDown, dGatePre, dUp);

        UpdateWeightsAdamW(Norm2Out, dGatePre, WGate, mWGate, vWGate, lr, stepCount, "WGate");
        UpdateWeightsAdamW(Norm2Out, dUp, WUp, mWUp, vWUp, lr, stepCount, "WUp");

        dGatePre.DotTranspose(WGate, dNorm2);
        dUp.DotTranspose(WUp, ScratchE);
        dNorm2.SumVectorized(ScratchE);

        RmsNormBackward(ResidualMid, dNorm2, Norm2Scale, ScratchE, dNorm2Scale, T);
        residualGrad.SumVectorized(ScratchE);
        UpdateScaleAdamW(dNorm2Scale, Norm2Scale, mNorm2, vNorm2, lr, stepCount);

        // ---- Attention backward ----
        residualGrad.DotTranspose(Wo, dAttnOut);
        UpdateWeightsAdamW(AttnOut, residualGrad, Wo, mWo, vWo, lr, stepCount, "Wo");

        AttentionBackward(batch, seqLen, dAttnOut);

        UpdateWeightsAdamW(Norm1Out, dQ, Wq, mWq, vWq, lr, stepCount, "Wq");
        UpdateWeightsAdamW(Norm1Out, dK, Wk, mWk, vWk, lr, stepCount, "Wk");
        UpdateWeightsAdamW(Norm1Out, dV, Wv, mWv, vWv, lr, stepCount, "Wv");

        dQ.DotTranspose(Wq, dNorm1);
        dK.DotTranspose(Wk, ScratchE);
        dNorm1.SumVectorized(ScratchE);
        dV.DotTranspose(Wv, ScratchE);
        dNorm1.SumVectorized(ScratchE);

        RmsNormBackward(InputResidual, dNorm1, Norm1Scale, ScratchE, dNorm1Scale, T);
        residualGrad.SumVectorized(ScratchE);
        UpdateScaleAdamW(dNorm1Scale, Norm1Scale, mNorm1, vNorm1, lr, stepCount);
    }

    private static void CopyRows(NeuralMatrix src, NeuralMatrix dst, int T, int E)
    {
        int sStride = src.ColumnsStride;
        int dStride = dst.ColumnsStride;
        float* sP = src.Pointer;
        float* dP = dst.Pointer;
        Parallel.For(0, T, i =>
        {
            float* s = sP + i * sStride;
            float* d = dP + i * dStride;
            for (int j = 0; j < E; j++) d[j] = s[j];
        });
    }

    private void AttentionForward(int batch, int seq)
    {
        int H = NumHeads;
        int D = HeadDim;
        float invSqrtD = 1f / MathF.Sqrt(D);

        float* pQ = Q.Pointer; int qStride = Q.ColumnsStride;
        float* pK = K.Pointer; int kStride = K.ColumnsStride;
        float* pV = V.Pointer; int vStride = V.ColumnsStride;
        float* pOut = AttnOut.Pointer; int outStride = AttnOut.ColumnsStride;
        float* pScores = AttnScores.Pointer; int scoreStride = AttnScores.ColumnsStride;

        Parallel.For(0, batch * H, bh =>
        {
            int b = bh / H;
            int h = bh % H;

            for (int q = 0; q < seq; q++)
            {
                int qRow = b * seq + q;
                float* qHead = pQ + qRow * qStride + h * D;
                float* scoreRow = pScores + (bh * seq + q) * scoreStride;

                float maxScore = float.NegativeInfinity;
                for (int k = 0; k <= q; k++)
                {
                    float* kHead = pK + (b * seq + k) * kStride + h * D;
                    float dot = 0f;
                    for (int d = 0; d < D; d++) dot += qHead[d] * kHead[d];
                    dot *= invSqrtD;
                    scoreRow[k] = dot;
                    if (dot > maxScore) maxScore = dot;
                }
                for (int k = q + 1; k < seq; k++) scoreRow[k] = 0f;

                float sumExp = 0f;
                for (int k = 0; k <= q; k++)
                {
                    float e = MathF.Exp(scoreRow[k] - maxScore);
                    scoreRow[k] = e;
                    sumExp += e;
                }
                float invSum = 1f / sumExp;
                for (int k = 0; k <= q; k++) scoreRow[k] *= invSum;

                float* outHead = pOut + qRow * outStride + h * D;
                for (int d = 0; d < D; d++) outHead[d] = 0f;
                for (int k = 0; k <= q; k++)
                {
                    float p = scoreRow[k];
                    if (p == 0f) continue;
                    float* vHead = pV + (b * seq + k) * vStride + h * D;
                    for (int d = 0; d < D; d++) outHead[d] += p * vHead[d];
                }
            }
        });
    }

    private void AttentionBackward(int batch, int seq, NeuralMatrix dAttnOutIn)
    {
        int H = NumHeads;
        int D = HeadDim;
        float invSqrtD = 1f / MathF.Sqrt(D);

        float* pQ = Q.Pointer; int qStride = Q.ColumnsStride;
        float* pK = K.Pointer; int kStride = K.ColumnsStride;
        float* pV = V.Pointer; int vStride = V.ColumnsStride;
        float* pScores = AttnScores.Pointer; int scoreStride = AttnScores.ColumnsStride;
        float* pdOut = dAttnOutIn.Pointer; int dOutStride = dAttnOutIn.ColumnsStride;
        float* pdQ = dQ.Pointer; int dqStride = dQ.ColumnsStride;
        float* pdK = dK.Pointer; int dkStride = dK.ColumnsStride;
        float* pdV = dV.Pointer; int dvStride = dV.ColumnsStride;

        Parallel.For(0, batch * H, bh =>
        {
            int b = bh / H;
            int h = bh % H;

            for (int s = 0; s < seq; s++)
            {
                int row = b * seq + s;
                float* dq = pdQ + row * dqStride + h * D;
                float* dk = pdK + row * dkStride + h * D;
                float* dv = pdV + row * dvStride + h * D;
                for (int d = 0; d < D; d++)
                {
                    dq[d] = 0f;
                    dk[d] = 0f;
                    dv[d] = 0f;
                }
            }

            for (int q = 0; q < seq; q++)
            {
                int qRow = b * seq + q;
                float* qHead = pQ + qRow * qStride + h * D;
                float* dqHead = pdQ + qRow * dqStride + h * D;
                float* doutHead = pdOut + qRow * dOutStride + h * D;
                float* scoreRow = pScores + (bh * seq + q) * scoreStride;

                float sumDS = 0f;
                Span<float> dS = stackalloc float[q + 1];

                for (int k = 0; k <= q; k++)
                {
                    float* vHead = pV + (b * seq + k) * vStride + h * D;
                    float* dvHead = pdV + (b * seq + k) * dvStride + h * D;
                    float p = scoreRow[k];

                    float dot = 0f;
                    for (int d = 0; d < D; d++)
                    {
                        dot += doutHead[d] * vHead[d];
                        dvHead[d] += p * doutHead[d];
                    }

                    dS[k] = dot;
                    sumDS += dot * p;
                }

                for (int k = 0; k <= q; k++)
                {
                    float p = scoreRow[k];
                    float dSoftmax = p * (dS[k] - sumDS) * invSqrtD;

                    float* kHead = pK + (b * seq + k) * kStride + h * D;
                    float* dkHead = pdK + (b * seq + k) * dkStride + h * D;

                    for (int d = 0; d < D; d++)
                    {
                        dqHead[d] += dSoftmax * kHead[d];
                        dkHead[d] += dSoftmax * qHead[d];
                    }
                }
            }
        });
    }

    private void ComputeSwiGlu(int T)
    {
        int F = MlpHiddenDim;
        int gpStride = GatePre.ColumnsStride;
        int upStride = UpBranch.ColumnsStride;
        int mAStride = MlpActivated.ColumnsStride;
        float* pG = GatePre.Pointer;
        float* pU = UpBranch.Pointer;
        float* pM = MlpActivated.Pointer;

        Parallel.For(0, T, r =>
        {
            float* g = pG + r * gpStride;
            float* u = pU + r * upStride;
            float* m = pM + r * mAStride;
            for (int i = 0; i < F; i++)
            {
                float x = g[i];
                float silu = x / (1f + MathF.Exp(-x));
                m[i] = silu * u[i];
            }
        });
    }

    private void BackwardSwiGlu(int T, NeuralMatrix dMlpActivated, NeuralMatrix dGateOut, NeuralMatrix dUpOut)
    {
        int F = MlpHiddenDim;
        int gStride = GatePre.ColumnsStride;
        int uStride = UpBranch.ColumnsStride;
        int dMStride = dMlpActivated.ColumnsStride;
        int dgStride = dGateOut.ColumnsStride;
        int duStride = dUpOut.ColumnsStride;

        float* pG = GatePre.Pointer;
        float* pU = UpBranch.Pointer;
        float* pDM = dMlpActivated.Pointer;
        float* pDG = dGateOut.Pointer;
        float* pDU = dUpOut.Pointer;

        Parallel.For(0, T, r =>
        {
            float* g = pG + r * gStride;
            float* u = pU + r * uStride;
            float* dM = pDM + r * dMStride;
            float* dG = pDG + r * dgStride;
            float* dU = pDU + r * duStride;

            for (int i = 0; i < F; i++)
            {
                float gi = g[i];
                float ui = u[i];
                float sig = 1f / (1f + MathF.Exp(-gi));
                float silu = gi * sig;
                float dSilu = sig * (1f + gi * (1f - sig));

                float dOut = dM[i];
                dG[i] = dOut * ui * dSilu;
                dU[i] = dOut * silu;
            }
        });
    }

    private void RmsNormBackward(
        NeuralMatrix x, NeuralMatrix dY, NeuralMatrix scale,
        NeuralMatrix dX, NeuralMatrix dScaleAcc, int T)
    {
        int E = EmbedDim;
        const float eps = 1e-5f;

        float* pX = x.Pointer; int xStride = x.ColumnsStride;
        float* pDY = dY.Pointer; int dyStride = dY.ColumnsStride;
        float* pDX = dX.Pointer; int dxStride = dX.ColumnsStride;
        float* pScale = scale.Pointer;
        float* pDScale = dScaleAcc.Pointer;
        float* rBuf = _rBuffer;

        Parallel.For(0, T, row =>
        {
            float* xRow = pX + row * xStride;
            float sumSq = 0f;
            for (int i = 0; i < E; i++) sumSq += xRow[i] * xRow[i];
            rBuf[row] = 1f / MathF.Sqrt(sumSq / E + eps);
        });

        Parallel.For(0, T, row =>
        {
            float* xRow = pX + row * xStride;
            float* dyRow = pDY + row * dyStride;
            float* dxRow = pDX + row * dxStride;
            float r = rBuf[row];
            float r3n = r * r * r / E;

            float s = 0f;
            for (int i = 0; i < E; i++) s += dyRow[i] * xRow[i] * pScale[i];

            for (int i = 0; i < E; i++)
                dxRow[i] = r * pScale[i] * dyRow[i] - r3n * xRow[i] * s;
        });

        for (int i = 0; i < E; i++)
        {
            float grad = 0f;
            for (int row = 0; row < T; row++)
                grad += pDY[row * dyStride + i] * pX[row * xStride + i] * rBuf[row];
            pDScale[i] = grad;
        }
    }

    private void ApplyRmsNorm(NeuralMatrix input, NeuralMatrix output, NeuralMatrix scale, int numRows)
    {
        int cols = EmbedDim;
        int inStride = input.ColumnsStride;
        int outStride = output.ColumnsStride;
        float* pIn = input.Pointer;
        float* pOut = output.Pointer;
        float* pScale = scale.Pointer;

        Parallel.For(0, numRows, r =>
        {
            float* inRow = pIn + r * inStride;
            float* outRow = pOut + r * outStride;

            float sumSq = 0f;
            int i = 0;

            if (Avx2.IsSupported)
            {
                var sumVec0 = Vector256<float>.Zero;
                var sumVec1 = Vector256<float>.Zero;
                int vecLimit = cols - (cols % 16);
                for (; i < vecLimit; i += 16)
                {
                    var v0 = Vector256.Load(inRow + i);
                    var v1 = Vector256.Load(inRow + i + 8);
                    sumVec0 = Fma.IsSupported ? Fma.MultiplyAdd(v0, v0, sumVec0) : Avx.Add(sumVec0, Avx.Multiply(v0, v0));
                    sumVec1 = Fma.IsSupported ? Fma.MultiplyAdd(v1, v1, sumVec1) : Avx.Add(sumVec1, Avx.Multiply(v1, v1));
                }
                sumVec0 = Avx.Add(sumVec0, sumVec1);
                var hi = Avx.ExtractVector128(sumVec0, 1);
                var lo = sumVec0.GetLower();
                var sum128 = Sse.Add(lo, hi);
                sum128 = Sse3.HorizontalAdd(sum128, sum128);
                sum128 = Sse3.HorizontalAdd(sum128, sum128);
                sumSq += sum128.ToScalar();
            }

            for (; i < cols; i++) sumSq += inRow[i] * inRow[i];

            float scaleFactor = 1.0f / MathF.Sqrt((sumSq / cols) + 1e-5f);

            i = 0;
            if (Avx2.IsSupported)
            {
                var vScaleFact = Vector256.Create(scaleFactor);
                int vecLimit = cols - (cols % 16);
                for (; i < vecLimit; i += 16)
                {
                    var vIn0 = Vector256.Load(inRow + i);
                    var vSc0 = Vector256.Load(pScale + i);
                    Avx.Multiply(Avx.Multiply(vIn0, vScaleFact), vSc0).Store(outRow + i);

                    var vIn1 = Vector256.Load(inRow + i + 8);
                    var vSc1 = Vector256.Load(pScale + i + 8);
                    Avx.Multiply(Avx.Multiply(vIn1, vScaleFact), vSc1).Store(outRow + i + 8);
                }
            }

            for (; i < cols; i++)
                outRow[i] = inRow[i] * scaleFactor * pScale[i];
        });
    }

    private void UpdateWeightsAdamW(
    NeuralMatrix inputActivations,
    NeuralMatrix outputGrads,
    NeuralMatrix weight,
    NeuralMatrix mState,
    NeuralMatrix vState,
    float lr,
    int stepCount,
    string name = "?")
    {
        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float eps = 1e-8f;
        const float weightDecay = 0.01f;

        float bc1 = 1f - MathF.Pow(beta1, stepCount);
        float bc2 = 1f - MathF.Pow(beta2, stepCount);

        int rows = weight.Rows;        // in_features
        int cols = weight.UsedColumns; // out_features
        int T = inputActivations.Rows;
        if (T == 0) return;

        float* pX = inputActivations.Pointer; int xStride = inputActivations.ColumnsStride;
        float* pDY = outputGrads.Pointer; int dyStride = outputGrads.ColumnsStride;
        float* pW = weight.Pointer; int wStride = weight.ColumnsStride;
        float* pM = mState.Pointer;
        float* pV = vState.Pointer;

        // Row-major scratch for the full gradient matrix [rows, cols].
        float* gradW = (float*)NativeMemory.Alloc((nuint)(rows * cols), sizeof(float));
        try
        {
            // Clear the scratch. rows*cols is at most 128*512 = 65536 floats = 256 KB.
            new Span<float>(gradW, rows * cols).Clear();

            // Pass 1: accumulate gradW[r, c] = sum_t X[t, r] * dY[t, c]
            // Parallelize over r. Inner loop reads dY[t, :] sequentially, so every
            // cache line of dY is fully consumed in one pass.
            Parallel.For(0, rows, r =>
            {
                float* gRow = gradW + r * cols;
                for (int t = 0; t < T; t++)
                {
                    float xVal = pX[t * xStride + r];
                    if (xVal == 0f) continue;
                    float* dyRow = pDY + t * dyStride;
                    for (int c = 0; c < cols; c++)
                        gRow[c] += xVal * dyRow[c];
                }
            });

            // Pass 2: AdamW elementwise.
            Parallel.For(0, rows, r =>
            {
                float* wRow = pW + r * wStride;
                float* mRow = pM + r * wStride;
                float* vRow = pV + r * wStride;
                float* gRow = gradW + r * cols;

                for (int c = 0; c < cols; c++)
                {
                    float grad = gRow[c];
                    mRow[c] = beta1 * mRow[c] + (1f - beta1) * grad;
                    vRow[c] = beta2 * vRow[c] + (1f - beta2) * (grad * grad);

                    float mHat = mRow[c] / bc1;
                    float vHat = vRow[c] / bc2;

                    wRow[c] -= lr * (mHat / (MathF.Sqrt(vHat) + eps) + weightDecay * wRow[c]);
                }
            });
        }
        finally
        {
            NativeMemory.Free(gradW);
        }
    }

    private void UpdateScaleAdamW(
        NeuralMatrix dScaleGrad, NeuralMatrix scale,
        NeuralMatrix mState, NeuralMatrix vState,
        float lr, int stepCount)
    {
        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float eps = 1e-8f;
        const float weightDecay = 0.01f;

        float bc1 = 1f - MathF.Pow(beta1, stepCount);
        float bc2 = 1f - MathF.Pow(beta2, stepCount);

        int E = EmbedDim;
        float* g = dScaleGrad.Pointer;
        float* s = scale.Pointer;
        float* m = mState.Pointer;
        float* v = vState.Pointer;

        for (int i = 0; i < E; i++)
        {
            float gi = g[i];
            m[i] = beta1 * m[i] + (1f - beta1) * gi;
            v[i] = beta2 * v[i] + (1f - beta2) * (gi * gi);
            float mHat = m[i] / bc1;
            float vHat = v[i] / bc2;
            s[i] -= lr * (mHat / (MathF.Sqrt(vHat) + eps) + weightDecay * s[i]);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;

        Wq.Dispose(); Wk.Dispose(); Wv.Dispose(); Wo.Dispose();
        WGate.Dispose(); WUp.Dispose(); WDown.Dispose();
        Norm1Scale.Dispose(); Norm2Scale.Dispose();

        mWq.Dispose(); vWq.Dispose(); mWk.Dispose(); vWk.Dispose();
        mWv.Dispose(); vWv.Dispose(); mWo.Dispose(); vWo.Dispose();
        mWGate.Dispose(); vWGate.Dispose(); mWUp.Dispose(); vWUp.Dispose();
        mWDown.Dispose(); vWDown.Dispose();
        mNorm1.Dispose(); vNorm1.Dispose(); mNorm2.Dispose(); vNorm2.Dispose();
        dNorm1Scale.Dispose(); dNorm2Scale.Dispose();

        InputResidual.Dispose(); Norm1Out.Dispose();
        Q.Dispose(); K.Dispose(); V.Dispose(); AttnOut.Dispose();
        AttnScores.Dispose(); ResidualMid.Dispose(); Norm2Out.Dispose();
        GatePre.Dispose(); UpBranch.Dispose(); MlpActivated.Dispose();

        dMlpPreDown.Dispose(); dGatePre.Dispose(); dUp.Dispose();
        dQ.Dispose(); dK.Dispose(); dV.Dispose(); dAttnOut.Dispose();
        dNorm1.Dispose(); dNorm2.Dispose(); ScratchE.Dispose();

        if (_rBuffer != null)
        {
            NativeMemory.Free(_rBuffer);
            _rBuffer = null;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
