using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.GPT;

public unsafe class GptNeuralFramework : IDisposable
{
    public readonly GptConfig Config;
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

    public NeuralMatrix InputTokenIds;
    public NeuralMatrix ResidualStream;
    public NeuralMatrix LogitsOutput;

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

        TokenEmbeddings.RandomizeGaussian(0f, 0.02f);
        PositionalEmbeddings.RandomizeGaussian(0f, 0.02f);
        OutputProjection.RandomizeGaussian(0f, 0.02f);

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
        LogitsOutput = NeuralMatrix.GetOrCreate(MaxBatchSize * MaxSequenceLength, VocabSize);
    }

    public void SetBatchAndSequenceLimit(int batchSize, int seqLen)
    {
        if (batchSize > MaxBatchSize || seqLen > MaxSequenceLength)
            throw new ArgumentOutOfRangeException("Requested dimensions exceed pre-allocated maximum capacity.");

        CurrentBatchSize = batchSize;
        CurrentSeqLen = seqLen;

        InputTokenIds.SetRowSize(batchSize);
        ResidualStream.SetRowSize(batchSize * seqLen);
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

        for (int l = 0; l < NumLayers; l++)
        {
            Layers[l].Forward(ResidualStream, batchSize, seqLen);
        }

        LmHeadForwardInPlace(ResidualStream, LogitsOutput);
    }

    public void ForwardStep(int* nextTokenIds, int batchSize, int currentStepIndex)
    {
        SetBatchAndSequenceLimit(batchSize, 1);

        EmbeddingStepInPlace(nextTokenIds, ResidualStream, batchSize, currentStepIndex);

        for (int l = 0; l < NumLayers; l++)
        {
            Layers[l].ForwardStep(ResidualStream, batchSize, currentStepIndex);
        }

        LmHeadForwardInPlace(ResidualStream, LogitsOutput);
    }

    public void TrainStep(int[] batchChunk)
    {
        int seqLen = batchChunk.Length - 1;
        Span<int> inputIds = stackalloc int[seqLen];
        Span<int> targetIds = stackalloc int[seqLen];

        for (int i = 0; i < seqLen; i++)
        {
            inputIds[i] = batchChunk[i];
            targetIds[i] = batchChunk[i + 1];
        }

        fixed (int* pInputs = inputIds)
        {
            Forward(pInputs, 1, seqLen);
        }

        float lr = Config.LearningRate;
        float* pLogits = LogitsOutput.Pointer;
        float* pResidual = ResidualStream.Pointer;
        float* pOutProj = OutputProjection.Pointer;

        int logitsStride = LogitsOutput.ColumnsStride;
        int resStride = ResidualStream.ColumnsStride;
        int projStride = OutputProjection.ColumnsStride;

        for (int t = 0; t < seqLen; t++)
        {
            int target = targetIds[t];
            float* logitRow = pLogits + t * logitsStride;
            float* resRow = pResidual + t * resStride;

            float maxLogit = float.NegativeInfinity;
            for (int v = 0; v < VocabSize; v++)
            {
                if (logitRow[v] > maxLogit) maxLogit = logitRow[v];
            }

            float sumExp = 0f;
            for (int v = 0; v < VocabSize; v++)
            {
                logitRow[v] = MathF.Exp(logitRow[v] - maxLogit);
                sumExp += logitRow[v];
            }

            float invSum = sumExp > 0f ? 1.0f / sumExp : 0f;

            for (int v = 0; v < VocabSize; v++)
            {
                float prob = logitRow[v] * invSum;
                float grad = prob - (v == target ? 1.0f : 0.0f);

                float* weightCol = pOutProj + v;
                for (int d = 0; d < EmbedDim; d++)
                {
                    weightCol[d * projStride] -= lr * grad * resRow[d];
                }
            }
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

            // Get last token logits
            int lastTokenIdx = currentContextLen - 1;
            float* logits = LogitsOutput.Pointer + lastTokenIdx * LogitsOutput.ColumnsStride;

            // Apply Temperature
            float maxLogit = float.NegativeInfinity;
            for (int v = 0; v < VocabSize; v++)
            {
                logits[v] /= Math.Max(temperature, 1e-5f);
                if (logits[v] > maxLogit) maxLogit = logits[v];
            }

            // Softmax
            float sumExp = 0f;
            Span<float> probs = stackalloc float[VocabSize];
            for (int v = 0; v < VocabSize; v++)
            {
                probs[v] = MathF.Exp(logits[v] - maxLogit);
                sumExp += probs[v];
            }

            for (int v = 0; v < VocabSize; v++)
                probs[v] /= sumExp;

            // Top-P (Nucleus) Filtering & Sampling
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
        for (int i = 0; i < probs.Length; i++)
            sorted.Add((probs[i], i));

        sorted.Sort((a, b) => b.Prob.CompareTo(a.Prob));

        float cumSum = 0f;
        int cutoffIndex = sorted.Count - 1;
        for (int i = 0; i < sorted.Count; i++)
        {
            cumSum += sorted[i].Prob;
            if (cumSum >= topP)
            {
                cutoffIndex = i;
                break;
            }
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

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seq; s++)
            {
                int tokenId = tokens[b * seq + s];
                int tokenRowIndex = (tokenId >= 0 && tokenId < VocabSize) ? tokenId : 0;

                float* tokRow = pTok + (tokenRowIndex * tokStride);
                float* posRow = pPos + (s * posStride);
                float* dstRow = pRes + ((b * seq + s) * resStride);

                int i = 0;
                if (Avx2.IsSupported)
                {
                    int vecLimit = embedDim - (embedDim % 8);
                    for (; i < vecLimit; i += 8)
                    {
                        var vT = Vector256.Load(tokRow + i);
                        var vP = Vector256.Load(posRow + i);
                        (vT + vP).Store(dstRow + i);
                    }
                }

                for (; i < embedDim; i++)
                {
                    dstRow[i] = tokRow[i] + posRow[i];
                }
            }
        }
    }

    private void EmbeddingStepInPlace(int* tokens, NeuralMatrix residual, int batch, int stepIndex)
    {
        int embedDim = EmbedDim;
        int tokStride = TokenEmbeddings.ColumnsStride;
        int posStride = PositionalEmbeddings.ColumnsStride;
        int resStride = residual.ColumnsStride;

        float* pTok = TokenEmbeddings.Pointer;
        float* pPos = PositionalEmbeddings.Pointer + (stepIndex * posStride);
        float* pRes = residual.Pointer;

        for (int b = 0; b < batch; b++)
        {
            int tokenId = tokens[b];
            int tokenRowIndex = (tokenId >= 0 && tokenId < VocabSize) ? tokenId : 0;

            float* tokRow = pTok + (tokenRowIndex * tokStride);
            float* dstRow = pRes + (b * resStride);

            int i = 0;
            if (Avx2.IsSupported)
            {
                int vecLimit = embedDim - (embedDim % 8);
                for (; i < vecLimit; i += 8)
                {
                    var vT = Vector256.Load(tokRow + i);
                    var vP = Vector256.Load(pPos + i);
                    (vT + vP).Store(dstRow + i);
                }
            }

            for (; i < embedDim; i++)
            {
                dstRow[i] = tokRow[i] + pPos[i];
            }
        }
    }

    private void LmHeadForwardInPlace(NeuralMatrix inputStream, NeuralMatrix logitsOut)
    {
        inputStream.Dot(OutputProjection, logitsOut);
    }

    public void Dispose()
    {
        if (_disposed) return;

        TokenEmbeddings.Dispose();
        PositionalEmbeddings.Dispose();
        OutputProjection.Dispose();
        InputTokenIds.Dispose();
        ResidualStream.Dispose();
        LogitsOutput.Dispose();

        if (Layers != null)
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i]?.Dispose();
            }
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
