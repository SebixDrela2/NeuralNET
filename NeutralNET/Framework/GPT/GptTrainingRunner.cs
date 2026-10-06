using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using NeutralNET.Framework.Neural.GPT;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.GPT;

public class GptTrainingRunner
{
    private const int CheckpointMagic = 0x47505432;  // 'GPT2'

    public static unsafe (float AvgLoss, float Accuracy) Evaluate(GptNeuralFramework gpt, List<int[][]> testMiniBatches)
    {
        double totalLoss = 0.0;
        int correctPredictions = 0;
        int totalTokens = 0;
        int skippedTokens = 0;
        int outOfRangeTargets = 0;

        int batchCounter = 0;
        int totalBatches = testMiniBatches.Count;

        foreach (var miniBatch in testMiniBatches)
        {
            batchCounter++;
            int batchSize = miniBatch.Length;
            int seqLen = miniBatch[0].Length - 1;
            int totalBatchTokens = batchSize * seqLen;

            int[] inputIdsArray = ArrayPool<int>.Shared.Rent(totalBatchTokens);
            int[] targetIdsArray = ArrayPool<int>.Shared.Rent(totalBatchTokens);

            try
            {
                Span<int> inputIds = inputIdsArray.AsSpan(0, totalBatchTokens);
                Span<int> targetIds = targetIdsArray.AsSpan(0, totalBatchTokens);

                for (int b = 0; b < batchSize; b++)
                {
                    for (int t = 0; t < seqLen; t++)
                    {
                        int idx = b * seqLen + t;
                        inputIds[idx] = miniBatch[b][t];
                        targetIds[idx] = miniBatch[b][t + 1];
                    }
                }

                fixed (int* pInput = inputIds)
                {
                    gpt.Forward(pInput, batchSize, seqLen);
                }

                float* logitsPtr = gpt.LogitsOutput.Pointer;
                int logitsStride = gpt.LogitsOutput.ColumnsStride;
                int vocabSize = gpt.VocabSize;
                int rows = gpt.LogitsOutput.Rows;

                // per-batch diagnostic
                {
                    float mn = float.PositiveInfinity;
                    float mx = float.NegativeInfinity;
                    double sum = 0.0;
                    long count = 0;
                    long nan = 0;
                    for (int r = 0; r < rows; r++)
                    {
                        float* row = logitsPtr + r * logitsStride;
                        for (int v = 0; v < vocabSize; v++)
                        {
                            float val = row[v];
                            if (float.IsNaN(val) || float.IsInfinity(val)) { nan++; continue; }
                            if (val < mn) mn = val;
                            if (val > mx) mx = val;
                            sum += val;
                            count++;
                        }
                    }
                    double mean = count > 0 ? sum / count : 0.0;
                    Console.WriteLine($"[evaluate-diag] batch {batchCounter}/{totalBatches} rows={rows} vocab={vocabSize} stride={logitsStride} |min={mn:G4} max={mx:G4} mean={mean:G4} nan={nan}|");
                }

                int batchCorrect = 0;
                int batchTotal = 0;
                double batchLossSum = 0.0;

                for (int i = 0; i < totalBatchTokens; i++)
                {
                    float* logitRow = logitsPtr + (i * logitsStride);
                    int target = targetIds[i];

                    if (target < 0 || target >= vocabSize)
                    {
                        outOfRangeTargets++;
                        continue;
                    }

                    float maxLogit = float.NegativeInfinity;
                    int bestToken = 0;

                    for (int v = 0; v < vocabSize; v++)
                    {
                        float logitVal = logitRow[v];
                        if (logitVal > maxLogit)
                        {
                            maxLogit = logitVal;
                            bestToken = v;
                        }
                    }

                    if (float.IsNaN(maxLogit) || float.IsInfinity(maxLogit))
                    {
                        skippedTokens++;
                        continue;
                    }

                    totalTokens++;
                    batchTotal++;

                    if (bestToken == target)
                    {
                        correctPredictions++;
                        batchCorrect++;
                    }

                    float sumExp = 0f;
                    for (int v = 0; v < vocabSize; v++)
                        sumExp += MathF.Exp(logitRow[v] - maxLogit);

                    float targetLogit = logitRow[target];
                    float tokenLoss = MathF.Log(MathF.Max(sumExp, 1e-7f)) - (targetLogit - maxLogit);

                    if (float.IsNaN(tokenLoss) || float.IsInfinity(tokenLoss))
                    {
                        totalTokens--;
                        batchTotal--;
                        if (bestToken == target)
                        {
                            correctPredictions--;
                            batchCorrect--;
                        }
                        skippedTokens++;
                        continue;
                    }

                    totalLoss += tokenLoss;
                    batchLossSum += tokenLoss;
                }

                double batchAcc = batchTotal > 0 ? (double)batchCorrect / batchTotal * 100.0 : 0.0;
                double batchAvgLoss = batchTotal > 0 ? batchLossSum / batchTotal : 0.0;
                Console.WriteLine($"[evaluate-diag]   batch loss={batchAvgLoss:F4} correct={batchCorrect}/{batchTotal} ({batchAcc:F3}%)");
            }
            finally
            {
                ArrayPool<int>.Shared.Return(inputIdsArray);
                ArrayPool<int>.Shared.Return(targetIdsArray);
            }
        }

        double finalAcc = totalTokens > 0 ? (double)correctPredictions / totalTokens * 100.0 : 0.0;
        double finalLoss = totalTokens > 0 ? totalLoss / totalTokens : 0.0;

        Console.WriteLine($"[evaluate-diag] FINAL: loss={finalLoss:F4} correct={correctPredictions} total={totalTokens} acc={finalAcc:F4}% skipped={skippedTokens} outOfRangeTargets={outOfRangeTargets}");

        return ((float)finalLoss, (float)finalAcc);
    }

    public static unsafe void SaveModel(GptNeuralFramework gpt, string filePath)
    {
        string? directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        using var stream = File.Create(filePath);
        using var writer = new BinaryWriter(stream);

        void WriteMatrix(NeuralMatrix mat)
        {
            writer.Write(mat.Rows);
            writer.Write(mat.UsedColumns);
            float* ptr = mat.Pointer;
            int stride = mat.ColumnsStride;

            for (int r = 0; r < mat.Rows; r++)
            {
                float* row = ptr + (r * stride);
                for (int c = 0; c < mat.UsedColumns; c++)
                {
                    writer.Write(row[c]);
                }
            }
        }

        // Header
        writer.Write(CheckpointMagic);
        writer.Write(gpt.StepCount);

        // Embeddings (weights + Adam m/v)
        WriteMatrix(gpt.TokenEmbeddings);
        WriteMatrix(gpt.DebugMTokEmb);
        WriteMatrix(gpt.DebugVTokEmb);

        WriteMatrix(gpt.OutputProjection);
        WriteMatrix(gpt.DebugMOutProj);
        WriteMatrix(gpt.DebugVOutProj);

        // Layers
        for (int l = 0; l < gpt.NumLayers; l++)
        {
            var layer = gpt.Layers[l];

            WriteMatrix(layer.Norm1Scale);
            WriteMatrix(layer.DebugMNorm1);
            WriteMatrix(layer.DebugVNorm1);

            WriteMatrix(layer.Wq);
            WriteMatrix(layer.DebugMWq);
            WriteMatrix(layer.DebugVWq);

            WriteMatrix(layer.Wk);
            WriteMatrix(layer.DebugMWk);
            WriteMatrix(layer.DebugVWk);

            WriteMatrix(layer.Wv);
            WriteMatrix(layer.DebugMWv);
            WriteMatrix(layer.DebugVWv);

            WriteMatrix(layer.Wo);
            WriteMatrix(layer.DebugMWo);
            WriteMatrix(layer.DebugVWo);

            WriteMatrix(layer.Norm2Scale);
            WriteMatrix(layer.DebugMNorm2);
            WriteMatrix(layer.DebugVNorm2);

            WriteMatrix(layer.WGate);
            WriteMatrix(layer.DebugMWGate);
            WriteMatrix(layer.DebugVWGate);

            WriteMatrix(layer.WUp);
            WriteMatrix(layer.DebugMWUp);
            WriteMatrix(layer.DebugVWUp);

            WriteMatrix(layer.WDown);
            WriteMatrix(layer.DebugMWDown);
            WriteMatrix(layer.DebugVWDown);
        }

        Console.WriteLine($"[Checkpoint] Model (with optimizer state) saved to: {filePath}");
    }

    /// <summary>
    /// Non-throwing wrapper around <see cref="LoadModel"/>.
    /// Returns false if the file doesn't exist or the format is invalid.
    /// </summary>
    public static unsafe bool TryLoadModel(GptNeuralFramework gpt, string filePath, out string message)
    {
        if (!File.Exists(filePath))
        {
            message = $"[Checkpoint] No checkpoint at {filePath}";
            return false;
        }

        try
        {
            LoadModel(gpt, filePath);
            message = $"[Checkpoint] Loaded {filePath}";
            return true;
        }
        catch (Exception ex)
        {
            message = $"[Checkpoint] Failed to load {filePath}: {ex.Message}";
            return false;
        }
    }

    public static unsafe void LoadModel(GptNeuralFramework gpt, string filePath)
    {
        if (!File.Exists(filePath)) throw new FileNotFoundException("Checkpoint not found!");

        using var stream = File.OpenRead(filePath);
        using var reader = new BinaryReader(stream);

        void ReadMatrix(NeuralMatrix mat)
        {
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();

            if (rows != mat.Rows || cols != mat.UsedColumns)
            {
                throw new InvalidOperationException($"Matrix shape mismatch in checkpoint. Expected ({mat.Rows}, {mat.UsedColumns}), got ({rows}, {cols}).");
            }

            float* ptr = mat.Pointer;
            int stride = mat.ColumnsStride;

            for (int r = 0; r < mat.Rows; r++)
            {
                float* row = ptr + (r * stride);
                for (int c = 0; c < mat.UsedColumns; c++)
                {
                    row[c] = reader.ReadSingle();
                }
            }
        }

        // Header
        int magic = reader.ReadInt32();
        if (magic != CheckpointMagic)
        {
            throw new InvalidOperationException(
                $"Checkpoint format mismatch (got 0x{magic:X8}, expected 0x{CheckpointMagic:X8}). " +
                "This checkpoint was saved with an older format. Delete or rename it and retrain, or downgrade to the previous code.");
        }
        gpt.StepCount = reader.ReadInt32();

        // Embeddings
        ReadMatrix(gpt.TokenEmbeddings);
        ReadMatrix(gpt.DebugMTokEmb);
        ReadMatrix(gpt.DebugVTokEmb);

        ReadMatrix(gpt.OutputProjection);
        ReadMatrix(gpt.DebugMOutProj);
        ReadMatrix(gpt.DebugVOutProj);

        // Layers
        for (int l = 0; l < gpt.NumLayers; l++)
        {
            var layer = gpt.Layers[l];

            ReadMatrix(layer.Norm1Scale);
            ReadMatrix(layer.DebugMNorm1);
            ReadMatrix(layer.DebugVNorm1);

            ReadMatrix(layer.Wq);
            ReadMatrix(layer.DebugMWq);
            ReadMatrix(layer.DebugVWq);

            ReadMatrix(layer.Wk);
            ReadMatrix(layer.DebugMWk);
            ReadMatrix(layer.DebugVWk);

            ReadMatrix(layer.Wv);
            ReadMatrix(layer.DebugMWv);
            ReadMatrix(layer.DebugVWv);

            ReadMatrix(layer.Wo);
            ReadMatrix(layer.DebugMWo);
            ReadMatrix(layer.DebugVWo);

            ReadMatrix(layer.Norm2Scale);
            ReadMatrix(layer.DebugMNorm2);
            ReadMatrix(layer.DebugVNorm2);

            ReadMatrix(layer.WGate);
            ReadMatrix(layer.DebugMWGate);
            ReadMatrix(layer.DebugVWGate);

            ReadMatrix(layer.WUp);
            ReadMatrix(layer.DebugMWUp);
            ReadMatrix(layer.DebugVWUp);

            ReadMatrix(layer.WDown);
            ReadMatrix(layer.DebugMWDown);
            ReadMatrix(layer.DebugVWDown);
        }

        Console.WriteLine($"[Checkpoint] Model (with optimizer state) loaded from: {filePath}");
    }
}
