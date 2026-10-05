using System;
using System.Collections.Generic;
using System.IO;
using NeutralNET.Framework.Neural.GPT;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.GPT;

public class GptTrainingRunner
{
    public static List<int[]> CreateBatches(List<int> tokens, int contextSize)
    {
        List<int[]> batches = new();
        int stride = contextSize / 2;

        for (int i = 0; i <= tokens.Count - (contextSize + 1); i += stride)
        {
            int[] chunk = new int[contextSize + 1];
            tokens.CopyTo(i, chunk, 0, contextSize + 1);
            batches.Add(chunk);
        }

        return batches;
    }

    public static unsafe (float AvgLoss, float Accuracy) Evaluate(GptNeuralFramework gpt, List<int[]> testBatches)
    {
        float totalLoss = 0f;
        int correctPredictions = 0;
        int totalTokens = 0;

        foreach (var batch in testBatches)
        {
            int seqLen = batch.Length - 1;
            int[] inputs = batch[0..seqLen];
            int[] targets = batch[1..(seqLen + 1)];

            fixed (int* pInput = inputs)
            {
                gpt.Forward(pInput, 1, seqLen);
            }

            float* logitsPtr = gpt.LogitsOutput.Pointer;
            int logitsStride = gpt.LogitsOutput.ColumnsStride;
            int vocabSize = gpt.VocabSize;

            for (int t = 0; t < seqLen; t++)
            {
                float* logitRow = logitsPtr + (t * logitsStride);
                int target = targets[t];

                int bestToken = 0;
                float maxLogit = float.NegativeInfinity;

                for (int v = 0; v < vocabSize; v++)
                {
                    if (logitRow[v] > maxLogit)
                    {
                        maxLogit = logitRow[v];
                        bestToken = v;
                    }
                }

                if (bestToken == target)
                {
                    correctPredictions++;
                }

                float sumExp = 0f;
                for (int v = 0; v < vocabSize; v++)
                {
                    sumExp += MathF.Exp(logitRow[v] - maxLogit);
                }

                totalLoss += MathF.Log(sumExp) - (logitRow[target] - maxLogit);
                totalTokens++;
            }
        }

        float avgLoss = totalTokens > 0 ? totalLoss / totalTokens : 0f;
        float accuracy = totalTokens > 0 ? ((float)correctPredictions / totalTokens) * 100f : 0f;
        return (avgLoss, accuracy);
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

        WriteMatrix(gpt.TokenEmbeddings);
        WriteMatrix(gpt.PositionalEmbeddings);
        WriteMatrix(gpt.OutputProjection);

        for (int l = 0; l < gpt.NumLayers; l++)
        {
            var layer = gpt.Layers[l];
            WriteMatrix(layer.Norm1Scale);
            WriteMatrix(layer.Wq);
            WriteMatrix(layer.Wk);
            WriteMatrix(layer.Wv);
            WriteMatrix(layer.Wo);

            WriteMatrix(layer.Norm2Scale);
            WriteMatrix(layer.WGate);
            WriteMatrix(layer.WUp);
            WriteMatrix(layer.WDown);
        }

        Console.WriteLine($"[Checkpoint] Model successfully saved to: {filePath}");
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

        ReadMatrix(gpt.TokenEmbeddings);
        ReadMatrix(gpt.PositionalEmbeddings);
        ReadMatrix(gpt.OutputProjection);

        for (int l = 0; l < gpt.NumLayers; l++)
        {
            var layer = gpt.Layers[l];
            ReadMatrix(layer.Norm1Scale);
            ReadMatrix(layer.Wq);
            ReadMatrix(layer.Wk);
            ReadMatrix(layer.Wv);
            ReadMatrix(layer.Wo);

            ReadMatrix(layer.Norm2Scale);
            ReadMatrix(layer.WGate);
            ReadMatrix(layer.WUp);
            ReadMatrix(layer.WDown);
        }

        Console.WriteLine($"[Checkpoint] Model weights successfully loaded from: {filePath}");
    }
}
