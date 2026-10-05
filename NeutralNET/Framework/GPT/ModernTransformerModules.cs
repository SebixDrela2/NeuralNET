using System;
using System.Collections.Generic;
using System.Linq;

namespace NeutralNET.Framework.Neural.GPT;

public static class ModernTransformerModules
{
    // =========================================================================
    // 1. RMSNorm (Root Mean Square Normalization)
    // =========================================================================
    /// <summary>
    /// Normalizes input using RMSNorm across hidden dimension with learnable gamma weights.
    /// </summary>
    public static unsafe void ApplyRMSNorm(float* input, float* weight, float* output, int dim, float eps = 1e-5f)
    {
        float sumSquares = 0f;
        for (int i = 0; i < dim; i++)
        {
            sumSquares += input[i] * input[i];
        }

        float scale = 1.0f / MathF.Sqrt((sumSquares / dim) + eps);

        for (int i = 0; i < dim; i++)
        {
            output[i] = input[i] * scale * weight[i];
        }
    }

    // =========================================================================
    // 2. SwiGLU Activation Layer (Swish-Gated Linear Unit)
    // =========================================================================
    /// <summary>
    /// Applies SwiGLU activation: (x * Sigmoid(x)) * gate
    /// </summary>
    public static unsafe void ApplySwiGLU(float* gate, float* up, float* output, int dim)
    {
        for (int i = 0; i < dim; i++)
        {
            float g = gate[i];
            float swish = g * (1.0f / (1.0f + MathF.Exp(-g)));
            output[i] = swish * up[i];
        }
    }

    // =========================================================================
    // 3. RoPE (Rotary Position Embeddings)
    // =========================================================================
    /// <summary>
    /// Applies Rotary Position Embedding to Query or Key vectors for position m.
    /// Operates on 2D pairs (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    /// </summary>
    public static unsafe void ApplyRoPE(float* vec, int pos, int headDim, float thetaBase = 10000.0f)
    {
        for (int i = 0; i < headDim; i += 2)
        {
            float freq = 1.0f / MathF.Pow(thetaBase, (float)i / headDim);
            float val = pos * freq;
            float cos = MathF.Cos(val);
            float sin = MathF.Sin(val);

            float x1 = vec[i];
            float x2 = vec[i + 1];

            vec[i] = x1 * cos - x2 * sin;
            vec[i + 1] = x1 * sin + x2 * cos;
        }
    }

    // =========================================================================
    // 4. Advanced Token Sampling (Temperature & Top-P / Nucleus Sampling)
    // =========================================================================
    /// <summary>
    /// Samples a token index from raw logits using Temperature scaling and Top-P (Nucleus) filtering.
    /// </summary>
    public static int SampleTopP(float[] logits, float temperature = 0.8f, float topP = 0.9f, Random? rng = null)
    {
        rng ??= Random.Shared;

        int vocabSize = logits.Length;
        float[] scaledLogits = new float[vocabSize];

        // 1. Apply Temperature Scaling
        float maxLogit = float.NegativeInfinity;
        for (int i = 0; i < vocabSize; i++)
        {
            scaledLogits[i] = logits[i] / Math.Max(temperature, 1e-5f);
            if (scaledLogits[i] > maxLogit) maxLogit = scaledLogits[i];
        }

        // 2. Softmax
        float expSum = 0f;
        float[] probs = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            probs[i] = MathF.Exp(scaledLogits[i] - maxLogit);
            expSum += probs[i];
        }
        for (int i = 0; i < vocabSize; i++)
        {
            probs[i] /= expSum;
        }

        // 3. Sort probabilities for Top-P nucleus dynamic thresholding
        var sortedTokens = probs
            .Select((p, idx) => new { TokenId = idx, Prob = p })
            .OrderByDescending(x => x.Prob)
            .ToList();

        // 4. Truncate to cumulative probability threshold p
        float cumSum = 0f;
        List<(int TokenId, float Prob)> nucleus = new();

        foreach (var token in sortedTokens)
        {
            nucleus.Add((token.TokenId, token.Prob));
            cumSum += token.Prob;
            if (cumSum >= topP) break;
        }

        // 5. Renormalize nucleus and sample probabilistically
        float totalNucleusProb = nucleus.Sum(x => x.Prob);
        float r = (float)rng.NextDouble() * totalNucleusProb;

        float currentAccum = 0f;
        foreach (var item in nucleus)
        {
            currentAccum += item.Prob;
            if (r <= currentAccum)
            {
                return item.TokenId;
            }
        }

        return nucleus[0].TokenId;
    }
}
