using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public sealed class BatchNormParams : IDisposable
{
    public int Channels;

    // ---- Per-channel parameters (shape 1 × C) ----
    public NeuralMatrix Gamma;
    public NeuralMatrix Beta;

    // ---- Running statistics for inference (shape 1 × C) ----
    public NeuralMatrix RunningMean;
    public NeuralMatrix RunningVar;

    // ---- Batch statistics cached from the forward pass (shape 1 × C) ----
    public NeuralMatrix Mean;
    public NeuralMatrix InvStd;

    // ---- Gradients accumulated in the backward pass (shape 1 × C) ----
    public NeuralMatrix GradGamma;
    public NeuralMatrix GradBeta;

    // ---- AdamW moment estimates for Gamma and Beta (shape 1 × C) ----
    public NeuralMatrix MGamma;
    public NeuralMatrix VGamma;
    public NeuralMatrix MBeta;
    public NeuralMatrix VBeta;

    // ---- Spatial buffers (shape B × C × H × W) ----
    public CnnMatrix Normalized;   // x̂ = (x - mean) * invStd
    public CnnMatrix Output;       // γ * x̂ + β, i.e. the activation input
    public CnnMatrix GradInput;    // dL/d(BN input), i.e. dL/d(conv preAct)

    // ---- Config ----
    public float Momentum;
    public float Epsilon;

    // ---- AdamW step state ----
    public int T;
    public float B1Pow = 1.0f;
    public float B2Pow = 1.0f;

    /// <summary>
    /// Initializes Gamma=1, Beta=0, RunningMean=0, RunningVar=1, all moments=0.
    /// Call after allocating the buffers.
    /// </summary>
    public unsafe void Init()
    {
        var pG = Gamma.Pointer;
        var pB = Beta.Pointer;
        var pRM = RunningMean.Pointer;
        var pRV = RunningVar.Pointer;
        var pMG = MGamma.Pointer;
        var pVG = VGamma.Pointer;
        var pMB = MBeta.Pointer;
        var pVB = VBeta.Pointer;

        for (int c = 0; c < Channels; c++)
        {
            pG[c] = 1.0f;
            pB[c] = 0.0f;
            pRM[c] = 0.0f;
            pRV[c] = 1.0f;
            pMG[c] = 0.0f;
            pVG[c] = 0.0f;
            pMB[c] = 0.0f;
            pVB[c] = 0.0f;
        }

        T = 0;
        B1Pow = 1.0f;
        B2Pow = 1.0f;
    }

    /// <summary>
    /// AdamW update for Gamma and Beta from their accumulated gradients.
    /// Contiguous 1D update over C channels — no strided access, safe for
    /// 1×C buffers. Weight decay applies to Gamma only (matching PyTorch).
    /// Advances this layer's own step counter, so bias correction is per-layer.
    /// </summary>
    public unsafe void Step(
        float learningRate,
        float weightDecay,
        float beta1,
        float beta2,
        float epsilon)
    {
        int C = Channels;

        T++;
        B1Pow *= beta1;
        B2Pow *= beta2;
        float c_m = 1.0f / (1.0f - B1Pow);
        float c_v = 1.0f / (1.0f - B2Pow);
        float omb1 = 1.0f - beta1;
        float omb2 = 1.0f - beta2;

        float* pG = Gamma.Pointer;
        float* pB = Beta.Pointer;
        float* pGG = GradGamma.Pointer;
        float* pGB = GradBeta.Pointer;
        float* pMG = MGamma.Pointer;
        float* pVG = VGamma.Pointer;
        float* pMB = MBeta.Pointer;
        float* pVB = VBeta.Pointer;

        for (int c = 0; c < C; c++)
        {
            // ---- Gamma (weights): AdamW with decoupled weight decay ----
            float gGrad = pGG[c];
            float mG = beta1 * pMG[c] + omb1 * gGrad;
            float vG = beta2 * pVG[c] + omb2 * gGrad * gGrad;
            pMG[c] = mG;
            pVG[c] = vG;

            float mHatG = mG * c_m;
            float vHatG = vG * c_v;
            float stepG = learningRate * mHatG / (MathF.Sqrt(vHatG) + epsilon);

            pG[c] -= stepG + learningRate * weightDecay * pG[c];

            // ---- Beta (biases): no weight decay ----
            float gGradB = pGB[c];
            float mB = beta1 * pMB[c] + omb1 * gGradB;
            float vB = beta2 * pVB[c] + omb2 * gGradB * gGradB;
            pMB[c] = mB;
            pVB[c] = vB;

            float mHatB = mB * c_m;
            float vHatB = vB * c_v;
            float stepB = learningRate * mHatB / (MathF.Sqrt(vHatB) + epsilon);

            pB[c] -= stepB;
        }
    }

    public void Dispose()
    {
        Gamma?.Dispose();
        Beta?.Dispose();
        RunningMean?.Dispose();
        RunningVar?.Dispose();
        Mean?.Dispose();
        InvStd?.Dispose();
        GradGamma?.Dispose();
        GradBeta?.Dispose();
        MGamma?.Dispose();
        VGamma?.Dispose();
        MBeta?.Dispose();
        VBeta?.Dispose();
        Normalized?.Dispose();
        Output?.Dispose();
        GradInput?.Dispose();
    }
}
