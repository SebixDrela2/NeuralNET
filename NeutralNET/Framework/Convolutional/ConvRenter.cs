using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public class ConvRenter : NeuralRenter
{
    public static NeuralMatrix GetPoolIndices(CnnMatrix postAct, int poolSize)
    {
        int batch = postAct.Batch;
        int channels = postAct.Channels;
        int inH = postAct.Height;
        int inW = postAct.Width;

        int outH = inH / poolSize;
        int outW = inW / poolSize;

        return RentNeural(batch * channels * outH * outW, 1);
    }

    public static NeuralMatrix GetColInput(CnnSize input, int kernelH, int kernelW, int stride, int padding)
    {
        int paddedH = input.Height + 2 * padding;
        int paddedW = input.Width + 2 * padding;

        int outH = (paddedH - kernelH) / stride + 1;
        int outW = (paddedW - kernelW) / stride + 1;
        int patchSize = input.Channels * kernelH * kernelW;
        int totalPatches = input.BatchSize * outH * outW;

        return RentNeural(totalPatches, patchSize);
    }

    public static CnnMatrix GetInput(CnnMatrix postAct, int poolSize)
    {
        int batch = postAct.Batch;
        int channels = postAct.Channels;
        int inH = postAct.Height;
        int inW = postAct.Width;

        int outH = inH / poolSize;
        int outW = inW / poolSize;

        return RentCnn(batch, channels, outH, outW);
    }

    public static NeuralMatrix GetPreGradMatrix(CnnMatrix preGrad)
    {
        int outH = preGrad.Height;
        int outW = preGrad.Width;
        int patches = preGrad.Batch * outH * outW;
        int filters = preGrad.Channels;

        var preGradMatrix = RentNeural(patches, filters);

        return preGradMatrix;
    }

    public static NeuralMatrix GetDWeights(NeuralMatrix colInput, CnnMatrix preGrad)
    {
        var filters = preGrad.Channels;
        var inDim = colInput.UsedColumns;

        return CnnNeuralFramework.EnableGpu
            ? RentNeural(filters, inDim)
            : RentNeural(inDim, filters);
    }

    public static CnnMatrix GetPooled(CnnLayerConfig layer, CnnMatrix preAct)
    {
        int batch = preAct.Batch;
        int channels = preAct.Channels;
        int inH = preAct.Height;
        int inW = preAct.Width;

        int outH = inH / layer.PoolSize;
        int outW = inW / layer.PoolSize;

        var pooled = RentCnn(batch, channels, outH, outW);
        return pooled;
    }
}
