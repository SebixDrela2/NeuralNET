using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public record class AdamHyperLayerParameters(
    NeuralMatrix MWeights,
    NeuralMatrix VWeights,
    NeuralMatrix MBiases,
    NeuralMatrix VBiases) : IDisposable
{
    public void Dispose()
    {
        MWeights.Dispose();
        VWeights.Dispose();
        MBiases.Dispose();
        VBiases.Dispose();
    }
}
