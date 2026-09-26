using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public sealed record class ReverseDenseHyperParamers(
    NeuralMatrix GradPre,
    NeuralMatrix DWeights,
    NeuralMatrix DBiases,
    NeuralMatrix GradInput) : IDisposable
{
    public void Dispose()
    {
        GradPre.Dispose();
        DWeights.Dispose();
        DBiases.Dispose();
        GradInput.Dispose();
    }
}
