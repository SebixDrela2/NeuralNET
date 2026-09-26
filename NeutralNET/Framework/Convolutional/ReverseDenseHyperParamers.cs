using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public sealed record class ReverseDenseHyperParamers(
    NeuralMatrix GradPre,
    NeuralMatrix DWeight,
    NeuralMatrix DBias,
    NeuralMatrix GradInput) : IDisposable
{
    public void Dispose()
    {
        GradPre.Dispose();
        DWeight.Dispose();
        DBias.Dispose();
        GradInput.Dispose();
    }
}
